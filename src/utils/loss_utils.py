import os
import math
import json
import numpy as np
import torch
import torch_scatter
import frnn
import cv2
from .conventions import get_mapping, SMPLX_55, SMPLX_118
from utils.data_utils import data_to_device, transform_human_gs, transform_object_gs

from configs.config import cfg

with open(f"{cfg.data.paths.assets_root}/body_models/smplx/smplx_vert_segmentation.json") as f:
    smplx_vert_segmentation = json.load(f)
    smplx_vert_segmentation = data_to_device(smplx_vert_segmentation)

def get_collision_loss(pred_kpt_3d, object_verts, sample_num=1000):
    B, M, _3 = object_verts.shape
    lefthand_verts = pred_kpt_3d[:B, smplx_vert_segmentation['leftHand'].long()]
    righthand_verts = pred_kpt_3d[:B, smplx_vert_segmentation['rightHand'].long()]
    device = lefthand_verts.device

    # Sample object points
    sample_idx = torch.randint(0, M, (sample_num,), device=device)
    sampled_obj = object_verts[:, sample_idx]  # (B, sample_num, 3)

    # Concatenate hand points
    hand_verts = torch.cat([lefthand_verts, righthand_verts], dim=1)  # (B, H, 3)

    # Broadcast compute pairwise distances
    hand_exp = hand_verts.unsqueeze(2)  # (B, H, 1, 3)
    obj_exp = sampled_obj.unsqueeze(1)  # (B, 1, sample_num, 3)

    dists = torch.norm(hand_exp - obj_exp, dim=-1)  # (B, H, sample_num)
    threshold = 0.02 # 1cm
    collision = torch.nn.functional.relu(threshold - dists)
    min_penalties = collision.max(dim=-1)[0]
    return min_penalties.mean(-1).sum()


def get_still_dict(still_mask):
    still_dict = {}
    stilling = False
    last_flag = False
    for i in range(len(still_mask)):
        this_flag = still_mask[i]
        if this_flag and not last_flag:
            stilling = True
            this_stilling_list = []
            this_stilling_list_from = i
        if last_flag and not this_flag:
            stilling = False
            still_dict[this_stilling_list_from] = this_stilling_list
            
        if stilling:
            this_stilling_list.append(i)
        last_flag = this_flag
        
    if last_flag == True:
        still_dict[this_stilling_list_from] = this_stilling_list
        
    return still_dict

def get_foot_sliding_loss(pred_kpt_3d, pgt_depth_map, contact_conf, camera):
    # contact_conf: [L_Ankle, L_foot, R_Ankle, R_foot]
    left_foot_verts = pred_kpt_3d[:, smplx_vert_segmentation['leftFoot'].long()].mean(1)
    right_foot_verts = pred_kpt_3d[:, smplx_vert_segmentation['rightFoot'].long()].mean(1)

    const_mask = torch.cat((contact_conf[:, 0:2].mean(-1, keepdims=True) > 0.5, contact_conf[:, 2:4].mean(-1, keepdims=True) > 0.5), dim=-1).float() # (B, 2)

    left_still_dict = get_still_dict(const_mask[:, 0])
    right_still_dict = get_still_dict(const_mask[:, 1])
    loss = 0
    for k,v in left_still_dict.items():
        loss = loss + torch.norm(left_foot_verts[list(v)] - left_foot_verts.detach()[list(v)].mean(0, keepdims=True), dim=-1, p=1).sum()
    for k,v in right_still_dict.items():
        loss = loss + torch.norm(right_foot_verts[list(v)] - right_foot_verts.detach()[list(v)].mean(0, keepdims=True), dim=-1, p=1).sum()

    return loss.mean()
    
def get_3dkpt_loss(pred_kpt_3d, pgt_kpt_3d):
    bs = pgt_kpt_3d.shape[0]
    mapping_list = get_mapping(SMPLX_118, SMPLX_55)
    pgt_kpt_3d_55 = torch.zeros((bs, 55, 4)).to(pgt_kpt_3d)
    pgt_kpt_3d_55[:, mapping_list[0]] = pgt_kpt_3d[:, mapping_list[1]]
    confidience = pgt_kpt_3d_55[..., 3]
    pgt_kpt_3d = pgt_kpt_3d_55[..., :3]
    loss = torch.norm(pred_kpt_3d[:bs] - pgt_kpt_3d, dim=-1) * confidience
    return loss.mean()

def get_smooth_loss(params, keys, weights):
    loss = 0
    for key, weight in zip(keys, weights):
        loss = loss + weight * torch.norm(params[key][1:] - params[key][:-1], dim=-1).mean()
    return loss

K = 400
r = 4
    
def get_src_tgt_indexes(frame_id):
    target_pose_index = frame_id
    src_pose_index = target_pose_index

    min_tracking_interval = 10
    max_tracking_interval = 14
    if target_pose_index == 0:
        min_tracking_interval = 0
        max_tracking_interval = 0
    elif target_pose_index < min_tracking_interval:
        min_tracking_interval = 1
        max_tracking_interval = target_pose_index
    else:
        min_tracking_interval = 1
    while abs(src_pose_index - target_pose_index) < min_tracking_interval:
        src_pose_index = np.random.randint(-max_tracking_interval, -min_tracking_interval+1) + target_pose_index
        src_pose_index = max(src_pose_index, 0)
    return src_pose_index, target_pose_index
            
def get_track_2d_loss(pgt_tracks_2d, frame_id, camera, object_gaussian, object_gs_cano, object_scale, object_rotations, object_offsets, object_R_params, object_T_params, object_W_params, gt_object_masks, gt_images=None):
    src_id, tgt_id = get_src_tgt_indexes(frame_id)
    if src_id == tgt_id:
        return torch.zeros(1).float().cuda(), None, None
    
    src_particles = pgt_tracks_2d[src_id][src_id]
    target_particles = pgt_tracks_2d[src_id][tgt_id]
    
    particle_mask = (src_particles[:, 2] > 0) & (target_particles[:, 2] > 0)
    src_particles = src_particles[particle_mask, :2]
    target_particles = target_particles[particle_mask, :2]
    
    particle_object_mask = gt_object_masks[src_id][src_particles[:, 1].long(), src_particles[:, 0].long()] > 0.5
    src_particles = src_particles[particle_object_mask]
    target_particles = target_particles[particle_object_mask]
    
    flow2d = src_particles - target_particles
    dynamic_mask = (flow2d[:, 0] ** 2 + flow2d[:, 1] ** 2) > 10
    target_particles[~dynamic_mask] = src_particles[~dynamic_mask]
    if dynamic_mask.sum() < 10:
        return torch.zeros(1).float().cuda(), None, None
    
    # get means and depths for the target frame
    target_gs = object_gaussian({
        "art_R": object_R_params[tgt_id:tgt_id+1],
        "art_T": object_T_params[tgt_id:tgt_id+1],
        "art_W": object_W_params[None]
    }, object_gs_cano)
    target_gs = transform_object_gs(target_gs, object_scale, object_rotations, object_offsets)
    target_means3D = target_gs['means3D'][0]
    scaling = target_gs['scales']
    opacities = target_gs['opacities']
    
    means2D_target, depths_target = get_means_2D(xyz=target_means3D, scales=scaling, opacity=opacities, camera=camera, return_depths=True)
    # get means and depths for the source frame
    src_gs = object_gaussian({
        "art_R": object_R_params[src_id:src_id+1],
        "art_T": object_T_params[src_id:src_id+1],
        "art_W": object_W_params[None]
    }, object_gs_cano)
    src_gs = transform_object_gs(src_gs, object_scale, object_rotations, object_offsets)
    src_means3D = src_gs['means3D'][0]
    
    means2D_src, depths_src, conical_opacity = get_means_2D(xyz=src_means3D, scales=scaling, opacity=opacities, camera=camera, return_depths=True, return_conical_opacity=True)
    
    means2D_src = means2D_src.detach()
    depths_src = depths_src.detach().flatten()
    conical_opacity = conical_opacity.detach()

    # append segmentation categories if we want to apply loss per-instance category
    means2D_src_wseg = means2D_src.clone()
    src_particles_wseg = src_particles.clone()

    # get KNN closest gaussians for each tracked particle
    device = means2D_src.device
    lengths1 = torch.tensor([src_particles_wseg.size(0)]).to(device)
    lengths2 = torch.tensor([means2D_src_wseg.size(0)]).to(device)
    dists2, raw_knn_idxs, nn, grid = \
        frnn.frnn_grid_points(src_particles_wseg.unsqueeze(0), means2D_src_wseg.unsqueeze(0), lengths1, lengths2,
                                K, r, grid=None, return_nn=False, return_sorted=True)
    
    # gather source depths by knn, and then densely order them
    src_particles_gather, means2D_src_gather, depths_src_gather, conical_opacity, found_mask, knn_idxs, particles_idxs = \
        _gather_knn(raw_knn_idxs, depths_src, src_particles, means2D_src, conical_opacity, K=K)

    # compute per-particle influences and distances in the source frame
    dists = (means2D_src_gather - src_particles_gather)  # MK x 2
    influences, dists = _accumulate_opacity_into_influence(particles_idxs, dists, conical_opacity, depths_src_gather, found_mask)
    
    dists_source = dists

    # get particles and means2D in target frame
    target_particles_gather = torch.gather(target_particles, 0, torch.stack([particles_idxs.flatten(), particles_idxs.flatten()], dim=-1)).reshape(-1, 2)  # MK' x 2
    means2D_target_gather = torch.gather(means2D_target, 0, torch.stack([knn_idxs.flatten(), knn_idxs.flatten()], dim=-1)).reshape(-1, 2)  # MK' x 2
    depths_target_gather = torch.gather(depths_target.flatten(), 0, knn_idxs.flatten()).flatten()
    # dists_target = torch.norm(means2D_target_gather - target_particles_gather, dim=-1)
    dists_target = (means2D_target_gather - target_particles_gather)
    dists_target[torch.logical_not(found_mask)] = 0
    
    per_gs_influences = torch_scatter.scatter_sum(influences[found_mask], index=knn_idxs[found_mask], dim_size=means2D_target.shape[0])
    per_gs_dists = torch_scatter.scatter_mean(torch.norm(src_particles_gather - target_particles_gather, p=2, dim=-1)[found_mask], index=knn_idxs[found_mask], dim_size=means2D_target.shape[0])
    static_gs_mask = (per_gs_influences > 0.5) & (per_gs_dists < 0.5)
    dynamic_gs_mask = (per_gs_influences > 0.5) & (per_gs_dists > 3.0)
    raw_errors = torch.norm(dists_target - dists_source.detach(), dim=-1) * influences
    depth_norms = torch.abs(depths_target_gather.detach() + depths_src_gather) / 2
    errors = raw_errors / torch.clip(depth_norms, min=0.1)
    particles_idxs_contiguous = torch.unique(particles_idxs, return_inverse=True)[1]
    errors = torch_scatter.scatter_sum(errors[found_mask], index=particles_idxs_contiguous[found_mask])

    return torch.mean(errors), dynamic_gs_mask, static_gs_mask

@torch.no_grad()
def get_init_dynamic_data(pgt_tracks_2d, N_frames, camera, object_gaussian, object_gs_cano, object_scale, object_rotations, object_offsets, object_R_params, object_T_params, object_W_params, gt_object_masks, gt_images=None):
    moving = False
    max_flow_mean = 0
    ret = {}
    for src_id in range(0, N_frames - 31):
        tgt_id = src_id + 30
        src_particles = pgt_tracks_2d[src_id][src_id]
        target_particles = pgt_tracks_2d[src_id][tgt_id]
            
        particle_mask = (src_particles[:, 2] > 0) & (target_particles[:, 2] > 0)
        src_particles = src_particles[particle_mask, :2]
        target_particles = target_particles[particle_mask, :2]
        
        particle_object_mask = gt_object_masks[src_id][src_particles[:, 1].long(), src_particles[:, 0].long()] > 0.5
        src_particles = src_particles[particle_object_mask]
        target_particles = target_particles[particle_object_mask]
    
        flow2d = (target_particles - src_particles)
        flow_mean = torch.norm(flow2d.mean(0), p=2)
        if flow_mean > 4.0:
            moving = True
            print("Flows Moving {}->{}".format(src_id, tgt_id))
            break
        max_flow_mean = max(max_flow_mean, flow_mean)

    if not moving:
        assert False, "Not Moving, max_flow_mean = {}".format(max_flow_mean)
    ret.update({
        "src_particles": src_particles,
        "tgt_particles": target_particles,
        "start_moving_id": src_id
    })
        
    ####### get dynamic_mask 2d
    flow2d_dynamic_mask = (torch.norm((target_particles - src_particles), p=2, dim=1) > 5)
    dynamic_prompts_2d = src_particles[flow2d_dynamic_mask].long()
    #######
    ####### get static_mask 2d
    flow2d_static_mask = (torch.norm((target_particles - src_particles), p=2, dim=1) < 2)
    static_prompts_2d = src_particles[flow2d_static_mask].long()
    ret.update({
        "dynamic_prompts_2d": dynamic_prompts_2d,
        "static_prompts_2d": static_prompts_2d,
    })
    return ret

def get_means_2D(xyz, camera, scales=None, opacity=None, return_depths=False, return_conical_opacity=False):
    full_proj_transform = camera['full_proj_transform']
    # project Gaussians to 2D
    projected_unhomog = xyz @ full_proj_transform[:3, :3]
    projected_unhomog = projected_unhomog + full_proj_transform[3:4, :3]
    reg = torch.sum(xyz * full_proj_transform[:3, 3].reshape(1, 3), dim=-1) + full_proj_transform[3, 3]
    projected_homogenous = projected_unhomog / (projected_unhomog[:, -1:] + 1e-4)

    # depths = projected_unhomog[:, 2:3]
    depths = reg.clone()  # don't do normalized depths
    projected_homogenous[:, 0] = ((projected_homogenous[:, 0] + 1.0) * camera['image_width'] - 1.0) * 0.5  # convert from NDC coords to pixel space
    projected_homogenous[:, 1] = ((projected_homogenous[:, 1] + 1.0) * camera['image_height'] - 1.0) * 0.5
    means2D_differentiable = projected_homogenous[:, :2]

    if return_depths and not return_conical_opacity:
        return means2D_differentiable, depths
    
    if return_conical_opacity:
        def fov2focal(fov, pixels):
            return pixels / (2 * math.tan(fov / 2))

        scales_1d = torch.exp(scales).max(-1).values
        cov_1d = scales_1d ** 2
        conical_x = cov_1d * (fov2focal(camera['FoVx'], camera['image_width']) / (reg + 0.0001)) ** 2 + 0.3
        conical_y = cov_1d * (fov2focal(camera['FoVy'], camera['image_height']) / (reg + 0.0001)) ** 2 + 0.3
        det = conical_x * conical_y
        conical_x, conical_y = conical_x / det, conical_y / det
        
        conical_xy = torch.zeros_like(conical_x)
        conical_opacity = torch.stack([conical_x, conical_xy, conical_y, torch.sigmoid(opacity).flatten()], dim=-1)
        return means2D_differentiable, depths, conical_opacity

    return means2D_differentiable

def _gather_knn(raw_knn_idxs, depths_source, src_particles, means2D_source, conical_opacity, K: int):
    device = raw_knn_idxs.device
    M = src_particles.shape[0]

    # gather source depths by knn, and then densely order them
    found_mask = (raw_knn_idxs != -1)
    raw_knn_idxs[~found_mask] = 0
    depths_source_knn = torch.gather(depths_source, 0, raw_knn_idxs.flatten()).reshape(-1, K)
    depths_source_knn[~found_mask[0]] = 1e6
    depth_knn_order = torch.argsort(depths_source_knn, dim=-1)  # depth-order each pixel's KNN
    raw_knn_idxs[~found_mask] = -1
    knn_idxs = torch.gather(raw_knn_idxs.squeeze(0), 1, depth_knn_order)  # shape (batch_size, K)
    found_mask = (knn_idxs != -1)
    knn_idxs[~found_mask] = 0
    found_mask = found_mask.flatten()
    knn_idxs = knn_idxs.flatten()
    
    # gather particles
    particles_idxs = torch.arange(M, device='cuda').repeat_interleave(K).to(device)
    src_particles_gathered = torch.gather(src_particles, 0, torch.stack([particles_idxs.flatten(), particles_idxs.flatten()], dim=-1)).reshape(-1, 2)  # MK' x 2
    src_particles_gathered[torch.logical_not(found_mask)] = -1
    
    # Gather gaussian properties
    means2D_source_gathered = torch.gather(means2D_source, 0, torch.stack([knn_idxs.flatten(), knn_idxs.flatten()], dim=-1)).reshape(-1, 2)  # MK' x 2
    means2D_source_gathered[torch.logical_not(found_mask)] = -1
    depths_source_gathered = torch.gather(depths_source, 0, knn_idxs.flatten())  # MK' x 2
    depths_source_gathered[torch.logical_not(found_mask)] = 1e6
    conical_opacity = torch.gather(conical_opacity, 0, torch.stack([knn_idxs.flatten(), knn_idxs.flatten(), knn_idxs.flatten(), knn_idxs.flatten()], dim=-1)).reshape(-1, 4)  # MK' x 2
    conical_opacity[torch.logical_not(found_mask)] = 0

    return src_particles_gathered, means2D_source_gathered, depths_source_gathered, conical_opacity, found_mask, knn_idxs, particles_idxs

def _accumulate_opacity_into_influence(particles_idxs, dists, conical_opacity, depths_src_gather, found_mask, continue_depth=False):
    # compute per-particle opacities
    power = -0.5 * (conical_opacity[:, 0] * dists[:, 0] ** 2 + conical_opacity[:, 2] * dists[:, 1] ** 2)
    power = power - conical_opacity[:, 1] * dists[:, 0] * dists[:, 1]
    power = torch.clip(power, max=0)
    alpha = torch.clip(conical_opacity[:, 3] * torch.exp(power), 0, 0.99)

    # use log trick to get cumulative product
    _, track_idxs_local = torch.unique(particles_idxs, return_inverse=True)
    ptr = torch.where((track_idxs_local[1:] - track_idxs_local[:-1]) > 0)[0] + 1
    log_transparencies = torch.log2(1 - alpha)
    log_agg_transparencies = torch.cumsum(log_transparencies, dim=0)
    diff_corrections = torch.cat([torch.zeros(1).to(log_transparencies), log_agg_transparencies[ptr - 1]])
    diff_corrections = diff_corrections[track_idxs_local]
    log_agg_transparencies = log_agg_transparencies - diff_corrections
    agg_transparencies = torch.exp2(log_agg_transparencies)

    # correct for 1st element being less than 1
    ptr_wzero = torch.cat([torch.zeros(1).to(ptr), ptr])
    init_transparencies = agg_transparencies[ptr_wzero]
    init_transparencies = init_transparencies[track_idxs_local]
    agg_transparencies = agg_transparencies / init_transparencies

    # finally, get influences
    influences = alpha * agg_transparencies
    influences = influences.detach().clone()
        
    if continue_depth:
        depth_info = depths_src_gather.reshape(-1, K).detach().contiguous() # (N, K)
        depth_diff = depth_info[:, 1:] - depth_info[:, :-1] #(N-1, K)
        depth_diff = torch.cat([torch.zeros((depth_info.shape[0], 1)).to(depth_diff), depth_diff], dim=1) # (N, K)
        continue_mask = depth_diff > 1e-2
        continue_mask = continue_mask & found_mask.reshape(-1, K)
        
        continue_mask_pre = torch.cat((continue_mask[:, 1:], torch.zeros((continue_mask.shape[0], 1)).to(continue_mask)), dim=1)
        index_map = torch.arange(depth_info.shape[0]).reshape(-1, 1).expand(-1, K).to(depth_diff.device)
        next_depth = torch_scatter.scatter_min(depth_info[continue_mask_pre], index=index_map[continue_mask_pre], dim_size=depth_info.shape[0])[0]
        next_depth = next_depth - depth_info[:, 0]
        next_depth = next_depth[next_depth > 0]
        next_depth_mean = next_depth.mean(dim=0)
        next_depth_std = next_depth.std(dim=0)
        next_depth_bar = next_depth_mean + 1 * next_depth_std
        continue_mask = torch.cumsum(continue_mask.float(), dim=1) < 1
        depth_bar_mask = (depth_info - depth_info[:, 0].unsqueeze(-1) < next_depth_bar.unsqueeze(-1))
        continue_mask = continue_mask & depth_bar_mask
        
        influences = influences.reshape(-1, K)
        influences[continue_mask] = influences.max(dim=1).values.unsqueeze(-1).expand(-1, K)[continue_mask]
        influences = influences.reshape(-1)

    return influences.reshape(-1), dists
