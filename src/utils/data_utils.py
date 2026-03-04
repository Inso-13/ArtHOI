import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, quaternion_multiply, quaternion_to_matrix, quaternion_to_axis_angle, axis_angle_to_quaternion, matrix_to_axis_angle, euler_angles_to_matrix, rotation_6d_to_matrix
from plyfile import PlyData, PlyElement
from utils.io_utils import data_to_device, load_ply

def resolve_path_template(template, cfg):
    """Parse path template and replace variables"""
    return template.format(
        data_root=cfg.data.paths.data_root,
        scene=cfg.scene
    )

def get_scene_paths(cfg):
    """Get all scene-related paths based on configuration"""
    scene_paths = {}
    for key, template in cfg.data.scene_paths.items():
        scene_paths[key] = resolve_path_template(template, cfg)
    return scene_paths

def load_align_data(cfg):
    """Load alignment data"""
    scene_paths = get_scene_paths(cfg)
    align_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), scene_paths['align_json'])
    with open(align_json_path, "r+") as f:
        return json.load(f)
    
def get_human_gs(cfg):
    scene_paths = get_scene_paths(cfg)
    align_data = load_align_data(cfg)
    
    # Get scale modifiers
    scale_modifiers = cfg.data.scale_modifiers
    human_scale_modifier = getattr(scale_modifiers, 'human', align_data['scale_modifier']['human'])
    
    # Build paths
    human_gs_cano_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), scene_paths['human_gs_cano'])
    human_gs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), scene_paths['human_gs'])
    
    human_gs_cano = load_ply(human_gs_cano_path)
    human_gs = load_ply(human_gs_path)
    
    human_offsets = align_data['human']['translation']
    human_scale = align_data['human']['scale']
    human_scale = human_scale + np.log(human_scale_modifier)
    human_rotations = align_data['human']['rotation']
    human_rotations = human_rotations - np.array([0,0,180])
    human_rotations = np.array([human_rotations[2], human_rotations[1], human_rotations[0]]) # ZYX
    human_offsets = torch.from_numpy(np.array(human_offsets) * np.array([-1,-1,1])).cuda().float()
    human_scale = torch.from_numpy(np.array(human_scale)).cuda().float()
    human_rotations = euler_angles_to_matrix(torch.from_numpy(np.array(human_rotations).astype(np.float32) / 180 * np.pi)[None], convention="ZYX")[0].cuda().float()
    return data_to_device(human_gs_cano), data_to_device(human_gs), human_offsets, human_scale, human_rotations

def get_object_gs_cano(cfg, object_gaussian, using_orient_anything=False):
    align_data = load_align_data(cfg)
    scale_modifiers = cfg.data.scale_modifiers
    object_scale_modifier = getattr(scale_modifiers, 'object', align_data['scale_modifier']['object'])
    
    object_offsets = align_data['object']['translation']
    object_scale = align_data['object']['scale']
    object_scale = object_scale + np.log(object_scale_modifier)
    object_rotations = align_data['object']['rotation']
    object_rotations = object_rotations - np.array([0,0,180])
    object_rotations = np.array([object_rotations[2], object_rotations[1], object_rotations[0]]) # ZYX
    object_offsets = torch.from_numpy(np.array(object_offsets) * np.array([-1,-1,1])).cuda().float()
    object_scale = torch.from_numpy(np.array(object_scale)).cuda().float()
    object_rotations = euler_angles_to_matrix(torch.from_numpy(np.array(object_rotations).astype(np.float32) / 180 * np.pi)[None], convention="ZYX")[0].cuda().float()
    object_gs_cano = object_gaussian.gaussians
    object_gs_cano = data_to_device(object_gs_cano)
    return object_gs_cano, object_offsets, object_scale, object_rotations

def get_camera(cfg):
    scene_paths = get_scene_paths(cfg)
    camera_filename = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), scene_paths['camera_json'])
    with open(camera_filename, "r+") as f:
        camera = json.load(f)
    return data_to_device(camera)

def get_smplx_init(cfg, use_gvhmr=True, use_hamer=True):
    scene_paths = get_scene_paths(cfg)
    smplx_init_json = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), scene_paths['smplx_init_json'])
    with open(smplx_init_json, "r+") as f:
        smplx_init = json.load(f)
        
    if use_gvhmr or use_hamer:
        gvhmr_init_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), scene_paths['gvhmr_init_json'])
        gvhmr_smpl = torch.load(gvhmr_init_json_path)['smpl_params_incam']
        hamer_init_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), scene_paths['hamer_init_json'])
        with open(hamer_init_json_path, 'r+') as f:
            hamer_mano = json.load(f)
        gvhmr_smplx = merge_gvhmr_and_hamer(gvhmr_smpl, hamer_mano)
        
        Es = torch.zeros((gvhmr_smpl['global_orient'].shape[0], 4, 4)).float()
        Es[:, :3, :3] = quaternion_to_matrix(axis_angle_to_quaternion(gvhmr_smpl['global_orient']))
        Es[:, :3, 3] = gvhmr_smpl['transl']
        Es[:, 3, 3] = 1.0

        for i, (frame_id, v) in enumerate(smplx_init.items()):
            if use_hamer:
                try:
                    if gvhmr_smplx['has_left'][i]:
                        smplx_init[frame_id]['lhand_pose'] = (gvhmr_smplx['left_hand_pose'][i]).tolist()
                    if gvhmr_smplx['has_right'][i]:
                        smplx_init[frame_id]['rhand_pose'] = (gvhmr_smplx['right_hand_pose'][i]).tolist()
                except:
                    pass
            if use_gvhmr:
                try:
                    smplx_init[frame_id]['body_pose'] = gvhmr_smpl['body_pose'][i]
                    smplx_init[frame_id]['cam_extrinsics'] = Es[i]
                except:
                    pass
    return data_to_device(smplx_init)

def get_feet_contacts(cfg):
    scene_paths = get_scene_paths(cfg)
    gvhmr_init_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), scene_paths['gvhmr_init_json'])
    gvhmr_contacts = torch.load(gvhmr_init_json_path)['net_outputs']['static_conf_logits'][0].sigmoid()
    return gvhmr_contacts.cuda()

def get_gt_images(cfg, smplx_init):
    scene_paths = get_scene_paths(cfg)
    image_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), scene_paths['image_root'])
    human_mask_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), scene_paths['human_mask'])
    
    object_mask_root = scene_paths.get("object_mask", None)
    object_mask_root = None if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), object_mask_root)) else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), object_mask_root)
    track2d_root = scene_paths.get("track2d_root", None)
    track2d_root = None if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), track2d_root)) else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), track2d_root)
    
    gt_images, gt_human_masks, gt_object_masks, tracks_2d = [], [], [], []
    for k, v in tqdm(smplx_init.items(), desc="Reading Images"):
        image_path = os.path.join(image_root, k+".jpg")
        if not os.path.exists(image_path):
            break
        human_mask_path = os.path.join(human_mask_root, k+".jpg")
        image = torch.from_numpy(cv2.resize(cv2.imread(image_path), (1024, 1024))).cuda().float()
        human_mask = torch.from_numpy(cv2.resize(cv2.imread(human_mask_path), (1024, 1024))).cuda().float()
        gt_images.append(image[None])
        gt_human_masks.append(human_mask[None].mean(-1) > 128)
        if object_mask_root is not None:
            object_mask_path = os.path.join(object_mask_root, "{:06}.jpg".format(int(k)))
            object_mask = torch.from_numpy(cv2.resize(cv2.imread(object_mask_path), (1024, 1024))).cuda().float()
            object_mask = object_mask[None].mean(-1) > 10
            gt_object_masks.append(object_mask)
        if track2d_root is not None:
            track2d_path = os.path.join(track2d_root, "track_{:06}.npy".format(int(k)))
            try:
                track2d = torch.from_numpy(np.load(track2d_path)).cuda().float()
            except:
                track2d = None
            tracks_2d.append(track2d)
    gt_images = torch.cat(gt_images, dim=0).permute(0, 3, 1, 2) / 255.0
    gt_human_masks = torch.cat(gt_human_masks, dim=0).float()
    if object_mask_root is not None:
        print("[Info] Using Object Masks")
        gt_object_masks = torch.cat(gt_object_masks, dim=0).float()
    else:
        gt_object_masks = None
    return gt_images, gt_human_masks, gt_object_masks, tracks_2d

@torch.no_grad()
def get_scene_bg(cfg, renderer, camera):
    scene_paths = get_scene_paths(cfg)
    align_data = load_align_data(cfg)
    scale_modifiers = cfg.data.scale_modifiers
    scene_scale_modifier = getattr(scale_modifiers, 'scene', align_data['scale_modifier']['scene'])
    
    scene_ply_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), scene_paths['scene_ply'])
    scene_ply = load_ply(scene_ply_path)
    scene_ply['scales'] = scene_ply['scales'] + np.log(scene_scale_modifier)
    scene_ply = data_to_device(scene_ply)
    render_pkg = renderer.simple_render(scene_ply, camera)
    scene_bg = render_pkg['img']
    scene_depth = render_pkg['depthmap'][0]
    del scene_ply
    return scene_bg, scene_depth


def merge_gvhmr_and_hamer(gvhmr_smplx_params, hamer_mano_params):
    from scipy.spatial.transform import Rotation as R
    def compute_global_rotation(pose_axis_anges, joint_idx):
        global_rotation = np.eye(3)
        parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19]
        while joint_idx != -1:
            joint_rotation = R.from_rotvec(pose_axis_anges[joint_idx]).as_matrix()
            global_rotation = joint_rotation @ global_rotation
            joint_idx = parents[joint_idx]
        return global_rotation

    M = np.diag([-1, 1, 1])
    B = gvhmr_smplx_params['body_pose'].shape[0]
    gvhmr_smplx_params['has_left'] = []
    gvhmr_smplx_params['has_right'] = []
    gvhmr_smplx_params["left_hand_pose"] = []
    gvhmr_smplx_params["right_hand_pose"] = []
    for i in range(B):
        hamer_i = "{:06}".format(i)
        if hamer_mano_params['left'].get(hamer_i, None) is None:
            has_left = False
        else:
            has_left = True
        if hamer_mano_params['right'].get(hamer_i, None) is None:
            has_right = False
        else:
            has_right = True

        left_hand_pose = np.zeros(45)
        right_hand_pose = np.zeros(45)
        # Assuming that your data are stored in gvhmr_smplx_params and hamer_mano_params
        full_body_pose = torch.concatenate((gvhmr_smplx_params["global_orient"][i:i+1], gvhmr_smplx_params["body_pose"][i].reshape(21, 3)), dim=0)     # gvhmr_smplx_params["global_orient"]: (3, 3)
        left_elbow_global_rot = compute_global_rotation(full_body_pose, 18) # left elbow IDX: 18
        right_elbow_global_rot = compute_global_rotation(full_body_pose, 19) # left elbow IDX: 19

        if has_left:
            left_wrist_global_rot = np.array(hamer_mano_params['left'][hamer_i]["global_orient"])                                                      
            left_wrist_global_rot = M @ left_wrist_global_rot @ M
            left_wrist_pose = np.linalg.inv(left_elbow_global_rot) @ left_wrist_global_rot
            left_wrist_pose_vec = R.from_matrix(left_wrist_pose).as_rotvec()
            for j in range(15):
                left_finger_pose = M @ np.array(hamer_mano_params['left'][hamer_i]["hand_pose"][j]) @ M
                left_finger_pose_vec = R.from_matrix(left_finger_pose).as_rotvec()
                left_hand_pose[j*3: j*3+3] = left_finger_pose_vec
        gvhmr_smplx_params["left_hand_pose"].append(left_hand_pose)
        if has_right:
            right_wrist_global_rot = np.array(hamer_mano_params['right'][hamer_i]["global_orient"])  
            right_wrist_pose = np.linalg.inv(right_elbow_global_rot) @ right_wrist_global_rot
            right_wrist_pose_vec = R.from_matrix(right_wrist_pose).as_rotvec()
            for j in range(15):
                right_finger_pose = np.array(hamer_mano_params['right'][hamer_i]["hand_pose"][j])
                right_finger_pose_vec = R.from_matrix(right_finger_pose).as_rotvec()
                right_hand_pose[j*3: j*3+3] = right_finger_pose_vec
        gvhmr_smplx_params["right_hand_pose"].append(right_hand_pose)

        gvhmr_smplx_params["body_pose"][i, 57: 60] = torch.as_tensor(left_wrist_pose_vec)
        gvhmr_smplx_params["body_pose"][i, 60: 63] = torch.as_tensor(right_wrist_pose_vec)
        gvhmr_smplx_params['has_left'].append(has_left)
        gvhmr_smplx_params['has_right'].append(has_right)
    gvhmr_smplx_params['has_left'] = np.array(gvhmr_smplx_params['has_left'])
    gvhmr_smplx_params['has_right'] = np.array(gvhmr_smplx_params['has_right'])
    return gvhmr_smplx_params

def transform_human_gs(human_gs, frame_id, trans_params, root_pose_params, human_scale=None, human_rotations=None, human_offsets=None):
    human_gs['means3D'] = human_gs['means3D'] - trans_params[0].detach()
    human_gs['means3D'] = torch.einsum("ij,nj->ni", root_pose_params[0].detach(), human_gs['means3D'])
    human_gs['means3D'] = torch.einsum("ij,ni->nj", root_pose_params[frame_id], human_gs['means3D'])
    human_gs['means3D'] = human_gs['means3D'] + trans_params[frame_id]
    if human_scale is not None:
        human_gs['means3D'] = human_gs['means3D'] * human_scale
        human_gs['scales'] = human_gs['scales'] + human_scale
    if human_rotations is not None:
        human_gs['means3D'] = torch.einsum("ji,nj->ni", human_rotations, human_gs['means3D'])
    if human_offsets is not None:
        human_gs['means3D'] = human_gs['means3D'] + human_offsets.reshape(-1, 3)
    return human_gs

def transform_object_gs(object_gs, object_scale=None, object_rotations=None, object_offsets=None):
    if object_scale is not None:
        object_gs['means3D'] = object_gs['means3D'] * object_scale
        object_gs['scales'] = object_gs['scales'] + torch.log(object_scale) + 0.5
    if object_rotations is not None:
        object_gs['means3D'] = torch.einsum("ij,...j->...i", object_rotations, object_gs['means3D'])
        gs_rotations = quaternion_to_matrix(object_gs['rotations'])
        gs_rotations = (object_rotations @ gs_rotations.permute(2,1,0)).permute(2,1,0)
        object_gs['rotations'] = matrix_to_quaternion(gs_rotations)
    if object_offsets is not None:
        object_gs['means3D'] = object_gs['means3D'] + object_offsets
    return object_gs

def get_human_RT(world_view_transform, R_emc, T_emc, scale):
    R_camera = world_view_transform[:3, :3] # (3, 3)
    T_camera = world_view_transform[3, :3] # (3)
    R_emc = rotation_6d_to_matrix(R_emc) # (bs, 3, 3)
    Rh = torch.einsum("ij,bjk->bik", R_camera, R_emc)
    Th = torch.einsum("ij,bj->bi", R_camera, (T_emc - T_camera / scale))[:, None]
    return Rh, Th