import os
import copy
import torch
import numpy as np
import frnn
from tqdm import tqdm
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion, quaternion_multiply, quaternion_to_matrix, quaternion_to_axis_angle, axis_angle_to_quaternion, matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix

from models.humangs import HumanGaussian
from models.objectgs import ObjectGaussian
from models.renderer import GaussianRenderer
from models.sags import SAGS

from utils.io_utils import data_to_device, load_ply
from utils.data_utils import get_human_gs, get_scene_bg, get_camera, get_smplx_init, get_gt_images, get_object_gs_cano, get_feet_contacts, get_scene_paths, transform_human_gs, transform_object_gs, get_human_RT
from utils.vis_utils import vis_render
from utils.loss_utils import get_track_2d_loss, get_init_dynamic_data, get_smooth_loss, get_3dkpt_loss, get_foot_sliding_loss, get_collision_loss

from configs.config import cfg
    
class Trainer(object):
    def __init__(self):
        self.cfg = cfg
        self.log_dir = f"{self.cfg.data.paths.results_root}/{self.cfg.scene}/{self.cfg.exp_name}"
        self.data_dir = f"{self.cfg.data.paths.data_root}/{self.cfg.scene}"
        os.makedirs(self.log_dir, exist_ok=True)

        self.human_gaussian = HumanGaussian(gender="male")
        self.human_gs_cano, self.human_gs, self.human_offsets, self.human_scale, self.human_rotations = get_human_gs(self.cfg)
        self.human_gs_frame0 = transform_object_gs(copy.deepcopy(self.human_gs), self.human_scale, self.human_rotations, self.human_offsets)
        
        self.camera = get_camera(self.cfg)
        self.gaussian_renderer = GaussianRenderer()
        self.smplx_init = get_smplx_init(self.cfg)
        
        self.feet_contacts = get_feet_contacts(self.cfg)
        
        self.gt_images, self.gt_human_masks, self.gt_object_masks, self.pgt_tracks_2d = get_gt_images(self.cfg, self.smplx_init)
        self.Nframe = len(self.gt_images)
        scene_paths = get_scene_paths(self.cfg)
        object_ply_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), scene_paths['object_ply'])
        self.object_gaussian = ObjectGaussian(object_ply_path)
        self.object_gs_cano, self.object_offsets, self.object_scale, self.object_rotations = get_object_gs_cano(self.cfg, self.object_gaussian)

        self.human_offsets = torch.nn.Parameter(self.human_offsets)
        self.human_scale = torch.nn.Parameter(self.human_scale)
        self.human_rotations = torch.nn.Parameter(self.human_rotations)
        self.object_offsets = torch.nn.Parameter(self.object_offsets)
        self.object_scale = torch.nn.Parameter(self.object_scale)
        self.object_rotations = torch.nn.Parameter(self.object_rotations)
        
        self.scene_bg, self.scene_depth = get_scene_bg(self.cfg, self.gaussian_renderer, self.camera)

        self.set_optimizer()
        self.gs_sam = SAGS(self.camera, copy.deepcopy(self.object_gs_cano), transform_object_gs(copy.deepcopy(self.object_gs_cano), self.object_scale, self.object_rotations, self.object_offsets), self.gaussian_renderer, results_root=f"{self.cfg.data.paths.results_root}/{self.cfg.scene}/{self.cfg.exp_name}/", scene_name=self.cfg.scene)
        
        self.pgt_2dkpt = torch.cat([(v['keypoints'][None]) for v in self.smplx_init.values()], dim=0)
        self.pgt_2dkpt_smpl = self.get_pgt_smpl_2dkpt()

        if self.cfg.experiment.is_continue:
            self.train_start_frame_id = self.load_checkpoints()
        else:
            self.train_start_frame_id = 0

    @torch.no_grad()
    def get_pgt_smpl_2dkpt(self):
        smplx_param = {
            "R_emc": self.params['R_emc'],
            "T_emc": self.params['T_emc'],
            "body_pose": self.params['body_pose'],
            "lhand_pose": self.params['lhand_pose'],
            "rhand_pose": self.params['rhand_pose'],
            "jaw_pose": self.params['jaw_pose'],
            "betas": self.params['betas']
        }
        Rh, Th = get_human_RT(self.camera['world_view_transform'], self.params['R_emc'], self.params['T_emc'], self.human_scale)
        human_gs, human_joints = self.human_gaussian(smplx_param, self.human_gs_cano, return_joints=True)
        human_gs['means3D'] = torch.einsum("bij,bnj->bni", Rh, human_gs['means3D']) + Th
        human_gs['means3D'] = human_gs['means3D'] * self.human_scale
        human_gs['scales'] = human_gs['scales'] + torch.log(self.human_scale) + 1
        human_joints = torch.einsum("bij,bnj->bni", Rh, human_joints) + Th
        human_joints = human_joints * self.human_scale
        # 3d -> 2d
        bs = human_joints.shape[0]
        human_joints_2d = human_joints @ self.camera['full_proj_transform'][:3, :3] + self.camera['full_proj_transform'][3:4, :3]
        human_joints_2d = human_joints_2d / (human_joints_2d[:, :, -1:] + 1e-4)
        human_joints_2d[..., 0] = ((human_joints_2d[..., 0] + 1.0) * self.camera['image_width'] - 1.0) * 0.5
        human_joints_2d[..., 1] = ((human_joints_2d[..., 1] + 1.0) * self.camera['image_height'] - 1.0) * 0.5
        human_joints_2d[..., 2] = 1.0
        return human_joints_2d
    
    @torch.no_grad()
    def load_checkpoints(self):
        # stage 1
        for frame_id in tqdm(range(0, self.Nframe), desc="Loading ckpts"):
            ckpt_path = f"{self.cfg.data.paths.results_root}/{self.cfg.scene}/{self.cfg.exp_name}/params/stage1/{frame_id:04d}.pth"
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path)
                self.params['object_R'][frame_id] = ckpt["art_R"]
                self.params['object_T'][frame_id] = ckpt["art_T"]
                if frame_id == 0:
                    self.params['object_W'][:] = ckpt["art_W"].reshape(-1,1)
                elif "art_W" in ckpt:
                    del ckpt["art_W"]
                    torch.save(ckpt, ckpt_path)
            else:
                return frame_id
        # stage 2
        ckpt_path = f"{self.cfg.data.paths.results_root}/{self.cfg.scene}/{self.cfg.exp_name}/params/stage2.pth"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            self.params['R_emc'][:] = ckpt["R_emc"]
            self.params['T_emc'][:] = ckpt["T_emc"]
            self.params['body_pose'][:] = ckpt["body_pose"]
            self.params['lhand_pose'][:] = ckpt["lhand_pose"]
            self.params['rhand_pose'][:] = ckpt["rhand_pose"]
            self.params['jaw_pose'][:] = ckpt["jaw_pose"]
            self.params['betas'][:] = ckpt["betas"]
            return -1
        return frame_id+1
            
    def set_optimizer(self):
        ### opt init 
        self.params = {}
        self.params['R_emc'] = torch.cat([(v['cam_extrinsics'][:3,:3][None]) for v in self.smplx_init.values()], dim=0).reshape(-1, 3, 3)
        self.params['R_emc'] = torch.nn.Parameter(matrix_to_rotation_6d(self.params['R_emc']))
        self.params['T_emc'] = torch.nn.Parameter(torch.cat([v['cam_extrinsics'][:3,3][None] for v in self.smplx_init.values()], dim=0).reshape(-1,3))
        self.init_body_pose = torch.cat([v['body_pose'][None] for v in self.smplx_init.values()], dim=0)
        self.params['body_pose'] = torch.nn.Parameter(torch.clone(self.init_body_pose))
        self.params['lhand_pose'] = torch.nn.Parameter(torch.cat([v['lhand_pose'][None] for v in self.smplx_init.values()], dim=0))
        self.params['rhand_pose'] = torch.nn.Parameter(torch.cat([v['rhand_pose'][None] for v in self.smplx_init.values()], dim=0))
        self.params['jaw_pose'] = torch.nn.Parameter(torch.cat([v['jaw_pose'][None] for v in self.smplx_init.values()], dim=0))
        self.params['betas'] = torch.nn.Parameter(torch.cat([v['betas'][None] for v in self.smplx_init.values()], dim=0).mean(0))
        self.params['object_R'] = torch.nn.Parameter(torch.zeros((self.Nframe, self.object_gaussian.N_joints, 6)).float().cuda())
        self.params['object_T'] = torch.nn.Parameter(torch.zeros((self.Nframe, self.object_gaussian.N_joints, 3)).float().cuda())
        self.params['object_W'] = torch.ones((self.object_gaussian.N_vertices, self.object_gaussian.N_joints)).float().cuda() * -10
        
        self.object_gs_cano['features_dc'] = torch.nn.Parameter(self.object_gs_cano['features_dc']).float().cuda()
        self.object_gs_cano['features_rest'] = torch.nn.Parameter(self.object_gs_cano['features_rest']).float().cuda()
        
        with torch.no_grad():
            self.params['object_R'][:, :, 0] = 1.0
            self.params['object_R'][:, :, 4] = 1.0
        ### opt init 
        self.optimizer = torch.optim.Adam(
            params=[
                {'params': [self.params['R_emc']], 'lr': self.cfg.optimizer.learning_rates.R_emc, "name": "R_emc"},
                {'params': [self.params['T_emc']], 'lr': self.cfg.optimizer.learning_rates.T_emc, "name": "T_emc"},
                {'params': [self.params['body_pose']], 'lr': self.cfg.optimizer.learning_rates.body_pose, "name": "body_pose"},
                {'params': [self.params['lhand_pose']], 'lr': self.cfg.optimizer.learning_rates.lhand_pose, "name": "lhand_pose"},
                {'params': [self.params['rhand_pose']], 'lr': self.cfg.optimizer.learning_rates.rhand_pose, "name": "rhand_pose"},
                {'params': [self.params['jaw_pose']], 'lr': self.cfg.optimizer.learning_rates.jaw_pose, "name": "jaw_pose"},
                {'params': [self.params['betas']], 'lr': self.cfg.optimizer.learning_rates.betas, "name": "betas"},
                {'params': [self.params['object_R']], 'lr': self.cfg.optimizer.learning_rates.object_R, "name": "object_R"},
                {'params': [self.params['object_T']], 'lr': self.cfg.optimizer.learning_rates.object_T, "name": "object_T"},
                
                {'params': [self.object_gs_cano['features_dc']], 'lr': self.cfg.optimizer.learning_rates.color_object_features_dc, "name": "color_object_features_dc"},
                {'params': [self.object_gs_cano['features_rest']], 'lr': self.cfg.optimizer.learning_rates.color_object_features_rest, "name": "color_object_features_rest"},
            ]
        )
        
    @torch.no_grad()            
    def switch_to_human_training(self):
        """Switch to human parameter training mode: enable human params, disable object params"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ["R_emc", "T_emc"]:
                param_group['lr'] = self.cfg.optimizer.dynamic_lr.human_update.R_emc
            elif param_group["name"] in ['body_pose']:
                param_group['lr'] = self.cfg.optimizer.dynamic_lr.human_update.body_pose
            elif param_group["name"] in ['lhand_pose', 'rhand_pose', "jaw_pose"]:
                param_group['lr'] = self.cfg.optimizer.dynamic_lr.human_update.lhand_pose
            elif param_group["name"] in ["betas"]:
                param_group['lr'] = self.cfg.optimizer.dynamic_lr.human_update.betas
            elif param_group["name"] in ["object_R", "object_T"]:
                param_group['lr'] = self.cfg.optimizer.dynamic_lr.off_update.object_R

    @torch.no_grad()            
    def switch_to_object_training(self):
        """Switch to object parameter training mode: enable object params, disable human params"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ["object_R", "object_T"]:
                param_group['lr'] = self.cfg.optimizer.dynamic_lr.object_update.object_R
            elif param_group["name"] in ["R_emc", "T_emc", 'body_pose', 'lhand_pose', 'rhand_pose', "jaw_pose", "betas"]:
                param_group['lr'] = self.cfg.optimizer.dynamic_lr.off_update.R_emc       

    @torch.no_grad()
    def get_contact_mask(self):
        object_Rs = self.params['object_R']
        object_Ts = self.params['object_T']
        object_Rs = matrix_to_quaternion(rotation_6d_to_matrix(object_Rs))
        d_object_Ts = torch.norm(object_Ts[1:] - object_Ts[:-1], dim=-1, p=2)
        d_object_Rs = torch.norm(object_Rs[1:] - object_Rs[:-1], dim=-1, p=2)
        
        contact_mask = ((d_object_Rs > 1e-3) & (d_object_Ts > 1e-3)).reshape(-1)
        n = self.cfg.training.contact.contact_mask_n
        contact_mask = torch.nn.functional.max_pool1d(contact_mask.unsqueeze(0).unsqueeze(0).float(), kernel_size=2*n+1, stride=1, padding=n).squeeze().bool()
        contact_mask = torch.cat([torch.zeros(1).to(contact_mask), contact_mask], dim=0)
        contact_mask[:self.start_moving_id] = False
        print("contact_mask: ", torch.where(contact_mask)[0])
        return contact_mask

    @torch.no_grad()
    def get_contact_time_region(self):
        human_2dkpt_threhold = self.cfg.training.contact.human_2dkpt_threshold
        contact_mask = self.get_contact_mask()
        human_sam_masks = self.gt_human_masks > 0
        object_sam_masks = self.gt_object_masks > 0
        human_2dkpt = self.pgt_2dkpt

        # Fingertip keypoint indices (left hand, right hand)
        left_fingertips = [29, 33, 37, 41, 45]
        right_fingertips = [50, 54, 58, 62, 66]
        pgt_ketpoints_3d = torch.zeros((self.Nframe, human_2dkpt.shape[1], 4)).float().cuda() # x, y, z, conf
        
        for frame_id in tqdm(torch.where(contact_mask)[0], desc="Geting Contact 3DJoints"):
            object_param = {
                "art_R": self.params['object_R'][frame_id:frame_id+1],
                "art_T": self.params['object_T'][frame_id:frame_id+1],
                "art_W": self.params['object_W'][None]
            }
            object_gs = self.object_gaussian(object_param, self.object_gs_cano)
            object_gs = transform_object_gs(object_gs, self.object_scale, self.object_rotations, self.object_offsets)
            object_gs['means3D'] = object_gs['means3D'][0]
            object_model_mask = (self.gaussian_renderer.simple_render(object_gs, self.camera)['mask'] > 0)[0]
        
            smplx_param = {
                "R_emc": self.params['R_emc'][frame_id:frame_id+1],
                "T_emc": self.params['T_emc'][frame_id:frame_id+1],
                "body_pose": self.params['body_pose'][frame_id:frame_id+1],
                "lhand_pose": self.params['lhand_pose'][frame_id:frame_id+1],
                "rhand_pose": self.params['rhand_pose'][frame_id:frame_id+1],
                "jaw_pose": self.params['jaw_pose'][frame_id:frame_id+1],
                "betas": self.params['betas']
            }
            Rh, Th = get_human_RT(self.camera['world_view_transform'], self.params['R_emc'], self.params['T_emc'], self.human_scale)
            human_gs = self.human_gaussian(smplx_param, self.human_gs_cano)[0]
            human_gs['means3D'] = torch.einsum("ij,nj->ni", Rh[frame_id], human_gs['means3D'][0]) + Th[frame_id]
            human_gs['means3D'] = human_gs['means3D'] * self.human_scale
            human_gs['scales'] = human_gs['scales'] + torch.log(self.human_scale) + 1
            human_model_mask = (self.gaussian_renderer.simple_render(human_gs, self.camera)['mask'] > 0)[0]
                
            #### human_before_object
            human_before_object_mask = human_sam_masks[frame_id] & ~object_sam_masks[frame_id] & object_model_mask
            human_before_object_kpt_mask = human_before_object_mask[human_2dkpt[frame_id, :, 1].long(), human_2dkpt[frame_id, :, 0].long()]
            human_before_object_kpt_mask = human_before_object_kpt_mask & (human_2dkpt[frame_id, :, 2] > human_2dkpt_threhold)
            human_before_object_kpt_idx = torch.where(human_before_object_kpt_mask)[0]
            if human_before_object_kpt_mask.sum() > 0:
                human_before_object_kpt = human_2dkpt[frame_id, human_before_object_kpt_mask, :2]

                ### get_min_depth
                dynamic_gs = object_gs['means3D'][(self.params['object_W'] > 5)[:, 0]]
                dynamic_gs_2d = dynamic_gs @ self.camera['full_proj_transform'][:3, :3] + self.camera['full_proj_transform'][3:4, :3]
                dynamic_gs_view = (self.camera['world_view_transform'][:3, :3].T @ dynamic_gs.T + self.camera['world_view_transform'][3, :3][:, None]).T
                dynamic_gs_depth = dynamic_gs_view[:, 2]

                dynamic_gs_2d = dynamic_gs_2d / (dynamic_gs_2d[:, -1:] + 1e-4)
                dynamic_gs_2d[:, 0] = ((dynamic_gs_2d[:, 0] + 1.0) * self.camera['image_width'] - 1.0) * 0.5
                dynamic_gs_2d[:, 1] = ((dynamic_gs_2d[:, 1] + 1.0) * self.camera['image_height'] - 1.0) * 0.5
                dynamic_gs_2d = dynamic_gs_2d[:, :2]
                
                dists, raw_knn_idxs, nn, grid = frnn.frnn_grid_points(
                    human_before_object_kpt.unsqueeze(0), dynamic_gs_2d.unsqueeze(0), 
                    torch.tensor([human_before_object_kpt.size(0)]).to(human_before_object_kpt.device), torch.tensor([dynamic_gs_2d.size(0)]).to(dynamic_gs_2d.device),
                    100, 1, grid=None, return_nn=False, return_sorted=True
                )
                raw_knn_idxs = raw_knn_idxs[0]
                for kpt_i in range(human_before_object_kpt.size(0)):
                    raw_knn_idxs_kpt_i = raw_knn_idxs[kpt_i][raw_knn_idxs[kpt_i] > 0]
                    if len(raw_knn_idxs_kpt_i) == 0:
                        continue
                    min_depth_idx = torch.argmin(dynamic_gs_depth[raw_knn_idxs_kpt_i])
                    min_depth_gs_view = dynamic_gs_view[raw_knn_idxs_kpt_i[min_depth_idx]].reshape(1,3)
                    min_depth_gs_view[:, 2] -= 0.05 # 5cm
                    min_depth_gs = (self.camera['world_view_transform'][:3, :3] @ (min_depth_gs_view.T - self.camera['world_view_transform'][3, :3][:, None])).T
                    pgt_ketpoints_3d[frame_id, human_before_object_kpt_idx[kpt_i], :3] = min_depth_gs
                    pgt_ketpoints_3d[frame_id, human_before_object_kpt_idx[kpt_i], 3] = 1.0

        # body25
        pgt_ketpoints_3d[:, :25] = 0.0
        # face
        pgt_ketpoints_3d[:, 67:] = 0.0
        return pgt_ketpoints_3d

    @torch.no_grad()
    def align_human_frame0(self):
        Rh, Th = get_human_RT(self.camera['world_view_transform'], self.params['R_emc'], self.params['T_emc'], self.human_scale)
        human_gs, human_joints, human_vertices = self.human_gaussian(self.params, self.human_gs_cano, return_joints=True, return_vertices=True)
        human_gs['means3D'] = torch.einsum("bij,bnj->bni", Rh, human_gs['means3D']) + Th
        human_gs['means3D'] = human_gs['means3D'] * self.human_scale
        d_Th_cam = (self.human_gs_frame0['means3D'] - human_gs['means3D'][0]).mean(0, keepdims=True) # (1, 3)

        R_camera = self.camera['world_view_transform'][:3, :3] # (3, 3)
        d_Th = torch.einsum("ji,bj->bi", R_camera, d_Th_cam) / self.human_scale
        self.params['T_emc'] += d_Th

    ############## train/render ##############
    def train(self):
        if self.train_start_frame_id == -1:
            return
        os.makedirs(f"{self.cfg.data.paths.results_root}/{self.cfg.scene}/{self.cfg.exp_name}/", exist_ok=True)
        
        self.train_stage1()
        self.train_stage2_batch()
        
    def train_stage1(self):
        self.switch_to_object_training()
        self.align_human_frame0()
        with torch.no_grad():
            init_dynamic_data = get_init_dynamic_data(self.pgt_tracks_2d, self.Nframe, self.camera, self.object_gaussian, self.object_gs_cano, self.object_scale, self.object_rotations, self.object_offsets, self.params['object_R'], self.params['object_T'], self.params['object_W'], gt_object_masks=self.gt_object_masks, gt_images=self.gt_images)
            self.start_moving_id = init_dynamic_data['start_moving_id']
            init_dynamic_data = self.gs_sam.run(init_dynamic_data)
            self.params['object_W'][init_dynamic_data['dynamic']] = float("inf")
            self.params['object_W'][init_dynamic_data['static']] = -float("inf")

            object_param = {
                "art_R": self.params['object_R'][0:1].detach(),
                "art_T": self.params['object_T'][0:1].detach(),
                "art_W": self.params['object_W'][None].detach()
            }
            object_gs = self.object_gaussian(object_param, self.object_gs_cano)
            object_gs['means3D'] = object_gs['means3D'][0]
            object_gs = transform_object_gs(object_gs, self.object_scale, self.object_rotations, self.object_offsets)

            self.object_quasi_static_pairs = init_dynamic_data['quasi_static_pairs']
            quasi_pairs_cano = self.object_gs_cano['means3D'][self.object_quasi_static_pairs]
            quasi_pairs_distance_cano = torch.norm(quasi_pairs_cano[:, 0] - quasi_pairs_cano[:, 1], dim=-1, p=2) * self.object_scale
        
        ##### optim
        loss_weights = {
            "l1": self.cfg.loss_weights.stage1.l1,
            "l1_object": self.cfg.loss_weights.stage1.l1_object,
            "object_mask": self.cfg.loss_weights.stage1.object_mask,
            "track_2d": self.cfg.loss_weights.stage1.track_2d,
            "object_axis": self.cfg.loss_weights.stage1.object_axis,
        }
        for frame_id in tqdm(range(self.train_start_frame_id, self.Nframe), desc="Frames"):
            if frame_id > 0:
                with torch.no_grad():
                    if frame_id > 1:
                        self.params['object_R'][frame_id] = self.params['object_R'][frame_id-1] + (self.params['object_R'][frame_id-1] - self.params['object_R'][frame_id-2])
                        self.params['object_T'][frame_id] = self.params['object_T'][frame_id-1] + (self.params['object_T'][frame_id-1] - self.params['object_T'][frame_id-2])
                    else:
                        self.params['object_R'][frame_id] = self.params['object_R'][frame_id-1]
                        self.params['object_T'][frame_id] = self.params['object_T'][frame_id-1]
                    self.params['object_W'][frame_id] = self.params['object_W'][frame_id-1]
                    self.params['R_emc'][frame_id] = self.params['R_emc'][frame_id-1]
                    self.params['T_emc'][frame_id] = self.params['T_emc'][frame_id-1]
                    
            smplx_param = {
                "R_emc": self.params['R_emc'][frame_id:frame_id+1],
                "T_emc": self.params['T_emc'][frame_id:frame_id+1],
                "body_pose": self.params['body_pose'][frame_id:frame_id+1],
                "lhand_pose": self.params['lhand_pose'][frame_id:frame_id+1],
                "rhand_pose": self.params['rhand_pose'][frame_id:frame_id+1],
                "jaw_pose": self.params['jaw_pose'][frame_id:frame_id+1],
                "betas": self.params['betas']
            }
            object_param = {
                "art_R": self.params['object_R'][frame_id:frame_id+1],
                "art_T": self.params['object_T'][frame_id:frame_id+1],
                "art_W": self.params['object_W'][None]
            }
            
            losses_for_log = {'loss': 0.0}
            for iter in tqdm(range(self.cfg.experiment.stage1_iterations), desc="Optiming Stage1 {}/{}".format(frame_id, self.Nframe)):
                Rh, Th = get_human_RT(self.camera['world_view_transform'], self.params['R_emc'], self.params['T_emc'], self.human_scale)
                human_gs = self.human_gaussian(smplx_param, self.human_gs_cano)[0]
                human_gs['means3D'] = torch.einsum("ij,nj->ni", Rh[frame_id], human_gs['means3D'][0]) + Th[frame_id]
                human_gs['means3D'] = human_gs['means3D'] * self.human_scale
                human_gs['scales'] = human_gs['scales'] + torch.log(self.human_scale) + 1
                
                losses = {}
                object_gs = self.object_gaussian(object_param, self.object_gs_cano)
                object_gs['means3D'] = object_gs['means3D'][0]
                object_gs = transform_object_gs(object_gs, self.object_scale, self.object_rotations, self.object_offsets)
                all_render = self.gaussian_renderer(human_gs, object_gs, self.scene_bg, cam_param=self.camera)
                
                gt_img = self.gt_images[frame_id]
                gt_human_mask = self.gt_human_masks[frame_id]
                pred_img = all_render['img'][[2,1,0]]
                pred_human_mask = all_render['mask_human'][0]
                
                loss_l1 = ((pred_img - gt_img) ** 2)
                losses['l1'] = loss_l1.mean()
                
                if self.gt_object_masks is not None:
                    gt_object_mask = self.gt_object_masks[frame_id]
                    pred_object_mask = all_render['mask_object'][0]
                    not_overlay_mask = (~((gt_human_mask > 0) & (pred_object_mask.detach() > 0))).float()
                    loss_human_mask = ((pred_human_mask - gt_human_mask) ** 2) * not_overlay_mask
                    loss_object_mask = ((pred_object_mask - gt_object_mask) ** 2) * not_overlay_mask
                    losses['object_mask'] = loss_object_mask.mean()
                    losses['l1_object'] = loss_l1[:, gt_object_mask > 0].mean()
                else:
                    not_overlay_mask = 1
                    pred_object_mask = None
                    gt_object_mask = None
                
                if self.pgt_tracks_2d is not None and len(self.pgt_tracks_2d) > 0:
                    losses['track_2d'], dynamic_gs_mask, static_gs_mask = get_track_2d_loss(self.pgt_tracks_2d, frame_id, self.camera, self.object_gaussian, self.object_gs_cano, self.object_scale, self.object_rotations, self.object_offsets, self.params['object_R'], self.params['object_T'], self.params['object_W'], gt_object_masks=self.gt_object_masks, gt_images=self.gt_images)
                    quasi_pairs = object_gs['means3D'][self.object_quasi_static_pairs]
                    quasi_pairs_distance = torch.norm(quasi_pairs[:, 0] - quasi_pairs[:, 1], dim=-1, p=2)
                    losses["object_axis"] = torch.norm(quasi_pairs_distance - quasi_pairs_distance_cano, dim=-1, p=2).mean()
                
                loss = sum([loss_weights[k] * v for k, v in losses.items() if not torch.isnan(v)])
                
                with torch.no_grad():
                    losses_for_log['loss'] = loss.item()
                    for loss_name, loss_value in losses.items():
                        if loss_name not in losses_for_log:
                            losses_for_log[loss_name] = 0.0
                        losses_for_log[loss_name] = loss_value.item() * loss_weights[loss_name]
                    if ((iter+1) % self.cfg.training.debug.log_interval == 0):
                        postfix = {
                            k: f"{v:.{3}f}" for k,v in losses_for_log.items()
                        }
                        print()
                        print(postfix)
                # if iter == self.cfg.experiment.stage1_iterations-1:
                #     vis_render(pred_img, gt_img, pred_human_mask, pred_object_mask, gt_human_mask, gt_object_mask, not_overlay_mask, save_path=f"{self.cfg.data.paths.results_root}/{self.cfg.scene}/{self.cfg.exp_name}/stage1_ends/{frame_id:04d}.png", ratio=1/3)
                            
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.save_parameters_stage1(object_param, frame_id)

    def train_stage2_batch(self):
        self.pgt_ketpoints_3d = self.get_contact_time_region()
        self.switch_to_human_training()
        loss_weights = {
            "l1": self.cfg.loss_weights.stage2.l1,
            "human_mask": self.cfg.loss_weights.stage2.human_mask,
            "human_smooth": self.cfg.loss_weights.stage2.human_smooth,
            "human_frame0": self.cfg.loss_weights.stage2.human_frame0,
            "human_3dkpt": self.cfg.loss_weights.stage2.human_3dkpt,
            "human_initpose": self.cfg.loss_weights.stage2.human_initpose,
            "foot_sliding": self.cfg.loss_weights.stage2.foot_sliding,
            "collision": self.cfg.loss_weights.stage2.collision,
        }
        losses_for_log = {'loss': 0.0}
        for iter in tqdm(range(self.cfg.experiment.stage2_iterations), desc="Human Iters"):
            smplx_param = {
                "R_emc": self.params['R_emc'],
                "T_emc": self.params['T_emc'],
                "body_pose": self.params['body_pose'],
                "lhand_pose": self.params['lhand_pose'],
                "rhand_pose": self.params['rhand_pose'],
                "jaw_pose": self.params['jaw_pose'],
                "betas": self.params['betas']
            }
            object_param = {
                "art_R": self.params['object_R'],
                "art_T": self.params['object_T'],
                "art_W": self.params['object_W'][None]
            }
            Rh, Th = get_human_RT(self.camera['world_view_transform'], self.params['R_emc'], self.params['T_emc'], self.human_scale)

            human_gs, human_joints, human_vertices = self.human_gaussian(smplx_param, self.human_gs_cano, return_joints=True, return_vertices=True)
            human_gs['means3D'] = torch.einsum("bij,bnj->bni", Rh, human_gs['means3D']) + Th
            human_gs['means3D'] = human_gs['means3D'] * self.human_scale
            human_gs['scales'] = human_gs['scales'] + torch.log(self.human_scale) + 1
            human_joints = torch.einsum("bij,bnj->bni", Rh, human_joints) + Th
            human_joints = human_joints * self.human_scale
            human_vertices = torch.einsum("bij,bnj->bni", Rh, human_vertices) + Th
            human_vertices = human_vertices * self.human_scale
            
            losses = {}
            object_gs = self.object_gaussian(object_param, self.object_gs_cano)
            object_gs = transform_object_gs(object_gs, self.object_scale, self.object_rotations, self.object_offsets)
            
            ### human losses
            losses['human_frame0'] = torch.norm((self.human_gs_frame0['means3D'] - human_gs['means3D'][0]), dim=-1).mean()
            losses['human_initpose'] = torch.norm((self.params['body_pose'] - self.init_body_pose), dim=-1).mean()
            losses['human_smooth'] = get_smooth_loss(self.params, keys=self.cfg.training.smooth_loss.keys, weights=self.cfg.training.smooth_loss.weights)
            losses['human_3dkpt'] = get_3dkpt_loss(human_joints, self.pgt_ketpoints_3d)
            losses['foot_sliding'] = get_foot_sliding_loss(human_vertices, self.scene_depth, self.feet_contacts, self.camera)
            if iter > self.cfg.experiment.stage2_iterations * self.cfg.training.collision.enable_after_iter_ratio:
                losses['collision'] = get_collision_loss(human_vertices, object_gs['means3D'])
            
            loss = sum([loss_weights[k] * v for k, v in losses.items() if not torch.isnan(v)])
            
            with torch.no_grad():
                losses_for_log['loss'] = loss.item()
                for loss_name, loss_value in losses.items():
                    if loss_name not in losses_for_log:
                        losses_for_log[loss_name] = 0.0
                    losses_for_log[loss_name] = loss_value.item() * loss_weights[loss_name]
                if ((iter+1) % self.cfg.training.debug.log_interval == 0):
                    postfix = {
                        k: f"{v:.{3}f}" for k,v in losses_for_log.items()
                    }
                    print()
                    print(postfix)
                
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.save_parameters_stage2()

    @torch.no_grad()
    def render(self):
        Rh, Th = get_human_RT(self.camera['world_view_transform'], self.params['R_emc'], self.params['T_emc'], self.human_scale)
        for frame_id in tqdm(range(0, self.Nframe), desc="Render"):
            smplx_param = {
                "R_emc": self.params['R_emc'][frame_id:frame_id+1],
                "T_emc": self.params['T_emc'][frame_id:frame_id+1],
                "body_pose": self.params['body_pose'][frame_id:frame_id+1],
                "lhand_pose": self.params['lhand_pose'][frame_id:frame_id+1],
                "rhand_pose": self.params['rhand_pose'][frame_id:frame_id+1],
                "jaw_pose": self.params['jaw_pose'][frame_id:frame_id+1],
                "betas": self.params['betas']
            }
            object_param = {
                "art_R": self.params['object_R'][frame_id:frame_id+1],
                "art_T": self.params['object_T'][frame_id:frame_id+1],
                "art_W": self.params['object_W'][None]
            }
            
            human_gs = self.human_gaussian(smplx_param, self.human_gs_cano)[0]
            human_gs['means3D'] = torch.einsum("ij,nj->ni", Rh[frame_id], human_gs['means3D'][0]) + Th[frame_id]
            human_gs['means3D'] = human_gs['means3D'] * self.human_scale
            human_gs['scales'] = human_gs['scales'] + torch.log(self.human_scale) 
            object_gs = self.object_gaussian(object_param, self.object_gs_cano)
            object_gs['means3D'] = object_gs['means3D'][0]
            object_gs = transform_object_gs(object_gs, self.object_scale, self.object_rotations, self.object_offsets)

            all_render = self.gaussian_renderer(human_gs, object_gs, self.scene_bg, cam_param=self.camera)
            gt_img = self.gt_images[frame_id]
            gt_human_mask = self.gt_human_masks[frame_id]
            pred_img = all_render['img'][[2,1,0]]
            pred_human_mask = all_render['mask_human'][0]

            if self.gt_object_masks is not None:
                gt_object_mask = self.gt_object_masks[frame_id]
                pred_object_mask = all_render['mask_object'][0]
                not_overlay_mask = (~((gt_human_mask > 0) & (pred_object_mask.detach() > 0))).float()
                loss_human_mask = ((pred_human_mask - gt_human_mask) ** 2) * not_overlay_mask
                loss_object_mask = ((pred_object_mask - gt_object_mask) ** 2) * not_overlay_mask
            else:
                not_overlay_mask = 1
                pred_object_mask = None
                gt_object_mask = None
            vis_render(pred_img, gt_img, pred_human_mask, pred_object_mask, gt_human_mask, gt_object_mask, not_overlay_mask, save_path=f"{self.cfg.data.paths.results_root}/{self.cfg.scene}/{self.cfg.exp_name}/renders/{frame_id:04d}.png", ratio=1)
                            

    def save_parameters_stage1(self, object_param, frame_id):
        filename = f"{self.cfg.data.paths.results_root}/{self.cfg.scene}/{self.cfg.exp_name}/params/stage1/{frame_id:04d}.pth"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if frame_id == 0:
            torch.save({
                'art_R': object_param['art_R'],
                'art_T': object_param['art_T'],
                'art_W': object_param['art_W'].reshape(-1,1),
            }, filename)
        else:
            torch.save({
                'art_R': object_param['art_R'],
                'art_T': object_param['art_T'],
            }, filename)
        
    def save_parameters_stage2(self):
        filename = f"{self.cfg.data.paths.results_root}/{self.cfg.scene}/{self.cfg.exp_name}/params/stage2.pth"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save({
            'R_emc': self.params['R_emc'],
            'T_emc': self.params['T_emc'],
            'body_pose': self.params['body_pose'],
            'lhand_pose': self.params['lhand_pose'],
            'rhand_pose': self.params['rhand_pose'],
            'jaw_pose': self.params['jaw_pose'],
            'betas': self.params['betas'],
        }, filename)
        
if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    trainer.render()