import os
import torch
import numpy as np
import torch.nn as nn
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix
from utils.io_utils import load_ply

class ObjectGaussian(nn.Module):
    def __init__(self, model_path):
        super(ObjectGaussian, self).__init__()
        self.use_articulation = False
        self.gaussians = load_ply(model_path)
        self.W = torch.ones((self.gaussians['means3D'].shape[0], 1)).cuda().float()
        # self.N_joints = self.W.shape[1]
        self.N_joints = 1
        self.N_vertices = self.gaussians['means3D'].shape[0]
        # self.vertices = torch.from_numpy(self.gaussians['means3D']).cuda().float()
        self.static_T = torch.zeros((1,3)).cuda().float()
        self.static_R = torch.eye(3)[None].cuda().float()
        
    def forward(self, articulation_params, gs_cano):
        R_6d = articulation_params["art_R"].cuda()
        T = articulation_params["art_T"].cuda()
        R = rotation_6d_to_matrix(R_6d)
        # W = self.W
        W = torch.sigmoid(articulation_params["art_W"])
        trans = torch.einsum('bnj, bjx -> bnx', W, T) + torch.einsum('bnj, jx -> bnx', 1-W, self.static_T)
        frame = torch.einsum('bnj, bjrc -> bnrc', W, R) + torch.einsum('bnj, jrc -> bnrc', 1-W, self.static_R) # frame: N, 3, 3
        mean_3d = torch.einsum('bnjk, nk -> bnj', frame, gs_cano['means3D']) + trans
        return {
            'means3D': mean_3d, 
            'opacities': gs_cano['opacities'], 
            'scales': gs_cano['scales'], 
            'rotations': gs_cano['rotations'].reshape(-1, 4), 
            'features_dc': gs_cano['features_dc']
        }
        
    
    