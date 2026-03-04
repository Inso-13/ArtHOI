import copy
import smplx
import torch
import torch.nn as nn
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops import knn_points
from smplx.lbs import batch_rigid_transform
from models.smpl_x import smpl_x

class HumanGaussian(nn.Module):
    @torch.no_grad()
    def __init__(self, gender="neutral"):
        super(HumanGaussian, self).__init__()
        self.smplx_layer = copy.deepcopy(smpl_x.layer[gender]).cuda()
        
        # upsample mesh and other assets
        skinning_weight_orig = self.smplx_layer.lbs_weights.float()
        pose_dirs_orig = self.smplx_layer.posedirs.permute(1,0).reshape(smpl_x.vertex_num,3*(smpl_x.joint_num-1)*9)
        expr_dirs_orig = self.smplx_layer.expr_dirs.view(smpl_x.vertex_num,3*smpl_x.expr_param_dim)
        is_rhand_orig, is_lhand_orig, is_face_orig, is_face_expr_orig = torch.zeros((smpl_x.vertex_num,1)).float().cuda(), torch.zeros((smpl_x.vertex_num,1)).float().cuda(), torch.zeros((smpl_x.vertex_num,1)).float().cuda(), torch.zeros((smpl_x.vertex_num,1)).float().cuda()
        is_rhand_orig[smpl_x.rhand_vertex_idx], is_lhand_orig[smpl_x.lhand_vertex_idx], is_face_orig[smpl_x.face_vertex_idx], is_face_expr_orig[smpl_x.expr_vertex_idx] = 1.0, 1.0, 1.0, 1.0
        is_cavity_orig = torch.FloatTensor(smpl_x.is_cavity).cuda()[:,None]
        _, skinning_weight, pose_dirs, expr_dirs, is_rhand, is_lhand, is_face, is_face_expr, is_cavity = smpl_x.upsample_mesh(torch.ones((smpl_x.vertex_num,3)).float().cuda(), [skinning_weight_orig, pose_dirs_orig, expr_dirs_orig, is_rhand_orig, is_lhand_orig, is_face_orig, is_face_expr_orig, is_cavity_orig]) # upsample with dummy vertex
        is_rhand_orig, is_lhand_orig, is_face_orig, is_face_expr_orig = is_rhand_orig[:,0] > 0, is_lhand_orig[:,0] > 0, is_face_orig[:,0] > 0, is_face_expr_orig[:,0] > 0
        is_rhand, is_lhand, is_face, is_face_expr = is_rhand[:,0] > 0, is_lhand[:,0] > 0, is_face[:,0] > 0, is_face_expr[:,0] > 0

        self.register_buffer('skinning_weight_orig', skinning_weight_orig)
        self.register_buffer('is_rhand_orig', is_rhand_orig)
        self.register_buffer('is_lhand_orig', is_lhand_orig)
        self.register_buffer('is_face_orig', is_face_orig)
        self.register_buffer('skinning_weight', skinning_weight)
        self.register_buffer('is_rhand', is_rhand)
        self.register_buffer('is_lhand', is_lhand)
        self.register_buffer('is_face', is_face)

    def get_neutral_pose_human(self, betas, jaw_zero_pose):
        zero_pose = torch.zeros((1,3)).float().cuda()
        neutral_body_pose = smpl_x.neutral_body_pose.view(1,-1).cuda() # T-pose
        zero_hand_pose = torch.zeros((1,len(smpl_x.joint_part['lhand'])*3)).float().cuda()
        zero_expr = torch.zeros((1,smpl_x.expr_param_dim)).float().cuda()
        if jaw_zero_pose:
            jaw_pose = torch.zeros((1,3)).float().cuda()
        else:
            jaw_pose = smpl_x.neutral_jaw_pose.view(1,3).cuda() # open mouth
        output = self.smplx_layer(global_orient=zero_pose, body_pose=neutral_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, betas=betas)
        
        mesh_neutral_pose = output.vertices[0] # T-pose human
        mesh_neutral_pose_upsampled = smpl_x.upsample_mesh(mesh_neutral_pose) # T-pose human
        joint_neutral_pose = output.joints[0][:smpl_x.joint_num,:] # T-pose human

        # compute transformation matrix for making T-pose to zero pose
        neutral_body_pose = neutral_body_pose.view(len(smpl_x.joint_part['body'])-1,3)
        zero_hand_pose = zero_hand_pose.view(len(smpl_x.joint_part['lhand']),3)
        neutral_body_pose_inv = matrix_to_axis_angle(torch.inverse(axis_angle_to_matrix(neutral_body_pose)))
        jaw_pose_inv = matrix_to_axis_angle(torch.inverse(axis_angle_to_matrix(jaw_pose)))
        pose = torch.cat((zero_pose, neutral_body_pose_inv, jaw_pose_inv, zero_pose, zero_pose, zero_hand_pose, zero_hand_pose)) 
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_neutral_pose = batch_rigid_transform(pose[None,:,:,:], joint_neutral_pose[None,:,:], self.smplx_layer.parents)
        transform_mat_neutral_pose = transform_mat_neutral_pose[0]
        return mesh_neutral_pose_upsampled, mesh_neutral_pose, joint_neutral_pose, transform_mat_neutral_pose

    def get_zero_pose_human(self, betas):
        bs = betas.shape[0]
        zero_pose = torch.zeros((bs,3)).float().cuda()
        zero_body_pose = torch.zeros((bs,(len(smpl_x.joint_part['body'])-1)*3)).float().cuda()
        zero_hand_pose = torch.zeros((bs,len(smpl_x.joint_part['lhand'])*3)).float().cuda()
        zero_expr = torch.zeros((bs,smpl_x.expr_param_dim)).float().cuda()
        betas = betas.reshape(bs, 10)
        output = self.smplx_layer(global_orient=zero_pose, body_pose=zero_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, betas=betas)
        
        joint_zero_pose = output.joints[:, :smpl_x.joint_num,:] # zero pose human
        return joint_zero_pose

    def get_transform_mat_joint_bs(self, transform_mat_neutral_pose, joint_zero_pose, smplx_param):
        # 1. T-pose -> zero pose
        bs = smplx_param['body_pose'].shape[0]
        transform_mat_joint_1 = torch.tile(transform_mat_neutral_pose, (bs,1,1,1))
        # 2. zero pose -> image pose
        body_pose = smplx_param['body_pose'].view(bs,len(smpl_x.joint_part['body'])-1,3)
        jaw_pose = smplx_param['jaw_pose'].view(bs,1,3)
        lhand_pose = smplx_param['lhand_pose'].view(bs,len(smpl_x.joint_part['lhand']),3)
        rhand_pose = smplx_param['rhand_pose'].view(bs,len(smpl_x.joint_part['rhand']),3)
        root_pose = torch.zeros((bs, 1, 3)).to(body_pose)
        leye_pose = torch.zeros((bs, 1, 3)).to(body_pose)
        reye_pose = torch.zeros((bs, 1, 3)).to(body_pose)
        
        # forward kinematitcs
        pose = torch.cat((root_pose, body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose), dim=1) 
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_joint_2 = batch_rigid_transform(pose, joint_zero_pose, self.smplx_layer.parents)
        
        # 3. combine 1. T-pose -> zero pose and 2. zero pose -> image pose
        transform_mat_joint = torch.bmm(transform_mat_joint_2.reshape(-1, 4, 4), transform_mat_joint_1.reshape(-1, 4, 4))
        return transform_mat_joint.reshape(bs, -1, 4, 4)

        
    def forward(self, smplx_param, gs_cano=None, return_joints=False, return_vertices=False):
        bs = smplx_param['body_pose'].shape[0]
        betas = torch.zeros((bs, 10)).to(smplx_param['body_pose'])
        mesh_neutral_pose, mesh_neutral_pose_wo_upsample, _, transform_mat_neutral_pose = self.get_neutral_pose_human(betas[0:1], jaw_zero_pose=True)
        joint_zero_pose = self.get_zero_pose_human(betas)

        mean_3d = gs_cano['means3D'].reshape(-1, 3)
        
        # for hands and face, assign original vertex index to use sknning weight of the original vertex
        nn_vertex_idxs = knn_points(mean_3d[None,:,:], mesh_neutral_pose_wo_upsample[None,:,:], K=1, return_nn=True).idx[0,:,0] # dimension: smpl_x.vertex_num_upsampled
        mask = (self.is_rhand + self.is_lhand + self.is_face) > 0
        nn_vertex_idxs[mask] = torch.arange(smpl_x.vertex_num_upsampled).cuda()[mask]

        # get transformation matrix of the nearest vertex and perform lbs
        transform_mat_joint = self.get_transform_mat_joint_bs(transform_mat_neutral_pose, joint_zero_pose, smplx_param)
        transform_mat_vertex = torch.einsum("vj,bjmn->bvmn", self.skinning_weight[nn_vertex_idxs,:], transform_mat_joint)
        
        # lbs
        mean_3d = torch.cat((mean_3d, torch.ones_like(mean_3d[..., :1])),dim=-1)
        mean_3d = torch.einsum("bvmn,vn->bvm", transform_mat_vertex, mean_3d)[..., :3]
        
        transform_mat = torch.matmul(self.skinning_weight_orig[None], transform_mat_joint.view(bs,smpl_x.joint_num,16)).view(bs,-1,4,4)
        smpl_vertices = torch.cat((mesh_neutral_pose_wo_upsample, torch.ones_like(mesh_neutral_pose_wo_upsample[..., :1])),dim=-1)
        smpl_vertices = torch.einsum("bvmn,vn->bvm", transform_mat, smpl_vertices)[..., :3]
        
        ret = [{
            'means3D': mean_3d, 
            'opacities': gs_cano['opacities'], 
            'scales': gs_cano['scales'], 
            'rotations': gs_cano['rotations'].reshape(-1, 4), 
            'features_dc': gs_cano['features_dc'],
        }]
        if return_joints:
            joints = torch.einsum("jn,bnx->bjx", self.smplx_layer.J_regressor, smpl_vertices)
            ret.append(joints)
        if return_vertices:
            ret.append(smpl_vertices)
        return ret