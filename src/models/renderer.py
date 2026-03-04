import torch
import torch.nn as nn

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.vis_utils import SH2RGB

class GaussianRenderer(nn.Module):
    def __init__(self, return_object_mask=True):
        super(GaussianRenderer, self).__init__()
        self.return_object_mask = return_object_mask
    
    def forward(self, human_gs=None, object_gs=None, scene_bg=None, cam_param=None):
        assert cam_param is not None

        gs_list = []
        if human_gs is not None:
            gs_list.append(human_gs)
        if object_gs is not None:
            gs_list.append(object_gs)

        # assets for the rendering
        mean_3d = torch.cat([gs['means3D'] for gs in gs_list], dim=0)
        opacity = torch.cat([gs['opacities'] for gs in gs_list], dim=0)
        scale = torch.cat([gs['scales'] for gs in gs_list], dim=0)
        rotation = torch.cat([gs['rotations'] for gs in gs_list], dim=0)
        features_dc = torch.cat([gs['features_dc'] for gs in gs_list], dim=0)
        rgb = SH2RGB(features_dc).reshape(-1, 3)

        raster_settings = GaussianRasterizationSettings(
            image_height=cam_param['image_height'],
            image_width=cam_param['image_width'],
            tanfovx=float(cam_param['FoVx']),
            tanfovy=float(cam_param['FoVy']),
            bg=torch.ones((3)).float().cuda(), 
            scale_modifier=1.0,
            viewmatrix=cam_param['world_view_transform'], 
            projmatrix=cam_param['full_proj_transform'],
            sh_degree=0, # dummy sh degree. as rgb values are already computed, rasterizer does not use this one
            campos=cam_param['camera_center'],
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # prepare Gaussian position in the image space for the gradient tracking
        point_num = mean_3d.shape[0]
        mean_2d = torch.zeros((point_num,3)).float().cuda()
        mean_2d.requires_grad = True
        mean_2d.retain_grad()
        
        # rasterize visible Gaussians to image and obtain their radius (on screen). 
        render_img, radius, render_depthmap, render_mask = rasterizer(
            means3D=mean_3d,
            means2D=mean_2d,
            shs=None,
            colors_precomp=rgb,
            opacities=torch.sigmoid(opacity),
            scales=torch.exp(scale),
            rotations=rotation,
            cov3D_precomp=None)
        if scene_bg is not None:
            render_img = render_img * render_mask + scene_bg * (1-render_mask)
        ret = {
            'img': render_img,
            'depthmap': render_depthmap,
            'mask': render_mask,
            'mean_2d': mean_2d,
            'radius': radius
        }
        if self.return_object_mask and (human_gs is not None) and (object_gs is not None):
            ## human
            mean_3d = human_gs['means3D']
            opacity = human_gs['opacities']
            scale = human_gs['scales']
            rotation = human_gs['rotations']
            features_dc = human_gs['features_dc']
            rgb = SH2RGB(features_dc).reshape(-1, 3)
            # prepare Gaussian position in the image space for the gradient tracking
            point_num = mean_3d.shape[0]
            mean_2d = torch.zeros((point_num,3)).float().cuda()
            mean_2d.requires_grad = True
            mean_2d.retain_grad()
            
            # rasterize visible Gaussians to image and obtain their radius (on screen). 
            render_img_human, _, render_depthmap_human, render_mask_human = rasterizer(
                means3D=mean_3d,
                means2D=mean_2d,
                shs=None,
                colors_precomp=rgb,
                opacities=torch.sigmoid(opacity),
                scales=torch.exp(scale),
                rotations=rotation,
                cov3D_precomp=None)
            ret.update({
                'img_human': render_img_human,
                'depthmap_human': render_depthmap_human,
                'mask_human': render_mask_human,
            })
            
            ## object
            mean_3d = object_gs['means3D']
            opacity = object_gs['opacities']
            scale = object_gs['scales']
            rotation = object_gs['rotations']
            features_dc = object_gs['features_dc']
            rgb = SH2RGB(features_dc).reshape(-1, 3)
            # prepare Gaussian position in the image space for the gradient tracking
            point_num = mean_3d.shape[0]
            mean_2d = torch.zeros((point_num,3)).float().cuda()
            mean_2d.requires_grad = True
            mean_2d.retain_grad()
            
            # rasterize visible Gaussians to image and obtain their radius (on screen). 
            render_img_object, _, render_depthmap_object, render_mask_object = rasterizer(
                means3D=mean_3d,
                means2D=mean_2d,
                shs=None,
                colors_precomp=rgb,
                opacities=torch.sigmoid(opacity),
                scales=torch.exp(scale),
                rotations=rotation,
                cov3D_precomp=None)
            ret.update({
                'img_object': render_img_object,
                'depthmap_object': render_depthmap_object,
                'mask_object': render_mask_object,
            })
        return ret
    
    def simple_render(self, pc, cam_param):
            
        # assets for the rendering
        mean_3d = pc['means3D']
        opacity = pc['opacities']
        scale = pc['scales']
        rotation = pc['rotations']
        features_dc = pc['features_dc']
        rgb = SH2RGB(features_dc).reshape(-1, 3)

        raster_settings = GaussianRasterizationSettings(
            image_height=cam_param['image_height'],
            image_width=cam_param['image_width'],
            tanfovx=float(cam_param['FoVx']),
            tanfovy=float(cam_param['FoVy']),
            bg=torch.ones((3)).float().cuda(), 
            scale_modifier=1.0,
            viewmatrix=cam_param['world_view_transform'], 
            projmatrix=cam_param['full_proj_transform'],
            sh_degree=0, # dummy sh degree. as rgb values are already computed, rasterizer does not use this one
            campos=cam_param['camera_center'],
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # prepare Gaussian position in the image space for the gradient tracking
        point_num = mean_3d.shape[0]
        mean_2d = torch.zeros((point_num,3)).float().cuda()
        mean_2d.requires_grad = True
        mean_2d.retain_grad()
        
        # rasterize visible Gaussians to image and obtain their radius (on screen). 
        render_img, radius, render_depthmap, render_mask = rasterizer(
            means3D=mean_3d,
            means2D=mean_2d,
            shs=None,
            colors_precomp=rgb,
            opacities=torch.sigmoid(opacity),
            scales=torch.exp(scale),
            rotations=rotation,
            cov3D_precomp=None)
        ret = {
            'img': render_img,
            'depthmap': render_depthmap,
            'mask': render_mask,
            'mean_2d': mean_2d,
            'radius': radius
        }
        return ret
