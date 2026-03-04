import numpy as np
import torch
import copy
import cv2
import math
import os
from segment_anything import SamPredictor, sam_model_registry
import torch_scatter
import frnn
from torch_kmeans import KMeans

from utils.loss_utils import _gather_knn, _accumulate_opacity_into_influence
from configs.config import cfg

class SAGS(object):
    def __init__(self, default_camera, gaussians_cano, gaussians, renderer, results_root="", scene_name=None):
        self.num_samples = cfg.sags.num_samples
        self.gaussians_cano = gaussians_cano
        self.gaussians = gaussians
        self.renderer = renderer
        self.results_root = results_root
        self.scene_name = scene_name
        self.data_root = os.path.join(cfg.data.paths.data_root, self.scene_name)
        self.device = self.gaussians['means3D'].device

        sam = sam_model_registry['vit_h'](checkpoint=cfg.data.paths.sam_ckpt_path).to('cuda')
        self.predictor = SamPredictor(sam)
        self.cameras = [default_camera]
        
    def project(self, camera, points_mask):
        xyz = self.gaussians['means3D'][points_mask]
        full_proj_transform = camera['full_proj_transform']
        # project Gaussians to 2D
        projected_unhomog = xyz @ full_proj_transform[:3, :3]
        projected_unhomog = projected_unhomog + full_proj_transform[3:4, :3]
        reg = torch.sum(xyz * full_proj_transform[:3, 3].reshape(1, 3), dim=-1) + full_proj_transform[3, 3]
        projected_homogenous = projected_unhomog / (projected_unhomog[:, -1:] + 1e-4)
        projected_homogenous[:, 0] = ((projected_homogenous[:, 0] + 1.0) * camera['image_width'] - 1.0) * 0.5  # convert from NDC coords to pixel space
        projected_homogenous[:, 1] = ((projected_homogenous[:, 1] + 1.0) * camera['image_height'] - 1.0) * 0.5
        points_2d = projected_homogenous[:, :2]
        return points_2d
    
    def prompt(self, render_image, mask_id, dynamic_points=None, static_points=None, dynamic_box=None):
        self.predictor.set_image(render_image)
        sam_feature = self.predictor.features
        if dynamic_box is not None:
            dynamic_box = dynamic_box.detach().cpu().numpy()
        input_points = torch.cat([dynamic_points, static_points], dim=0).detach().cpu().numpy()
        input_labels = torch.cat([torch.ones(dynamic_points.shape[0]), torch.zeros(static_points.shape[0])], dim=0).detach().cpu().numpy()

        self.predictor.features = sam_feature
        masks, _, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=dynamic_box,
            multimask_output=True,
        )
        return_mask = (masks[mask_id, :, :, None]*255).astype(np.uint8)

        return return_mask / 255

    def inverse_project(self, camera, sam_mask, src_particles=None, tgt_particles=None, render_image=None):
        xyz = self.gaussians['means3D']
        scales = self.gaussians['scales']
        opacity = self.gaussians['opacities']
        device = self.device
        K = cfg.sags.knn.K
        r = cfg.sags.knn.r
        
        def fov2focal(fov, pixels):
            return pixels / (2 * math.tan(fov / 2))
        
        # project Gaussians to 2D
        full_proj_transform = camera['full_proj_transform']
        projected_unhomog = xyz @ full_proj_transform[:3, :3]
        projected_unhomog = projected_unhomog + full_proj_transform[3:4, :3]
        reg = torch.sum(xyz * full_proj_transform[:3, 3].reshape(1, 3), dim=-1) + full_proj_transform[3, 3]
        projected_homogenous = projected_unhomog / (projected_unhomog[:, -1:] + 1e-4)
        depths = reg.clone()  # don't do normalized depths
        projected_homogenous[:, 0] = ((projected_homogenous[:, 0] + 1.0) * camera['image_width'] - 1.0) * 0.5  # convert from NDC coords to pixel space
        projected_homogenous[:, 1] = ((projected_homogenous[:, 1] + 1.0) * camera['image_height'] - 1.0) * 0.5
        gs_2d = projected_homogenous[:, :2]

        scales_1d = torch.exp(scales).max(-1).values
        cov_1d = scales_1d ** 2
        conical_x = cov_1d * (fov2focal(camera['FoVx'], camera['image_width']) / (reg + 0.0001)) ** 2 + 0.3
        conical_y = cov_1d * (fov2focal(camera['FoVy'], camera['image_height']) / (reg + 0.0001)) ** 2 + 0.3
        det = conical_x * conical_y
        conical_x, conical_y = conical_x / det, conical_y / det
        conical_xy = torch.zeros_like(conical_x)
        conical_opacity = torch.stack([conical_x, conical_xy, conical_y, torch.sigmoid(opacity).flatten()], dim=-1)
        
        sam_2d_points = torch.where(sam_mask > 0)
        sam_2d_points = torch.stack((sam_2d_points[1], sam_2d_points[0]), dim=1).float()
        
        lengths1 = torch.tensor([sam_2d_points.size(0)]).to(device)
        lengths2 = torch.tensor([gs_2d.size(0)]).to(device)
        dists2, raw_knn_idxs, nn, grid = \
            frnn.frnn_grid_points(sam_2d_points.unsqueeze(0), gs_2d.unsqueeze(0), lengths1, lengths2, K, r, grid=None, return_nn=False, return_sorted=True)
        sam_2d_points_gather, means2D_src_gather, depths_src_gather, conical_opacity, found_mask, knn_idxs, particles_idxs = \
        _gather_knn(raw_knn_idxs, depths, sam_2d_points, gs_2d, conical_opacity, K=K)
        dists = (means2D_src_gather - sam_2d_points_gather)  # MK x 2
        influences, dists = _accumulate_opacity_into_influence(particles_idxs, dists, conical_opacity, depths_src_gather, found_mask, continue_depth=True)
        
        per_gs_influences = torch_scatter.scatter_sum(influences[found_mask], index=knn_idxs[found_mask], dim_size=gs_2d.shape[0])
        points_mask = (per_gs_influences > cfg.sags.influence_threshold)
        
        ### try to get quasi-pairs
        valid_mask = sam_mask[src_particles[:, 1].long(), src_particles[:, 0].long()] > 0.5
        src_particles = src_particles[valid_mask]
        tgt_particles = tgt_particles[valid_mask]
        
        flow2d = src_particles - tgt_particles
        
        motions = (flow2d[:, 0] ** 2 + flow2d[:, 1] ** 2)
        static_mask = (motions < max(torch.quantile(motions, cfg.sags.motion.quantile_threshold), cfg.sags.motion.min_motion_threshold))

        static_points = src_particles[static_mask]
        ##### vis static points
        def draw_static_points(static_mask):
            import cv2
            img = sam_mask.detach().cpu().numpy() * 255
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for point in src_particles[static_mask]:
                cv2.circle(img, (int(point[0]), int(point[1])), cfg.sags.visualization.circle_radius, (255, 0, 255), -1)
            cv2.imwrite(os.path.join(self.results_root, 'static_points.png'), img)
            print(f"Static points visualization saved to {os.path.join(self.results_root, 'static_points.png')}!")
        draw_static_points(static_mask)
        #####
        dynamic_gs_2d = gs_2d[points_mask]
        static_gs_2d = gs_2d[~points_mask]
        dynamic_gs_3d = self.gaussians['means3D'][points_mask]
        static_gs_3d = self.gaussians['means3D'][~points_mask]
    
        dists, raw_knn_idxs, nn, grid = frnn.frnn_grid_points(
            static_points.unsqueeze(0), dynamic_gs_2d.unsqueeze(0), 
            torch.tensor([static_points.size(0)]).to(device), torch.tensor([dynamic_gs_2d.size(0)]).to(device), 
            cfg.sags.quasi_static.search_k, cfg.sags.quasi_static.search_r, grid=None, return_nn=False, return_sorted=True
        )
        quasi_static_gs_idx = torch.unique(raw_knn_idxs[raw_knn_idxs > 0])
        quasi_static_gs_idx = torch.where(points_mask)[0][quasi_static_gs_idx]
        quasi_static_gs = self.gaussians['means3D'][quasi_static_gs_idx]
        
        N_pairs = cfg.sags.quasi_static.n_pairs
        dists, raw_knn_idxs, nn, grid = frnn.frnn_grid_points(
            quasi_static_gs.unsqueeze(0), static_gs_3d.unsqueeze(0), 
            torch.tensor([quasi_static_gs.size(0)]).to(device), torch.tensor([static_gs_3d.size(0)]).to(device), 
            N_pairs, cfg.sags.quasi_static.pair_distance, grid=None, return_nn=False, return_sorted=True
        )
        quasi_static_gs_idx = quasi_static_gs_idx[:, None].tile((1, N_pairs))
        quasi_static_gs_idx = quasi_static_gs_idx[raw_knn_idxs[0] > 0]
        
        quasi_static_gs_bind_idx = raw_knn_idxs[raw_knn_idxs > 0]
        quasi_static_gs_bind_idx = torch.where(~points_mask)[0][quasi_static_gs_bind_idx]
        
        quasi_static_pairs = torch.cat([quasi_static_gs_idx[:, None], quasi_static_gs_bind_idx[:, None]], dim=1)
        ##### vis static points
        import cv2
        img = sam_mask.detach().cpu().numpy() * 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        for pair in quasi_static_pairs:
            point1 = gs_2d[pair[0]].cpu().numpy()
            point2 = gs_2d[pair[1]].cpu().numpy()
            cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0, 255, 0), cfg.sags.visualization.line_thickness)
        cv2.imwrite(os.path.join(self.results_root, 'pairs.png'), img)
        print(f"Pair points visualization saved to {os.path.join(self.results_root, 'pairs.png')}!")
        return points_mask, quasi_static_pairs

    def vis_sam_mask(self, sam_mask, render_image, dynamic_points, static_points, dynamic_box=None):
        # Draw points
        render_image = render_image.copy()
        sam_mask = sam_mask.detach().cpu().numpy()
        for point in dynamic_points:
            cv2.circle(render_image, (int(point[0]), int(point[1])), cfg.sags.visualization.point_radius, (0, 0, 255), -1)
        for point in static_points:
            cv2.circle(render_image, (int(point[0]), int(point[1])), cfg.sags.visualization.point_radius, (255, 0, 0), -1)
        # Draw dynamic box
        if dynamic_box is not None:
            dynamic_box = dynamic_box.detach().cpu().numpy()
            dynamic_box = dynamic_box.astype(np.int32)
            cv2.rectangle(render_image, (dynamic_box[0], dynamic_box[1]), (dynamic_box[2], dynamic_box[3]), (0, 255, 0), 2)
        # Draw SAM mask
        sam_mask = sam_mask * 255
        sam_mask = sam_mask.astype(np.uint8)
        sam_mask = cv2.cvtColor(sam_mask, cv2.COLOR_GRAY2BGR)
        render_image = cv2.addWeighted(render_image, cfg.sags.visualization.alpha, sam_mask, cfg.sags.visualization.alpha, 0)
        cv2.imwrite(os.path.join(self.results_root, 'vis_sam_mask.png'), render_image)
        print(f"SAM mask visualization saved to {os.path.join(self.results_root, 'vis_sam_mask.png')}!")

    def vis_sam_prompts(self, render_image, dynamic_points, static_points):
        # Draw points
        render_image = render_image.copy()
        for point in dynamic_points:
            cv2.circle(render_image, (int(point[0]), int(point[1])), cfg.sags.visualization.point_radius, (0, 0, 255), -1)
        for point in static_points:
            cv2.circle(render_image, (int(point[0]), int(point[1])), cfg.sags.visualization.point_radius, (255, 0, 0), -1)
        cv2.imwrite(os.path.join(self.results_root, 'vis_sam_prompts.png'), render_image)
        print(f"SAM prompts visualization saved to {os.path.join(self.results_root, 'vis_sam_prompts.png')}!")
        
    def fill_holes(self, mask, kernel_size=None):
        if kernel_size is None:
            kernel_size = cfg.sags.morphology.kernel_size
        # Fill holes in the mask
        mask = (mask.detach().cpu().numpy()[:, :, None] * 255).astype(np.uint8)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = torch.from_numpy(mask).squeeze(-1).to(self.device) > 0
        return mask.long()
    
    @torch.no_grad()
    def run(self, dynamic_data=None, mask_id=2):
        print("Start Segmentation...")
        src_particles = dynamic_data['src_particles'].to(self.device)
        tgt_particles = dynamic_data['tgt_particles'].to(self.device)
        dynamic_points = dynamic_data['dynamic_prompts_2d'].to(self.device).float()
        static_points = dynamic_data['static_prompts_2d'].to(self.device).float()

        camera = self.cameras[0]
        render_pkg = self.renderer.simple_render(self.gaussians, camera)
        render_image = render_pkg["img"].permute(1, 2, 0).detach().cpu().numpy()
        render_image = (255 * np.clip(render_image, 0, 1)).astype(np.uint8)
        
        self.vis_sam_prompts(render_image, dynamic_points, static_points)

        dynamic_points = KMeans(n_clusters=min(cfg.sags.clustering.dynamic_max_clusters, dynamic_points.shape[0] - 1))(dynamic_points[None])[1][0]
        static_points = KMeans(n_clusters=min(cfg.sags.clustering.static_max_clusters, static_points.shape[0] - 1))(static_points[None])[1][0]
        
        # Generate new SAM mask
        sam_mask = self.prompt(render_image, mask_id=mask_id, dynamic_points=dynamic_points, static_points=static_points, dynamic_box=None)
        sam_mask = torch.from_numpy(sam_mask).squeeze(-1).to(self.device).long()
        sam_mask = self.fill_holes(sam_mask)

        self.vis_sam_mask(sam_mask, render_image, dynamic_points, static_points)
        point_mask, quasi_static_pairs = self.inverse_project(camera, sam_mask, src_particles, tgt_particles, render_image)
        return {
            "dynamic": point_mask,
            "static": ~point_mask,
            "quasi_static_pairs": quasi_static_pairs,
        }
