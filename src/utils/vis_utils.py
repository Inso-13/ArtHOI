import os
import torch
import numpy as np
import cv2
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5
    
@torch.no_grad()
def vis_render(image, gt_image, pred_human_mask, pred_object_mask, gt_human_mask, gt_object_mask, not_overlay_mask, save_path=None, ratio=1):
    vis_seg_object = (pred_object_mask is not None) and (gt_object_mask is not None)
    vis_seg_human = (pred_human_mask is not None) and (gt_human_mask is not None)
    vis_rgb = (gt_image is not None) and (image is not None)
    
    _3, H, W = gt_image.shape

    nH = int(vis_rgb) + int(vis_seg_human) + int(vis_seg_object)
    save_image = np.zeros((H*nH, W*2, 3)).astype(np.float32)
    iH = 0
    
    if vis_rgb:
        save_image[iH:iH+H, :W] = image.detach().cpu().numpy().transpose(1,2,0)
        save_image[iH:iH+H, W:] = gt_image.detach().cpu().numpy().transpose(1,2,0)
        iH += H

    if vis_seg_human:
        save_image[iH:iH+H, :W, :] = (pred_human_mask * not_overlay_mask).detach().cpu().numpy()[..., None]
        save_image[iH:iH+H, W:, :] = (gt_human_mask * not_overlay_mask).detach().cpu().numpy()[..., None]
        iH += H

    if vis_seg_object:
        save_image[iH:iH+H, :W, :] = (pred_object_mask * not_overlay_mask).detach().cpu().numpy()[..., None]
        save_image[iH:iH+H, W:, :] = (gt_object_mask * not_overlay_mask).detach().cpu().numpy()[..., None]
        iH += H

    save_image = np.clip(save_image, 0.0, 1.0)
    save_image = (save_image*255).astype(np.uint8)
    if ratio != 1:
        save_image = cv2.resize(save_image, (int(save_image.shape[1]*ratio), int(save_image.shape[0]*ratio)))
        
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, save_image)
    else:
        cv2.imwrite("debug.png", save_image)