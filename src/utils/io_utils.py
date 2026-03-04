import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement

def data_to_device(data, device=torch.device("cuda:0")):
    for k, v in data.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, torch.Tensor):
                    data[k][k2] = v2.to(device).float()
                elif isinstance(v2, np.ndarray):
                    data[k][k2] = torch.from_numpy(v2).to(device).float()
                elif isinstance(v2, list) and (len(v2) > 0) and not isinstance(v2[0], str):
                    data[k][k2] = torch.from_numpy(np.array(v2)).to(device).float()
        elif isinstance(v, torch.Tensor):
            data[k] = v.to(device)
        elif isinstance(v, np.ndarray):
            data[k] = torch.from_numpy(v).to(device).float()
        elif isinstance(v, list) and (len(v) > 0) and not isinstance(v[0], str):
            data[k] = torch.from_numpy(np.array(v)).to(device).float()
    return data

def load_ply(path):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    new_scales = np.zeros_like(scales)
    new_scales[:] = scales.mean()
    
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))
    return {
        "means3D": xyz,
        "opacities": opacities,
        "scales": scales,
        "rotations": rots,
        "features_dc": features_dc,
        "features_rest": features_extra
    }