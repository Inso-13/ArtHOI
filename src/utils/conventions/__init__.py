from .smplx_118 import SMPLX_118
from .smplx_55 import SMPLX_55

def get_mapping(src_names,
                dst_names):
    dst_idxs, src_idxs, intersection = [], [], []
    unmapped_names, approximate_names = [], []
    for dst_idx, dst_name in enumerate(dst_names):
        try:
            src_idx = src_names.index(dst_name)
        except ValueError:
            src_idx = -1
        if src_idx >= 0:
            dst_idxs.append(dst_idx)
            src_idxs.append(src_idx)
            intersection.append(dst_name)

        if unmapped_names:
            warn_message = \
                f'Approximate mapping {unmapped_names}' +\
                f' to {approximate_names}'
            print(msg=warn_message)

    mapping_list = [dst_idxs, src_idxs, intersection]

    return mapping_list[:3]