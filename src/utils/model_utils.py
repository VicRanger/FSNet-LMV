import re
from utils.parser_utils import parse_buffer_name
from utils.log import log
import torch
import torch.nn as nn

def fix_the_size_with_dec(enc, dec):
    for ref_id in [2, 3]:
        if dec.shape[ref_id] != enc.shape[ref_id]:
            dec = dec.narrow(ref_id, 0, enc.shape[ref_id])
    return dec

def fix_the_size_with_dec_and_flow(enc, dec, flows=[]):
    for ref_id in [2, 3]:
        if dec.shape[ref_id] != enc.shape[ref_id]:
            dec = dec.narrow(ref_id, 0, enc.shape[ref_id])
            for i in range(len(flows)):
                flows[i] = flows[i].narrow(ref_id, 0, enc.shape[ref_id])
    return dec, flows

def retain_bn_float(net: nn.Module): 
    if isinstance(net, torch.nn.modules.batchnorm._BatchNorm) and net.affine is True:
        net.float()
    for child in net.children():
        retain_bn_float(child)
    return net
    

def model_to_half(net):
    net = net.half()
    return retain_bn_float(net)

def get_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total

dim2d_dict = {
    'base_color': 3,
    'brdf_color': 3,
    'dmdl_color': 3,
    'shadow': 3,
    'shadow_y': 1,
    "metallic": 1,
    "specular": 1,
    "roughness": 1,
    "nov": 1,
    "stencil": 1,
    "world_normal": 3,
    "world_position": 3,
    "camera_normal": 3,
    "camera_position": 3,
    "depth": 1,
    "shadow_mask": 1,
    "occlusion_mask": 1,
    "motion_vector": 2,
    "skybox_mask": 1,
    "discontinuity_mask": 1,
    "shadow_discontinuity_mask": 1,
    "metadata": 0,
    "sky_color": 3,
    "history_sky_color": 3,
    "black3": 3,
    "white1": 1,
    "history_sky_color": 3,
    "st_color": 3,
    "history_st_color": 3,
    "st_alpha": 1,
    "history_motion_vector": 2,
    "history_base_color": 3,
    "history_roughness": 1,
    "history_brdf_color": 3,
    "history_dmdl_color": 3,
    "history_depth": 1,
    "merged_motion_vector": 2,
    # "history_gt": 3,
    # "history_gt_no_shadow": 3,
    # "history_scene_color": 3,
    "history_occlusion_mask": 1,
    "history_st_alpha": 1,
    # "history_warped_scene_color": 3,
    # "history_warped_gt": 3,
    "history_warped_motion_vector": 2,
    "history_warped_world_normal": 3,
    "history_warped_dmdl_color": 3,
    "history_warped_depth": 1,
    # "occlusion_warped_scene_color": 3,
    # "history_warped_scene_color_mv": 2,
    "history_occlusion_mask_extranet": 1,
    "occlusion_motion_vector": 2,
    "history_occlusion_warped_scene_color_extranet": 3,
    "history_masked_warped_scene_color_extranet": 3,
    "history_masked_occlusion_warped_scene_color_extranet": 3,
}


def get_2d_dim(item):
    if ('scene_color' in item or 'scene_light' in item):
        return 3
    buffer_name = parse_buffer_name(item)['buffer_name']
    if buffer_name not in dim2d_dict.keys():
        # log.warn("{} isnt in dim_dict, set dim = 0".format(item))
        # return 0
        raise KeyError("{} isnt in dim_dict.".format(buffer_name))
    return dim2d_dict[buffer_name]


dim1d_dict = {
    "camera__position": 3,
    "camera__forward": 3,
    "light__directional_light_world_position": 3,
    "light__directional_light_camera_position": 3,
    "light__directional_light_world_direction": 3,
    "light__directional_light_camera_direction": 3,
}


def get_1d_dim(item):
    # log.debug(item)
    if not (item in dim1d_dict.keys()):
        raise KeyError("{} isnt in dim_dict.".format(item))
    return dim1d_dict[item]


def calc_2d_dim(inputs):
    out_dim = 0
    for item in inputs:
        out_dim += get_2d_dim(item)
    return out_dim


def calc_1d_dim(inputs):
    out_dim = 0
    # log.debug(inputs)
    for item in inputs:
        out_dim += get_1d_dim(item)
    return out_dim


def calc_dim(inputs):
    out_dim = 0
    for item in inputs:
        if item in dim1d_dict:
            out_dim += get_1d_dim(item)
        elif item in dim2d_dict:
            out_dim += get_2d_dim(item)
        else:
            raise KeyError("{} isnt in dim_dict.".format(item))
    return out_dim


def min_max_scalar(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-6)
