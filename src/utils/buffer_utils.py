import math
import imghdr
import re
from utils.str_utils import dict_to_string
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from utils.parser_utils import parse_buffer_name
from utils.utils import create_dir, get_file_component, get_tensor_mean_min_max_str
from utils.log import log
from tqdm import tqdm
import imageio
import skimage.io
albedo_min_clamp = 0.01


def hdr_to_ldr(img, use_gamma=False):
    if use_gamma:
        gamma_func = gamma
    else:
        gamma_func = lambda x:x
    ret = torch.round(gamma_func(aces_tonemapper(img)) * 255)
    return ret.type(torch.uint8)


def flow_to_motion_vector(flow) -> torch.Tensor:
    ret = flow * 1.0
    if len(ret.shape) == 3:
        C, H, W = ret.shape
        ret[0] /= W / 2
        ret[1] /= H / 2
        return ret
    elif len(ret.shape) == 4:
        B, C, H, W = ret.shape
        ret[:, 0] /= W / 2
        ret[:, 1] /= H / 2
        return ret
    else:
        raise Exception("flow must be a 3D or 4D tensor.")


def motion_vector_to_flow(mv):
    ret = mv * 1.0
    if len(ret.shape) == 3:
        C, H, W = ret.shape
        ret[0] *= W / 2
        ret[1] *= H / 2
        return ret
    elif len(ret.shape) == 4:
        B, C, H, W = ret.shape
        ret[:, 0] *= W / 2
        ret[:, 1] *= H / 2

        return ret
    return None


def get_buffer_filename(pattern, dir_path, buffer_name, index, suffix='png'):
    return pattern.format(dir_path, buffer_name, index, suffix)


def demodulate_buffer_name(buffers):
    selected_names = ['history_warped_scene_color_{}',
                      'warped_scene_color',
                      'scene_color_no_shadow',
                      'occlusion_warped_scene_color',
                      'masked_warped_scene_color',
                      'masked_occlusion_warped_scene_color']
    for i in range(len(buffers)):
        if buffers[i] in selected_names:
            buffers[i] = "de_" + buffers[i]
    return buffers


def fix_dmdl_color_zero_value(brdf_color, skybox_mask=None, sum_clamp=False):
    ret = brdf_color
    if skybox_mask is not None:
        ret = torch.ones_like(ret) * skybox_mask + ret * (1 - skybox_mask)
    return torch.clamp(ret, min=albedo_min_clamp)


def buffer_raw_to_data(data, buffer_name):
    ops = {
        'base_color': lambda x: x,  # fix_base_color_zero_value(x),
        'brdf_color': lambda x: x,  # fix_base_color_zero_value(x),
        'dmdl_color': lambda x: x,  # fix_base_color_zero_value(x),
        'depth': lambda x: x,
        'nov': lambda x: x,
        'metallic': lambda x: x,
        'specular': lambda x: x,
        'roughness': lambda x: x,
        'stencil': lambda x: x,
        'shadow_mask': lambda x: x,
        'scene_color': lambda x: x,
        'scene_color_no_shadow': lambda x: x,
        'scene_light': lambda x: x,
        'skybox_mask': lambda x: x,
        'scene_light_no_shadow': lambda x: x,
        'motion_vector': lambda x: x,
        'world_normal': lambda x: x,
        'world_position': lambda x: x,
        'world_to_clip': lambda x: x,
        "st_alpha": lambda x: x,
        "st_color": lambda x: x,
        "scene_color_no_st": lambda x: x,
        "sky_color": lambda x: x,
        "sky_depth": lambda x: x,
    }
    # res = re.search("(s\d+_)*(.+)(?:_[lr])*", buffer_name)
    # print(res)
    # if res:
    #     buffer_name = res.group(1)
    buffer_name = re.sub(r"((^(s|d|a|u)[\d]+_)|aa_)|(_[lr]$)", "", buffer_name)
    if buffer_name in ops.keys():
        ret = ops[buffer_name](data)
        if buffer_name == "motion_vector":
            ret[1, ...] *= -1
        if buffer_name == "depth":
            ret = torch.clamp(ret, max=65536.0)
            ret /= 65536.0
        return ret
    else:
        raise KeyError("{} is not in ops map".format(buffer_name))


def buffer_data_to_raw(data, buffer_name):
    ops = {
        'base_color': lambda x: x,
        'depth': lambda x: x,
        'nov': lambda x: x,
        'metallic': lambda x: x,
        'roughness': lambda x: x,
        'stencil': lambda x: x,
        'scene_light': lambda x: x,
        'motion_vector': lambda x: x,
        'skybox_mask': lambda x: x,
        'world_normal': lambda x: x,
        'world_position': lambda x: x
    }
    if buffer_name in ops.keys():
        ret = ops[buffer_name](data)
        # if buffer_name == "motion_vector":
        #     ret[1, ...] *= -1
        return ret
    else:
        raise KeyError("{} is not in ops map".format(buffer_name))


def aces_tonemapper(x, inv_gamma=False):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    if inv_gamma:
        x = x ** (1 / 2.2)
    return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def gamma(image):
    return image ** (1 / 2.2)


def inv_gamma(image):
    return image ** (2.2)


def gamma_log(image, mu=8.0):
    if isinstance(image, torch.Tensor):
        if mu == 0.0:
            ret = torch.log(image + 1.0)
        else:
            ret = torch.log(1 + image * mu) / math.log(1 + mu)
        return ret
    else:
        raise NotImplementedError(
            f'op \"gamma_log\": data type only support \"torch.Tensor\", but type of image is {type(image)}')


def inv_gamma_log(image, mu=8.0):
    if isinstance(image, torch.Tensor):
        if mu == 0.0:
            ret = torch.exp(image) - 1.0
        else:
            ret = (torch.exp(image * math.log(1 + mu)) - 1) / mu
        return ret
    else:
        raise NotImplementedError(
            "op \"inv_gamma_log\": data type only support \"torch.Tensor\".")


def buffer_data_to_vis(data, buffer_name, scale=1.0) -> torch.Tensor:
    ops = {
        'base_color': lambda x: x,
        'depth': lambda x: x,
        'abs': lambda x: torch.abs(x),
        'nov': lambda x: x * 0.5 + 0.5,
        'metallic': lambda x: x,
        'roughness': lambda x: x,
        'stencil': lambda x: x,
        'world_normal': lambda x: x,
        'scene_light': lambda x: aces_tonemapper(x),
        'scene_color': lambda x: aces_tonemapper(x),
        'normal': lambda x: torch.clamp(16 * x, -0.5, 0.5) + 0.5,
        'normal_scale': lambda x: torch.clamp(scale * x, -0.5, 0.5) + 0.5,
        'normal_8': lambda x: torch.clamp(8 * x, -0.5, 0.5) + 0.5,
        'normal_64': lambda x: torch.clamp(64 * x, -0.5, 0.5) + 0.5,
        'motion_vector': lambda x: torch.clamp(16 * x, -0.5, 0.5) + 0.5,
        'motion_vector_8': lambda x: torch.clamp(8 * x, -0.5, 0.5) + 0.5,
        'motion_vector_64': lambda x: torch.clamp(64 * x, -0.5, 0.5) + 0.5,
        'world_position': lambda x: torch.clamp(4096 * x, -4096, 4096) / 4096 * 0.5 * 20 + 0.5
    }
    if buffer_name in ops.keys():
        buffer_name = parse_buffer_name(buffer_name)['buffer_name']
        ret = ops[buffer_name](data)
        return ret
    else:
        raise KeyError("{} is not in ops map".format(buffer_name))


def create_flip_data(data, vertical=True, horizontal=True, use_batch=True, batch_mask=None):
    if not use_batch:
        assert batch_mask is None
    if vertical or horizontal:
        if use_batch:
            target_pos = 1
        else:
            target_pos = 0
        for k in data.keys():
            if isinstance(data[k], torch.Tensor) and len(data[k].shape) == 3 + target_pos:
                flip_axis = []
                if vertical:
                    flip_axis.append(target_pos + 1)
                if horizontal:
                    flip_axis.append(target_pos + 2)
                if batch_mask is not None:
                    data[k][batch_mask,...] = torch.flip(data[k][batch_mask, ...], flip_axis)
                else:
                    data[k] = torch.flip(data[k], flip_axis)
                    
                if 'motion_vector' in k:
                    # log.debug(k)
                    # log.debug(dict_to_string(data))
                    if vertical:
                        if use_batch:
                            if batch_mask is not None:
                                data[k][batch_mask, 1, ...] *= -1
                            else:
                                data[k][:, 1, ...] *= -1
                        else:
                            data[k][1, ...] *= -1
                    if horizontal:
                        if use_batch:
                            if batch_mask is not None:
                                data[k][batch_mask, 0, ...] *= -1
                            else:
                                data[k][:, 0, ...] *= -1
                        else:
                            data[k][0, ...] *= -1
        return data
    return data


def show(img):
    plt.imshow(img)
    plt.show()


def read_buffer(path, channel=None):
    '''
    C,H,W
    '''
    if path.endswith("EXR") or path.endswith("exr"):
        image = cv2.imread(path, flags=cv2.IMREAD_UNCHANGED)
        if image.shape[2] == 4:
            image = image[:, :, [2, 1, 0, 3]]
        elif image.shape[2] == 3:
            image = image[:, :, ::-1]
        image = np.array(image)
    else:
        image = imageio.imread(path).astype(float) / 255.0

    if channel is not None:
        image = image[:, :, channel]
    image = to_torch(image).type(torch.float32)
    return image


def align_channel_buffer(data, channel_num=3, mode='zero', value=0):
    n = len(data.shape)
    if n == 3:
        c, h, w = data.shape
        if c < channel_num:
            if mode == 'repeat':
                if c != 1:
                    raise NotImplementedError(
                        'mode "repeat" only support c=1, now c={}'.format(c))
                data = data.repeat(channel_num, 1, 1)
            elif mode == 'zero':
                data = torch.cat(
                    [data, torch.zeros(channel_num - c, h, w).to(data.device)])
            elif mode == 'one':
                data = torch.cat(
                    [data, torch.ones(channel_num - c, h, w).to(data.device)])
            elif mode == 'value':
                data = torch.cat(
                    [data, value * torch.ones(channel_num - c, h, w).to(data.device)])
    elif n == 4:
        b, c, h, w = data.shape
        if c < channel_num:
            if mode == 'repeat':
                if c != 1:
                    raise NotImplementedError(
                        'mode "repeat" only support c=1, now c={}'.format(c))
                data = data.repeat(1, channel_num, 1, 1)
            elif mode == 'zero':
                data = torch.cat(
                    [data, torch.zeros(b, channel_num - c, h, w).to(data.device)], dim=1)
            elif mode == 'one':
                data = torch.cat(
                    [data, torch.ones(b, channel_num - c, h, w).to(data.device)])
            elif mode == 'value':
                data = torch.cat(
                    [data, value * torch.ones(b, channel_num - c, h, w).to(data.device)])
    return data


def write_buffer(path, image, channel_num=3, mkdir=False, is_numpy=False, hdr=False, convert_to_uint8=True):
    if mkdir:
        res = get_file_component(path)
        create_dir(res['path'])
    c, h, w = image.shape
    if c < channel_num:
        if c == 1:
            image = align_channel_buffer(
                image, channel_num=channel_num, mode="repeat")
        else:
            image = align_channel_buffer(
                image, channel_num=channel_num, mode="zero")
    elif c > channel_num:
        image = image[:channel_num - c, ...]
    if path.endswith(".png") or path.endswith(".jpg"):
        if hdr:
            image = hdr_to_ldr(image)
        if convert_to_uint8:
            image *= 255.0
    if not is_numpy:
        output = to_numpy(image)
    else:
        output = image
    if path.endswith(".png"):
        cv2.imwrite(path, output[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        imageio.imwrite(path, output)


def to_numpy(arr, detach=True, cpu=True):
    assert len(arr.shape) == 3
    data = arr.permute(1, 2, 0)
    if detach:
        data = data.detach()
    if cpu:
        data = data.cpu()
    data = data.numpy()
    return data


def to_ldr_numpy(arr, normalize=255.0):
    data = arr.permute(1, 2, 0)
    img = (data.detach().cpu().numpy() * normalize).astype(np.uint8).numpy()
    return img


def save_to_img(arr, path):
    skimage.io.imsave(path, to_ldr_numpy(arr))


def to_torch(np_img):
    data = torch.from_numpy(np_img)
    data = data.permute(2, 0, 1)
    return data


def torch_d2_to_d3(data):
    H = data.shape[1]
    W = data.shape[2]
    return torch.cat((data, torch.zeros(1, H, W, dtype=torch.float32).to(data.device)), 0)


def d3_to_d2(data):
    return data[:-1, :, :]


def export_video_in_path(path, image_files, output_path, fps, tonemap=False):
    log.debug("{} ... {}".format(str(image_files[:3]), str(image_files[-3:])))
    video = cv2.VideoWriter()
    image_0 = read_buffer(path + "/" + image_files[0])
    C, H, W = image_0.shape
    video.open(output_path, cv2.VideoWriter_fourcc(
        'm', 'p', '4', 'v'), fps, (W, H), True)
    # log.debug(image_files)
    for f in tqdm(image_files):
        # tmp_image = read_buffer(path + "/" + f)
        tmp_image = read_buffer(path + "/" + f)
        if tonemap:
            tmp_image = aces_tonemapper(tmp_image)
        tmp_image = to_numpy(align_channel_buffer(tmp_image, channel_num=3))
        if f.lower().endswith(".exr"):
            tmp_image = (
                gamma(tmp_image[:, :, [2, 1, 0]]) * 255.0).astype(np.uint8)
        video.write(tmp_image)
        # video.write(cv2.imread(os.path.join(path, f)))
