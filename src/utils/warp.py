import torch
from utils.buffer_utils import flow_to_motion_vector
from utils.str_utils import dict_to_string
from utils.log import log
# from mmcv.ops.point_sample import bilinear_grid_sample

check = True


def get_merged_motion_vector_from_last(last_mv, merged_mv, residual, with_batch=True):
    '''
    last_mv: last (i+1)^{th} mv
    merged_mv: last i^{th} mv using to warp last i^{th} frame to current frame
    '''
    if not with_batch and check:
        if len(last_mv.shape) == 3:
            last_mv = last_mv.unsqueeze(0)
        if merged_mv is not None and len(merged_mv.shape) == 3:
            merged_mv = merged_mv.unsqueeze(0)
        if residual is not None and len(residual.shape) == 3:
            residual = residual.unsqueeze(0)
    # log.debug(dict_to_string(warp(last_mv, merged_mv, mode="nearest", padding_mode="zeros"), "warped last_mv" ,mmm=True))
    # log.debug(dict_to_string(merged_mv, "merged_mv input" ,mmm=True))
    if merged_mv is None:
        assert residual is not None
        return last_mv + residual
    if residual is None:
        residual = merged_mv
    ret = warp(last_mv, merged_mv, mode="nearest", padding_mode="zeros") + residual
    # log.debug(dict_to_string(ret, "merged mv ouptut", mmm=True))
    return ret


flow_base_storage = {}


def warp(img, flow, flow_type="mv", mode="nearest", padding_mode="border"):
    '''
    flow_type (str): input flow type
        'mv' | 'flow'. Default: 'mv'
    mode (str): sample mode for warp
        'nearest' | 'bilinear'. Default: 'nearest'
    padding_mode (str): padding mode for outside grid values
        'zeros' | 'border' | 'reflection'. Default: 'zeros'
    '''
    if check:
        if img.device != flow.device:
            print("warp function deal with two different device.")
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(flow.shape) == 3:
            flow = flow.unsqueeze(0)
    device = img.device
    # print(img.shape, flow.shape)
    if img.shape[2] != flow.shape[2] or img.shape[3] != flow.shape[3]:
        log.debug(img.shape)
        log.debug(flow.shape)
        assert img.shape[2] == flow.shape[2]
        assert img.shape[3] == flow.shape[3]
    k = (str(flow.device), str(flow.size()))
    # k = (str(flow.device), str(flow.size()), str(flow.type))
    if k not in flow_base_storage.keys():
        hori = torch.linspace(-1.0, 1.0, flow.shape[3]).view(
            1, 1, 1, flow.shape[3]).expand(flow.shape[0], -1, flow.shape[2], -1)
        verti = torch.linspace(-1.0, 1.0, flow.shape[2]).view(
            1, 1, flow.shape[2], 1).expand(flow.shape[0], -1, -1, flow.shape[3])
        g = torch.cat([hori, verti], 1)
        g = g.to(device)
        flow_base_storage[k] = g
    g = flow_base_storage[k]
    g = g.type(flow.dtype)
    # log.debug("g.dtype: {}, flow.dtype:{}".format(g.dtype, flow.dtype))
    if flow_type == "flow":
        flow = flow_to_motion_vector(flow)
    flow = g - flow
    flow = flow.permute(0, 2, 3, 1)
    # log.debug(dict_to_string(img, "img", mmm=True))
    # log.debug(dict_to_string(flow, "flow", mmm=True))
    # onnx = False
    # if onnx:
    #     return bilinear_grid_sample(img, flow, align_corners=True)
    # else:
    return torch.nn.functional.grid_sample(input=img, grid=flow,
                                           mode=mode, padding_mode=padding_mode,
                                           align_corners=True)
