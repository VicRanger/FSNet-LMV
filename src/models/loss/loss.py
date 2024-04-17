import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch
from utils.utils import del_dict_item
from utils.loss_utils import lpips, lpips_hdr, psnr, psnr_hdr, ssim_hdr, ssim
from utils.buffer_utils import aces_tonemapper, align_channel_buffer, buffer_data_to_vis, gamma_log, inv_gamma, inv_gamma_log, write_buffer
from utils.log import get_local_rank, log
from utils.str_utils import dict_to_string
from utils.utils import create_dir, get_tensor_mean_min_max_str
from .lap_loss import LapLoss


vgg_kernel = None
sobel = None
def get_sobel():
    global sobel 
    if sobel is None:
        sobel = SOBEL()
    return sobel

# def get_mask_kth(loss_l1, ratio=0, num=20):
#     B, C, H, W = loss_l1.shape
#     device = loss_l1.device
#     loss_l1_flat = loss_l1.resize(B, C, H*W).to(device)
#     if ratio > 0 and ratio <= 1:
#         num = max(int(ratio*H*W), 1)
#     val, _indices = torch.kthvalue(-loss_l1_flat, num)
#     val = val.view(B, C, 1, 1).expand(-1, -1, H, W).to(device)
#     one_mask = torch.ones(*loss_l1.shape).to(device)
#     zero_mask = torch.zeros(*loss_l1.shape).to(device)
#     mask_kth = torch.where(
#         loss_l1 > -val, one_mask, zero_mask).to(device)
#     return mask_kth


'''
data[0]: input
data[1]: target
if reduce=True, then return the mean of loss tensor
b_c_e(a=data[0], b=data[1]): -b*torch.log(a) - (1-b)*torch.log(1-a)
'''


def zero_l1_loss(data: list, config=None, **kwargs):
    return torch.abs(data[0])


def binary_cross_entropy_loss(data: list, config=None, reduction='none', **kwargs):
    return F.binary_cross_entropy(data[0], data[1], reduction=reduction)

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        features:nn.Sequential = nn.Sequential(torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).features)[0]
        blocks.append(features[:4].eval()) # relu 1_2
        blocks.append(features[4:9].eval()) # relu 2_2
        blocks.append(features[9:16].eval()) # relu 3_3
        blocks.append(features[16:23].eval()) # relu 4_3
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target, layers=[1, 3]):
        ''' layers: 0-1: feature, 2-3: style.  '''
        if pred.shape[1] != 3:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            if i in layers:
                x = block(x)
                y = block(y)
                loss += torch.nn.functional.l1_loss(x, y)
            # if i in style_layers:
            #     act_x = x.reshape(x.shape[0], x.shape[1], -1)
            #     act_y = y.reshape(y.shape[0], y.shape[1], -1)
            #     gram_x = act_x @ act_x.permute(0, 2, 1)
            #     gram_y = act_y @ act_y.permute(0, 2, 1)
            #     loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss




def vgg_loss(data: list, config=None, **kwargs):
    global vgg_kernel
    if vgg_kernel is None:
        vgg_kernel = VGGPerceptualLoss()
    if data[0].device != next(vgg_kernel.parameters()).device:
        vgg_kernel = vgg_kernel.to(data[0].device)
    layers = config.get('layers', False) if config is not None else [1,3]
    return vgg_kernel(data[0], data[1], layers=layers)


class Charbonnier_L1(nn.Module):
    def __init__(self):
        super(Charbonnier_L1, self).__init__()

    def forward(self, diff, mask=None):
        if mask is None:
            loss = ((diff ** 2 + 1e-6) ** 0.5).mean()
        else:
            loss = (((diff ** 2 + 1e-6) ** 0.5) * mask).mean() / (mask.mean() + 1e-9)
        return loss


def charbonnier_loss(data: list, config=None, **kwargs):
    return ((l1_loss(data) ** 2 + 1e-6) ** 0.5).mean()


def l1_loss(data: list, config=None, **kwargs):
    return F.l1_loss(data[0], data[1], reduction='none')


def rel_l1_loss(data: list, config=None, **kwargs):
    return F.l1_loss(data[0], data[1], reduction='none') / (torch.mean(data[1], dim=1, keepdim=True) + 1e-1)


def shadow_attention_mask(data: list, config=None, **kwargs):
    return (torch.abs(data[0] - data[1])) / (torch.mean(torch.min(data[0], data[1]), dim=1, keepdim=True) + 1e-2)


def rel_l1_loss_before(data: list, config=None, **kwargs):
    pass
#     return F.l1_loss(data[0], data[1], reduction='none') / (F.l1_loss(data[1].mean(), data[1], reduction='none') + 1e-1)

    # torch.abs(scene_color - warped_scene_color) /\
    #         torch.abs(scene_color.mean() - scene_color)


def rel_l2_loss(data: list, config=None, **kwargs):
    return F.mse_loss(data[0], data[1], reduction='none') / (F.mse_loss(data[1].mean(), data[1], reduction='none') + 1e-1)


def l2_loss(data, config=None, **kwargs):
    for i in range(len(data[0].shape)):
        if data[0].shape[i] != data[1].shape[i]:
            raise Exception("data.0 and data.1 dont have same shape: {} {}".format(
                data[0].shape, data[1].shape))
    return F.mse_loss(data[0], data[1], reduce=False)


class SOBEL(nn.Module):
    def __init__(self):
        super(SOBEL, self).__init__()
        self.kernelX = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).float()
        self.kernelY = self.kernelX.clone().T
        self.kernelX = self.kernelX.unsqueeze(0).unsqueeze(0)
        self.kernelY = self.kernelY.unsqueeze(0).unsqueeze(0)

    def to(self, device):
        self.kernelX = self.kernelX.to(device)
        self.kernelY = self.kernelY.to(device)
        return self

    def forward_x(self, data):
        if self.kernelX.device != data:
            self.kernelX = self.kernelX.to(data)
        ret = F.conv2d(data, self.kernelX, padding=1)
        return ret

    def forward_y(self, data):
        if self.kernelY.device != data:
            self.kernelY = self.kernelY.to(data)
        ret = F.conv2d(data, self.kernelY, padding=1)
        # log.debug(dict_to_string({
        #     'ret': ret,
        # }))
        return ret

    def forward(self, pred, gt):
        N, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        img_stack = torch.cat(
            [pred.reshape(N * C, 1, H, W), gt.reshape(N * C, 1, H, W)], 0)
        if self.kernelX.device != gt.device:
            self.kernelX = self.kernelX.to(gt.device)
            self.kernelY = self.kernelY.to(gt.device)
        sobel_stack_x = F.conv2d(img_stack, self.kernelX, padding=1)
        sobel_stack_y = F.conv2d(img_stack, self.kernelY, padding=1)
        pred_X, gt_X = sobel_stack_x[:N * C], sobel_stack_x[N * C:]
        pred_Y, gt_Y = sobel_stack_y[:N * C], sobel_stack_y[N * C:]

        L1X, L1Y = torch.abs(pred_X - gt_X), torch.abs(pred_Y - gt_Y)
        loss = (L1X + L1Y)
        return loss



def l1_mask_loss(data, config=None, **kwargs):
    return F.l1_loss(data[0] * data[2], data[1] * data[2], reduce=False)


def l2_mask_loss(data, config=None, **kwargs):
    return F.mse_loss(data[0] * data[2], data[1] * data[2], reduce=False)


def l2_minus_mask_loss(data, config=None, **kwargs):
    return F.mse_loss(data[0] * (1 - data[2]), data[1] * (1 - data[2]), reduce=False)

# pred, gt, mask


def shadow_mask_loss(data, config=None, **kwargs):
    loss_l1 = l1_loss(data[0], data[1])
    mask = 1 - data[2]
    return loss_l1 * mask

# pred, gt, mask


def extranet_hole_loss(data, config=None, output=None, **kwargs):
    debug = config.get('debug', False) if config is not None else False
    mask = data[2]

    ret = l1_loss([data[0] * mask, data[1] * mask])
    if debug:
        assert output is not None
        output['hole_data_0'] = data[0] * mask
        output['hole_data_1'] = data[1] * mask
        output['hole_diff'] = torch.abs(data[0] * mask - data[1] * mask)
        output['hole_l1'] = F.l1_loss(data[0], data[1], reduction='none')
    return ret


def extranet_shadow_loss(data, config=None, output=None, **kwargs):
    ratio = config.get('ratio', 0.1) if config is not None else 0.1
    debug = config.get('debug', False) if config is not None else False
    # log.debug(dict_to_string(config))
    # log.debug(dict_to_string(output))
    # log.debug(dict_to_string(data))
    # log.debug("{} {}".format("extra_shadow_loss", debug))
    B, C, H, W = data[0].shape
    if len(data) == 3:
        tmp_data0 = data[0] * data[2]
        tmp_data1 = data[1] * data[2]
    elif len(data) == 2:
        tmp_data0 = data[0]
        tmp_data1 = data[1]
    else:
        raise
    val, ind = torch.topk(l1_loss([tmp_data0, tmp_data1]).view(B, C, -1), k=int(H * W * ratio))
    val *= ratio
    if debug:
        assert output is not None
        output['shadow_val'] = val
        output['shadow_ind'] = ind
        output['shadow_diff'] = torch.abs(tmp_data0 - tmp_data1)

    return val


def ssim_hdr_value(data, config=None, **kwargs):
    return ssim_hdr(data[0], data[1])


def ssim_value(data, config=None, **kwargs):
    return ssim(data[0], data[1])


def ssim_hdr_loss(data, config=None, **kwargs):
    return 1 - ssim_hdr_value(data, config=config)


def psnr_hdr_value(data, config=None, **kwargs):
    return psnr_hdr(data[0], data[1])


def psnr_value(data, config=None, **kwargs):
    return psnr(data[0], data[1])


def psnr_hdr_loss(data, config=None, **kwargs):
    return 1 - psnr_hdr_value(data, config=config)


def lpips_hdr_value(data, config=None, **kwargs):
    return lpips_hdr(data[0], data[1])


def lpips_value(data, config=None, **kwargs):
    return lpips(data[0], data[1])


def lpips_hdr_loss(data, config=None, **kwargs):
    return 1 - lpips_hdr_value(data, config=config)


def contrastive_loss(feature1, feature2, output1, output2, num_pair=10):
    f_shape = feature1.shape
    o_shape = output1.shape
    pair_index1 = (torch.rand(num_pair) *
                   f_shape[-2] * f_shape[-1]).type(torch.int).tolist()
    pair_index2 = (torch.rand(num_pair) *
                   o_shape[-2] * o_shape[-1]).type(torch.int).tolist()
    feature1 = feature1.reshape(f_shape[0], f_shape[1], -1)[..., pair_index1]
    feature2 = feature2.reshape(f_shape[0], f_shape[1], -1)[..., pair_index2]
    output1 = output1.reshape(o_shape[0], o_shape[1], -1)[..., pair_index1]
    output2 = output2.reshape(o_shape[0], o_shape[1], -1)[..., pair_index2]
    log.debug("{} {} {} {}".format(feature1.shape,
              feature2.shape, output1.shape, output2.shape))
    diff_feature = torch.sum((feature1 - feature2) ** 2, dim=1, keepdim=True)
    diff_output = torch.sum((output1 - output2) ** 2, dim=1, keepdim=True)
    return l2_loss(diff_feature, diff_output)


lap_loss_ins = LapLoss()


def laplace_loss(data, config=None, **kwargs):
    return lap_loss_ins(data[0], data[1])


class LossFunction:
    single_ops = {
        "l1": l1_loss,
        "zero_l1": zero_l1_loss,
        "charbonnier_l1": charbonnier_loss,
        "lap": laplace_loss,
        "l1_rel": rel_l1_loss,
        "l2_rel": rel_l2_loss,
        "l2": l2_loss,
        "l1_mask": l1_mask_loss,
        "l2_mask": l2_mask_loss,
        "l2_minus_mask": l2_minus_mask_loss,
        "shadow_mask": shadow_mask_loss,
        "extranet_hole": extranet_hole_loss,
        "extranet_shadow": extranet_shadow_loss,
        "psnr": psnr_value,
        "psnr_hdr": psnr_hdr_value,
        "ssim": ssim_value,
        "ssim_hdr": ssim_hdr_value,
        "lpips": lpips_value,
        "lpips_hdr": lpips_hdr_value,
        "binary_cross_entropy_loss": binary_cross_entropy_loss,
        "vgg": vgg_loss,
    }
    paired_ops = {
        "contrastive": contrastive_loss
    }

    def __init__(self, config):
        self.config = config
        self.loss_func = []
        self.debug_loss_func = []
        self.involved_loss_mode = set()
        # self.sum_ratio = 0
        for loss_name in self.config["train_loss"].keys():
            mode = self.config["train_loss"][loss_name]["mode"]
            self.loss_func.append({
                "name": loss_name,
                "args": self.config["train_loss"][loss_name]['args'],
                "mode": mode,
                # "ratio": self.config["train_loss"][loss_name]["ratio"],
                "scale": self.config["train_loss"][loss_name].get("scale", 1.0),
                "config": self.config["train_loss"][loss_name].get("config", {}),
                "is_paired": self.is_paired(mode),
                'enable': self.config["train_loss"][loss_name].get("enable", True)
            })
            # self.sum_ratio += self.loss_func[-1]["ratio"]
            self.involved_loss_mode.add(
                self.config["train_loss"][loss_name]["mode"])
        # log.debug(self.loss_func)
        self.has_paired_loss = self.has_paired()
        for loss_name in self.config["debug_loss"].keys():
            mode = self.config["debug_loss"][loss_name]["mode"]
            self.debug_loss_func.append({
                "name": loss_name,
                "args": self.config['debug_loss'][loss_name]['args'],
                "config": self.config["debug_loss"][loss_name].get("config", {}),
                "mode": mode,
                'enable': True
            })
        log.debug("loss func details: {}".format(self.loss_func))
        log.debug("debug loss func details: {}".format(self.debug_loss_func))
        log.info("[LossFunction]: created. info: {}".format(self.__str__()))

    def __str__(self):
        return "loss: {}, has_paired_loss: {}, debug_loss: {}".format(
            [item['name'] for item in self.loss_func],
            self.has_paired_loss,
            [item['name'] for item in self.debug_loss_func])

    def is_paired(self, name):
        if name == "contrastive":
            return True
        return False

    def has_paired(self):
        for mode in self.involved_loss_mode:
            if self.is_paired(mode):
                return True
        return False

    def check_data(self, data, non_local=None):
        for item in (self.loss_func + self.debug_loss_func):
            item['enable'] = True
            if "is_paired" in item.keys() and item["is_paired"]:
                pass
            else:
                input_names = [item['args'][i]
                               for i in range(len(item['args']))]
                for name in input_names:
                    if name not in data.keys():
                        log.warn('[Loss] {} in "{}" is not in data ({})'.format(
                            name, item['name'], list(data.keys())))
                        item['enable'] = False

    def get_active_loss_func_names(self) -> list:
        names = []
        for item in (self.loss_func + self.debug_loss_func):
            if item.get('enable', True):
                names.append(item['name'])
        return names

    def forward(self, data, non_local=None):
        loss = {}
        total_loss = 0.0
        for item in self.loss_func:
            if not (item.get('enable', True)):
                continue
            if item["is_paired"]:
                if non_local is None:
                    continue
                loss[item['name']] = self.forward_paired(item['mode'],
                                                         data[item['args'][0]],
                                                         data[item['args'][1]],
                                                         non_local[item['args'][0]],
                                                         non_local[item['args'][1]])
            else:
                loss[item["name"]] = self.forward_single(
                    item, data, output=data) * item['scale']
            # data[item['name']] = loss[item['name']]
            # log.debug(dict_to_string(data))

        keys_list = list(data.keys())
        for name in keys_list:
            if name.endswith("_loss"):
                loss[name] = data[name]

        raw_loss = {}
        for item in loss:
            if len(loss[item].shape) > 0:
                raw_loss[item + '_ls'] = loss[item].clone()
                loss[item] = loss[item].mean()
            total_loss += loss[item]
            # loss[item] = loss[item].item()
        loss.update(raw_loss)

        for name in keys_list:
            if name.endswith("_loss"):
                data = del_dict_item(data, name)

        loss['loss'] = total_loss
        # log.debug(dict_to_string(loss))
        return loss

    def forward_debug(self, data):
        if get_local_rank() != 0:
            return {}
        loss = {}
        for item in self.debug_loss_func:
            if not (item.get('enable', True)):
                continue
            with torch.no_grad():
                loss[item["name"]] = self.forward_single(
                    item, data).mean()
        return loss

    def forward_single(self, item, data, output=None):
        mode = item['mode']
        data_input = [data[item['args'][i]] for i in range(len(item['args']))]
        config = item['config']

        if config.get("skybox_mask", False):
            # log.debug("+++++ ENABLE SKYBOX_MASK: {:>10}, {:>16} +++++".format(item['mode'], item['name']))
            for i in range(len(data_input)):
                data_input[i] = data_input[i] * data['skybox_mask']
        elif config.get("skybox_mask_out", False):
            # log.debug("+++++ ENABLE SKYBOX_MASK_OUT: {:>10}, {:>16} +++++".format(item['mode'], item['name']))
            for i in range(len(data_input)):
                data_input[i] = data_input[i] * (1 - data['skybox_mask'])

        if (tonemap_name := config.get("tonemap", None)) == "gamma_log":
            for i in range(len(data_input)):
                data_input[i] = gamma_log(data_input[i])
        elif tonemap_name == "aces":
            for i in range(len(data_input)):
                data_input[i] = aces_tonemapper(data_input[i])
        elif tonemap_name is not None:
            raise Exception(
                f'unsupported tonemap func given in LossFunction.forward_single("{item["name"]}"): "{tonemap_name}"')

        return self.single_ops[mode](data_input, config=config, output=output)

    def forward_paired(self, mode, local_f, local_o, non_local_f, non_local_o):
        if (non_local_f is None or non_local_o is None):
            raise ValueError("lcoal and non_local must not be None, with\n local={},\n non_local={}".format(
                dict_to_string(local_f, "local_f") + "\n" +
                dict_to_string(local_o, " local_o"),
                dict_to_string(non_local_f, "non_local_f") + "\n" + dict_to_string(non_local_o, "non_local_o")))
        ret = 0
        cnt = 0
        # ret += self.paired_ops[mode](local_f, non_local_f, local_o, non_local_o)
        # cnt += 1
        # log.debug(get_tensor_mean_min_max_str(local_f))
        # log.debug(get_tensor_mean_min_max_str(non_local_f))
        ret += self.paired_ops[mode](local_f,
                                     non_local_f, local_o, non_local_o)
        cnt += 1
        return ret / cnt
