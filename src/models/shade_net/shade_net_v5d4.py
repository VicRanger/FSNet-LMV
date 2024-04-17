''' shade_net v5d4 = shade_net v5d3 + simplify logic'''
from __future__ import annotations
from re import I
from dataloaders.patch_loader import history_extend
from utils.utils import add_metaname
from models.general.unet import single_conv
from dataloaders.dataset_base import DatasetBase
from datasets.shade_net_dataset import ShadeNetV5Dataset, ShadeNetV5SeqDataset
import copy
import torch.nn.functional as F
import torch
import torch.nn as nn
from models.general.unet import UNetDecoder, UNetEncoder
from .conv_lstm_v5 import ConvLSTMCell
from models.general.common_structure import NetBase, create_act_func, create_norm_func
from .loss.flow_loss import flow_loss, flow2_loss
from dataloaders.raw_data_importer import tensor_as_type_str
from utils.dataset_utils import create_de_color
from utils.buffer_utils import aces_tonemapper, fix_dmdl_color_zero_value, flow_to_motion_vector
from utils.warp import get_merged_motion_vector_from_last, warp
from utils.buffer_utils import gamma_log, inv_gamma_log, write_buffer
from models.model_base import ModelBase, ModelBaseEXT
from utils.str_utils import dict_to_string
from utils.model_utils import get_1d_dim, get_2d_dim, model_to_half
from utils.log import log
from models.loss.loss import charbonnier_loss, l1_loss, l1_mask_loss, shadow_attention_mask, rel_l1_loss, rel_l1_loss_before
mu = 8.0
def tonemap_func(data):
    return gamma_log(data, mu=mu)

def inv_tonemap_func(data):
    return inv_gamma_log(data, mu=mu)



def data_to_input(data, config, cat_axis=1):
    data = DatasetBase.preprocess(data, config={'mu': mu})
    ret = {}
    cats = []
    for name in config['shade_encoder']['input_buffer']:
        cats.append(data[name])
    ret['shade_encoder_input'] = torch.cat(cats, dim=cat_axis)
    ret['history_encoder_input'] = []
    for item in config['history_encoders']['inputs']:
        cats = []
        for name in item:
            if config['tonemap_in_his_encoder']:
                if 'st_color' in name or 'sky_color' in name:
                    tmp_data =  aces_tonemapper(inv_tonemap_func(data[name]))
                else:
                    tmp_data = data[name].clone()
            else:
                tmp_data = data[name].clone()
            cats.append(tmp_data)
        ret['history_encoder_input'].append(torch.cat(cats, dim=cat_axis))
        ret['history_encoder_input_cats'] = cats
    ret.update(data)
    return ret


class ShadeNetEncoder(UNetEncoder):
    name = "ShadeNetEncoder"
    cnt_instance = 0

    def __init__(self, config, *args, **kwargs):
        self.instance_names = []
        self.instance_names.append("{}{}".format(
            ShadeNetEncoder.name, ShadeNetEncoder.cnt_instance))
        ShadeNetEncoder.cnt_instance += 1
        super().__init__(config=config)


class ResBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, side_channels, bias=True, act_func_name=None, norm_func_name=None):
        super(ResBlock, self).__init__()
        if norm_func_name == 'batch_norm_2d':
            bias = False
        self.side_channels = side_channels
        self.conv1 = single_conv(in_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=bias,
                                 norm_func_name=norm_func_name, act_func_name=act_func_name, single_act_channel=False)
        self.conv5 = nn.Conv2d(mid_channel, mid_channel,
                               kernel_size=3, stride=1, padding=1, bias=bias)
        self.single_norm_act = nn.Sequential()
        if norm_func_name:
            self.single_norm_act.add_module("{}_1".format(norm_func_name),
                                            create_norm_func(norm_func_name)(mid_channel))
        if act_func_name:
            self.single_norm_act.add_module("{}_1".format(act_func_name),
                                            create_act_func(act_func_name, out_channel=mid_channel)())

    def forward(self, x):
        out = self.conv1(x)
        out = self.single_norm_act(x + self.conv5(out))
        return out


class DecoderBlock(nn.Module):
    def __init__(self, last_c, cat_c, mid_c, out_c, act_func_name=None, norm_func_name=None):
        super().__init__()
        self.single_conv = single_conv(
            last_c + cat_c, mid_c, act_func_name=act_func_name, norm_func_name=norm_func_name, single_act_channel=False)
        self.res_block = ResBlock(
            mid_c, mid_c, out_c, act_func_name=act_func_name, norm_func_name=norm_func_name)
        self.deconv = nn.ConvTranspose2d(
            mid_c, out_c, kernel_size=4, stride=2, padding=1, bias=True)

    def forward(self, input_layer, skip_conn=None, skip_add=None):
        cur_input = input_layer
        if skip_conn is not None:
            cur_input = torch.cat([cur_input, skip_conn], dim=1)
        if skip_add is not None:
            cur_input = cur_input + skip_add
        cur_input = self.single_conv(cur_input)
        cur_input = self.res_block(cur_input)
        cur_input = self.deconv(cur_input)
        return cur_input


class OutputBlock(nn.Module):
    def __init__(self, in_channel,  mid_channel, out_channel, act_func_name=None, norm_func_name=None):
        super(OutputBlock, self).__init__()
        if norm_func_name == 'batch_norm_2d':
            bias = False
        else:
            bias = True
        self.seq = nn.Sequential()
        self.conv1x1 = single_conv(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=bias,
                                   norm_func_name=norm_func_name, act_func_name=act_func_name, single_act_channel=False)
        self.output1x1 = nn.Conv2d(
            mid_channel, out_channel, kernel_size=1)

    def forward(self, data):
        ret = self.conv1x1(data)
        ret = self.output1x1(ret)
        return ret


class RecurrentEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.compress_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, data):
        ret = self.compress_conv(data)
        ret = self.tanh(ret)
        return ret


class ShadeNetDecoder(UNetDecoder):
    name = "ShadeNetDecoder"
    cnt_instance = 0

    def __init__(self, config, *args, **kwargs):
        self.instance_names = []
        self.instance_names.append("{}{}".format(
            ShadeNetDecoder.name, ShadeNetDecoder.cnt_instance))
        self.enable_output_concat = False
        ShadeNetDecoder.cnt_instance += 1
        super().__init__(config=config)

    def create_decoder_block(self, config) -> nn.Module:
        return DecoderBlock(config['in_channel'], config['concat_channel'], config['mid_channel'], config['out_channel'],
                            act_func_name=config['act_func_name'], norm_func_name=config['norm_func_name'])

    def create_output_block(self, config) -> nn.Module:
        return OutputBlock(config['in_channel'], config['mid_channel'], config['out_channel'],
                           config['act_func_name'], config['norm_func_name'])

    def forward(self, data, ret):
        raise Exception("")


def resize(x, scale_factor):
    if scale_factor != 1:
        return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)
    return x


hidden_cache = {}


def get_demodulated_color(scene_color, dmdl_color):
    return create_de_color(
        scene_color, dmdl_color, fix=True)


def set_fake(buffer):
    zero_buffer = torch.zeros_like(buffer)
    zero_buffer = zero_buffer.type(zero_buffer.dtype)
    return zero_buffer


def get_lmv(rmv: torch.Tensor, smv0: torch.Tensor | None, smv_res: torch.Tensor | None, rmv_scale=1.0, smv_scale=1.0, re_scale=1.0):
    if smv0 is not None:
        tmv0 = get_merged_motion_vector_from_last(
            get_resized_mv(rmv, scale=rmv_scale),
          get_resized_mv(smv0, scale=smv_scale, re_scale=re_scale),
           get_resized_mv(smv_res, scale=smv_scale, re_scale=re_scale))
    else:
        tmv0 = None
    assert tmv0 is not None
    return tmv0


def get_resized_mv(mv, scale=1.0, re_scale=1.0):
    if scale == 1.0:
        if re_scale > 1:
            return resize(resize(mv, 1 / re_scale), re_scale)
        else:
            return mv
    else:
        return resize(mv, scale)


class ShadeNet(NetBase):
    class_name = "ShadeNet"
    cnt_instance = 0

    def __init__(self, config):
        add_metaname(self, ShadeNet)
        super().__init__(config)
        self.layer_flow_channel = 2
        self.config = config
        self.gt_alias_name = self.config['gt_alias']
        self.method = self.config['method']
        self.precision = self.config['precision']
        self.config['shade_decoder'] = self.config[f'shade_decoder__{self.method}']
        self.enable_demodulate = "demodulate" in self.config['feature']
        self.enable_demodulate_before_warp = "demodulate_before_warp" in self.config['feature']
        self.enable_demodulate_after_warp = "demodulate_after_warp" in self.config['feature']
        assert int(self.enable_demodulate_before_warp) + int(self.enable_demodulate_after_warp) == 1
        self.enable_recurrent = "recurrent" in self.config['feature']
        self.enable_temporal_warped_feature = "temporal_warped_feature" in self.config[
            'feature']
        if self.enable_temporal_warped_feature:
            log.debug("temporal_warped_feature enabled")
        else:
            log.debug("temporal_warped_feature disabled")

        self.enable_smv0 = "smv0" in self.config['feature']
        self.enable_sep_smv_res = "sep_smv_res" in self.config['feature']
        self.enable_history_smv0 = "history_smv0" in self.config['feature']
        self.enable_history_sep_smv_res = "history_sep_smv_res" in self.config['feature']
        if self.enable_sep_smv_res:
            assert self.enable_smv0
        if self.enable_history_sep_smv_res:
            assert self.enable_history_smv0
        self.enable_output_warp_upscale = 'output_warp_upscale' in self.config['feature']

        self.enable_output_warp1 = 'output_warp1' in self.config['feature']
        self.enable_output_warp1_upscale = 'output_warp1_upscale' in self.config['feature']
        if self.enable_output_warp1_upscale:
            assert self.enable_output_warp1

        self.enable_output_warp2 = 'output_warp2' in self.config['feature']
        self.enable_output_warp2_upscale = 'output_warp2_upscale' in self.config['feature']
        if self.enable_output_warp2_upscale:
            assert self.enable_output_warp2

        if self.enable_output_warp1 or self.enable_output_warp2 or self.enable_output_warp_upscale:
            assert self.enable_smv0
        self.enable_no_st_warp1 = 'no_st_warp1' in self.config['feature']
        self.enable_no_st_warp2 = 'no_st_warp2' in self.config['feature']
        self.enable_no_st_warp_upscale = 'no_st_warp_upscale' in self.config['feature']
        assert int(self.enable_no_st_warp1) + int(self.enable_no_st_warp2) + \
            int(self.enable_no_st_warp_upscale) <= 1
        if self.enable_no_st_warp_upscale:
            self.enable_output_warp_upscale
        self.enable_st_warp1 = 'st_warp1' in self.config['feature']
        self.enable_st_warp2 = 'st_warp2' in self.config['feature']
        self.enable_st_warp_upscale = 'st_warp_upscale' in self.config['feature']
        assert not (self.enable_st_warp1 and self.enable_st_warp2)
        self.enable_sky_warp1 = 'sky_warp1' in self.config['feature']
        self.enable_sky_warp2 = 'sky_warp2' in self.config['feature']
        self.enable_sky_warp_upscale = 'sky_warp_upscale' in self.config['feature']
        if self.enable_sky_warp_upscale:
            self.enable_output_warp_upscale
        assert not (self.enable_sky_warp1 and self.enable_sky_warp2)
        self.enable_st = 'st' in self.config['feature']
        self.enable_st_residual = 'st_residual' in self.config['feature']
        self.enable_sky_residual = 'sky_residual' in self.config['feature']
        self.enable_zero_flow_loss = 'zero_flow' in self.config['loss']
        self.enable_zero_flow2_loss = 'zero_flow2' in self.config['loss']
        self.enable_rec_occ_loss = 'rec_occ' in self.config['loss']
        self.enable_c1_flow_loss = 'c1_flow' in self.config['loss']
        self.enable_debug_fake_temporal = 'fake_temporal' in self.config['debug']
        self.enable_debug_fake_spatial = 'fake_spatial' in self.config['debug']
        self.enable_debug_no_st = 'no_st' in self.config['debug']
        self.enalbe_debug_no_history_gbuffer = 'no_history_gbuffer' in self.config['debug']
        self.enable_no_sigmoid = 'no_sigmoid' in self.config['debug']
        self.enable_timing = False

        if self.enable_output_warp2:
            if self.enable_smv0:
                self.config['shade_decoder']['output_buffer'].append(
                    {"name": "pred_smv0_raw", "channel": 2})
            if self.enable_sep_smv_res:
                self.config['shade_decoder']['output_buffer'].append(
                    {"name": "pred_smv_res_raw", "channel": 2})

        if self.enable_st_residual:
            self.config['shade_decoder']['output_buffer'].append(
                {"name": "pred_st_color_residual", "channel": 3})
            self.config['shade_decoder']['output_buffer'].append(
                {"name": "pred_st_alpha_residual", "channel": 1})
        if self.enable_sky_residual:
            self.config['shade_decoder']['output_buffer'].append(
                {"name": "pred_sky_color_residual", "channel": 3})

        self.se_pf = self.config['shade_encoder']['output_prefix']
        self.sce_pf = self.config['scene_color_encoder_output_prefix']
        self.he_pfs = [item
                       for item in self.config['history_encoders']['output_prefixs']]

        self.he_mv_name = self.config['history_encoders']['mv_name']
        self.history_id = self.config['history_encoders']['history_id']

        if self.config['scene_color_encoder']['class'] == 'ShadeNetEncoder':
            self.scene_color_encoder = ShadeNetEncoder(self.config['scene_color_encoder'])
        else:
            raise Exception('Wrong class name {} for ShadeNetEncoder!'.format(self.config['scene_color_encoder']['class']))

        if self.config['shade_encoder']['class'] == 'ShadeNetEncoder':
            self.shade_encoder = ShadeNetEncoder(self.config['shade_encoder'])
        else:
            raise Exception('Wrong class name {} for ShadeNetEncoder!'.format(self.config['shade_encoder']['class']))

        ''' accumulate concat channel of history_encoders '''
        self.num_history_encoder = self.config['history_encoders']['num']
        encoders_out_channel = copy.deepcopy(self.config['shade_encoder']['encoders_out_channel'])
        for i in range(len(encoders_out_channel)):
            encoders_out_channel[i] += self.config['scene_color_encoder']['encoders_out_channel'][i] * self.num_history_encoder

        self.config['shade_decoder']['encoders_out_channel'] = encoders_out_channel
        self.config['shade_decoder']['num_he'] = self.num_history_encoder
        self.shade_decoder = eval(self.config['shade_decoder']["class"])(
            self.config['shade_decoder'], self)
        num_dec = self.num_shade_decoder_layer = len(self.shade_decoder.decoders)

        if self.enable_recurrent:
            self.recurrent_encoders = nn.ModuleList([])
            num_ru = num_dec - 1
            ''' num_dec - 1: we dont recurent encoding at last decoder'''
            for i in range(num_ru):
                in_channel = int(self.config['shade_decoder']['struct']['decoder'][i][-1])
                out_channel = int(
                    self.config['scene_color_encoder']['struct']['encoder'][-(i + 1) - 1][-1])
                # index of feature
                re = RecurrentEncoder(in_channel, out_channel)
                self.recurrent_encoders.append(re)
            self.recurrent_blocks = nn.ModuleList([])
            for i in range(num_ru):
                ''' in_channel=24, gbuffer_channel=24'''
                in_channel = int(
                    self.config['scene_color_encoder']['struct']['encoder'][i][-1])
                gbuffer_channel = int(
                    self.config['shade_encoder']['struct']['encoder'][i][-1])
                ru = ConvLSTMCell(in_channel, in_channel + gbuffer_channel, in_channel)
                self.recurrent_blocks.append(ru)
        self.get_feature_list()
        log.debug(dict_to_string(self.enabled_features, f"[{self.__class__}] enabled feature: \n", full_name=False))
        log.debug(f"[{self.__class__}] disabled feature: {self.disabled_features}")

    def get_feature_list(self):
        self.enabled_features = []
        self.disabled_features = []
        for attr in dir(self):
            if attr.startswith("enable_"):
                if getattr(self, attr):
                    self.enabled_features.append(attr.replace("enable_", ""))
                else:
                    self.disabled_features.append(attr.replace("enable_", ""))

    def generate_recurrent_encoding(self, data, ret, key_name, he_id, channels=[]):
        ''' create feature_0 '''
        sc_layers_key = f"{key_name}sc_layers"
        device = data['merged_motion_vector_0'].device
        num_dec = self.num_shade_decoder_layer
        if sc_layers_key not in hidden_cache.keys():
            tmp_shape = list(data['merged_motion_vector_0'].shape)
            tmp_shape[2] = tmp_shape[2] // 2
            tmp_shape[3] = tmp_shape[3] // 2

            hidden_cache[sc_layers_key] = []
            for i in range(num_dec - 1):
                h_channel = channels[i]
                layer = torch.zeros(
                    [tmp_shape[0], h_channel, tmp_shape[2], tmp_shape[3]], device=device)

                hidden_cache[sc_layers_key].append(layer)
                tmp_shape[2] = tmp_shape[2] // 2
                tmp_shape[3] = tmp_shape[3] // 2
            h_channel = channels[-1]

        ret[f'{key_name}sc_layers'] = [
            item.detach().clone().half() if self.precision == 'fp16' else item.detach().clone() for item in hidden_cache[sc_layers_key]]
        return

    def get_inference_output(self, ret):
        return ret

    def get_train_output(self, ret):
        return ret

    def get_debug_output(self, ret):
        return ret

    def forward(self, data):
        ret = {}
        ret['input_data'] = data
        self.enable_timing = self.enable_timing or self.config.get("export_onnx", False)

        num_he = self.num_history_encoder
        num_dec = self.num_shade_decoder_layer

        ''' 
        history encoding
        '''

        for he_id in range(num_he):
            if self.he_pfs[he_id]+'sc_layers' in data.keys():
                ret[self.he_pfs[he_id]+'sc_layers'] = data[self.he_pfs[he_id]+'sc_layers']
                ret[self.he_pfs[he_id]+'output'] = data[self.he_pfs[he_id]+'output']
                # log.debug(f"use cached {he_pfs[he_id]+'sc_layers'}")
                continue
            tmp_output = self.scene_color_encoder(
                {'input': data['history_encoder_input'][he_id]})
            ''' 
            tmp_output: he_$NUM_ + output, sc_layers 
            '''
            ''' remove first sc_layer because no skip-connection from input_block '''
            tmp_output['sc_layers'] = tmp_output['sc_layers'][1:]
            prefix_output = {}
            for k in tmp_output.keys():
                prefix_output[self.he_pfs[he_id] + k] = tmp_output[k]
            ret.update(prefix_output)

            ''' for time testing '''
            if self.enable_timing and (self.enable_temporal_warped_feature and not (self.enable_debug_fake_temporal)):
                for tmp_id in range(self.num_history_encoder):
                    prefix_output = {}
                    for k in tmp_output.keys():
                        prefix_output[self.he_pfs[tmp_id] + k] = tmp_output[k]
                    ret.update(prefix_output)
                break

        ''' 
        gbuffer encoding
        '''

        tmp_output = self.shade_encoder({'input': data['shade_encoder_input']})
        ''' se_ + output, sc_layers  '''
        ret.update(tmp_output)
        ''' remove firs t sc_layer because no skip-connection from input_block '''
        ret[self.se_pf + 'sc_layers'] = ret[self.se_pf + 'sc_layers'][1:]

        ''' 
        recurrent feature streaming 
        '''

        if self.enable_recurrent:
            num_ru = len(self.recurrent_blocks)
            for he_id in range(num_he):
                if f"hidden_{he_id}_sc_layers" not in data.keys():
                    self.generate_recurrent_encoding(data, ret, f"hidden_{he_id}_", he_id,
                                                     channels=[self.recurrent_blocks[ru_ind].out_channel
                                                               for ru_ind in range(num_ru)])
                else:
                    ret[f'hidden_{he_id}_sc_layers'] = data[f'hidden_{he_id}_sc_layers']
                    # log.debug(f"find hidden cache: hidden_{he_id}_sc_layers")
                    
                if f"history_{he_id}_{self.se_pf}sc_layers" not in data.keys():
                    self.generate_recurrent_encoding(data, ret, f"history_{he_id}_{self.se_pf}", he_id,
                                                     channels=[self.recurrent_blocks[ru_ind].out_channel
                                                               for ru_ind in range(num_ru)])
                else:
                    if self.enalbe_debug_no_history_gbuffer:
                        ret[f"history_{he_id}_{self.se_pf}sc_layers"] = ret[self.se_pf + 'sc_layers']
                    else:
                        ret[f"history_{he_id}_{self.se_pf}sc_layers"] = data[f"history_{he_id}_{self.se_pf}sc_layers"]
                    # log.debug(f"find se cache: history_{he_id}_{self.se_pf}sc_layers")
                    
            for he_id in range(num_he):
                ''' TODO: think how to deal with displacement between hidden_layers and se_sc_layers
                (maybe inverse backwarp to he_id?) '''

                ret[self.he_pfs[he_id] + 'sc_layers'] = [
                    self.recurrent_blocks[i](
                        ret[self.he_pfs[he_id] + 'sc_layers'][i],
                        [
                            ret[f'hidden_{he_id}_sc_layers'][i],
                            # ret[self.se_pf+'sc_layers'][i],
                            ret[f"history_{he_id}_{self.se_pf}sc_layers"][i],
                        ]
                    )
                    for i in range(num_ru)]

        '''
        decoding
        '''

        if self.enable_temporal_warped_feature:
            for ind in range(num_he):
                he_pf = self.he_pfs[ind]
                final_mv = resize(data[self.he_mv_name.format(
                    self.history_id[ind])], 1 / (2**num_dec))
                if self.enable_debug_fake_temporal:
                    final_mv = set_fake(final_mv)
                mode = self.config['config'].get('feature_warp_mode', 'bilinear')
                ''' warp the smallest encoding only by merged motion vector '''
                ret[he_pf + "output"] = warp(ret[he_pf + "output"], final_mv,
                                             padding_mode="border", mode=mode, flow_type="mv")

        ''' concat output of decoders which is fed into the first decoder '''
        cat_arr = []
        for i in range(num_he):
            cat_arr.append(ret[self.he_pfs[i] + "output"])
        cat_arr.append(ret[self.se_pf + "output"])
        tmp_tensor = torch.cat(cat_arr, dim=1)

        if self.enable_recurrent:
            ret['sd_sc_layers'] = []

        ''' start decoding loop '''
        for i in range(num_dec + 1):
            if i == num_dec:
                layer_id = 0
            else:
                layer_id = num_dec - i
            ''' TODO: sth to do with custom skip conn '''
            if i > 0:
                ratio = 2 ** (layer_id)
                ''' skip-connection will perform when i>0 and i < num_dec '''
                skip_conn = i < num_dec
                ''' warp_last_dec: calculate lmv at output_block '''
                warp_last_dec = i == num_dec and self.enable_output_warp1
                skip_conn_arr = []
                if skip_conn:
                    for j in range(num_he):
                        skip_conn_arr.append(
                            ret[self.he_pfs[j] + "sc_layers"][-i])
                offset = 0
                ''' calculate lmv for each history_encoding '''
                for he_id in range(num_he):
                    def calculate_pyramid_flow(name, in_offset, in_flow_c, is_enable_act=True):
                        ''' calc smallest flow and upper residual '''
                        ret[f"pred_layer_{layer_id}_{name}_{he_id}_raw"] = tmp_tensor[:,
                                                                                      in_offset:in_offset + in_flow_c]
                        if is_enable_act:
                            ret[f'pred_layer_{layer_id}_{name}_{he_id}'] = torch.tanh(
                                ret[f'pred_layer_{layer_id}_{name}_{he_id}_raw'])
                        else:
                            ret[f'pred_layer_{layer_id}_{name}_{he_id}'] = ret[f'pred_layer_{layer_id}_{name}_{he_id}_raw']
                        if i > 1:
                            ''' calc larger flow '''
                            ret[f"pred_layer_{layer_id}_{name}_{he_id}"] = ret[f"pred_layer_{layer_id}_{name}_{he_id}"] + resize(
                                ret[f"pred_layer_{layer_id+1}_{name}_{he_id}"], 2.0)

                        if self.enable_debug_fake_spatial:
                            ret[f"pred_layer_{layer_id}_{name}_{he_id}"] = set_fake(
                                ret[f"pred_layer_{layer_id}_{name}_{he_id}"])
                        return ret[f"pred_layer_{layer_id}_{name}_{he_id}"], in_offset + in_flow_c

                    if self.enable_smv0 and (he_id == 0 or self.enable_history_smv0):
                        ''' calc smallest flow and upper residual '''
                        smv0, offset = calculate_pyramid_flow(
                            "smv0", offset, self.layer_flow_channel, is_enable_act=not self.enable_no_sigmoid)
                    else:
                        smv0 = None

                    if self.enable_sep_smv_res and (he_id == 0 or self.enable_history_sep_smv_res):
                        smv_res, offset = calculate_pyramid_flow(
                            "smv_res", offset, self.layer_flow_channel, is_enable_act=not self.enable_no_sigmoid)
                    else:
                        smv_res = None

                    rmv = data[self.he_mv_name.format(
                        self.history_id[he_id])]

                    if not (self.enable_temporal_warped_feature) or self.enable_debug_fake_temporal:
                        rmv = set_fake(rmv)

                    tmv = get_lmv(rmv, smv0, smv_res, rmv_scale=1 / ratio)

                    if tmv is not None:
                        ret[f"pred_layer_{layer_id}_tmv_{he_id}"] = tmv
                        final_mv = tmv
                    else:
                        final_mv = resize(rmv, 1 / ratio)

                    if (self.enable_smv0 or self.enable_temporal_warped_feature) and final_mv is not None:
                        ''' last_dec only deal with he_id=0, no need to continue the for-loop(num_he), break here. '''
                        if warp_last_dec:
                            break
                        ''' if skip_conn == False, it means we dont need to warp the feature, break here. '''
                        if not (skip_conn):
                            break
                        ''' skip-connection start here, concat the warped history encoding '''
                        mode = self.config['config'].get('feature_warp_mode', 'bilinear')
                        skip_conn_arr[he_id] = warp(skip_conn_arr[he_id], final_mv, mode=mode,
                                                    padding_mode="border", flow_type="mv")

                ''' concat ops start '''
                cat_offset = 0

                def add_output_to_skip_conn_arr(he_id_list, size=1):
                    cat_offset = 0
                    if self.enable_smv0:
                        for he_id in he_id_list:
                            if he_id != 0 and not self.enable_history_smv0:
                                continue
                            skip_conn_arr.append(
                                ret[f'pred_layer_{layer_id}_smv0_{he_id}'])
                            cat_offset += self.layer_flow_channel

                    if self.enable_sep_smv_res:
                        for he_id in he_id_list:
                            if he_id != 0 and not self.enable_history_sep_smv_res:
                                continue
                            skip_conn_arr.append(
                                ret[f'pred_layer_{layer_id}_smv_res_{he_id}'])
                            cat_offset += self.layer_flow_channel

                    return cat_offset
                ''' warp_last_dec: no skip-connection in last decoder, so there's no nead to warping the feature '''
                ''' warp_last_dec: last decoder only calculates lmv for his_id == 0 '''
                ''' warp_last_dec: finish concat here in advance, and break for-loop(num_dec) here.'''
                if warp_last_dec:
                    cat_offset = add_output_to_skip_conn_arr([0])
                    skip_conn_arr.append(tmp_tensor[:, cat_offset:])
                    '''debug: check offset are equal between feature warping and final concat '''
                    if cat_offset != offset:
                        log.debug(
                            f"cat_offset: {cat_offset}, offset: {offset}.")
                    assert cat_offset == offset
                    break

                ''' skip_conn'''
                skip_conn_arr.append(ret[self.se_pf + "sc_layers"][-i])
                cat_offset = add_output_to_skip_conn_arr(range(num_he), 1 / ratio)
                skip_conn_arr.append(tmp_tensor[:, cat_offset:])

                if cat_offset != offset:
                    log.debug(f"cat_offset: {cat_offset}, offset: {offset}.")
                assert cat_offset == offset

                tmp_tensor = torch.cat(skip_conn_arr, dim=1)
            ''' end of if i>0 '''

            ''' i == num_dec means we are at output_block, the decoding is done here. '''
            if i == num_dec:
                break

            ''' upscale the feature with the decoder '''
            tmp_tensor = self.shade_decoder.decoders[i](
                tmp_tensor,
                skip_conn=None,
                skip_add=None)
            ''' compress the upscaled feature into input for recurrent feature streaming in future frames '''
            if self.enable_recurrent:
                ''' when i == num_dec -1 we are at last decoder where no recurrent feature streaming '''
                if i < num_dec - 1:
                    ret['sd_sc_layers'].append(
                        self.recurrent_encoders[i](tmp_tensor))
        ''' end of decoding loop '''

        ''' output_block at full-size '''
        if self.shade_decoder.output_block is not None:
            tmp_tensor = self.shade_decoder.output_block(tmp_tensor)

        ''' rearrange the output tensor into meaningful dict-like data '''
        tmp_output = self.shade_decoder.split_output_to_dict(
            tmp_tensor, prefix=self.shade_decoder.output_prefix)
        ret.update(tmp_output)

        ''' enable learnable motoin vector with upscale'''
        rmv = data[self.he_mv_name.format(
            self.history_id[0])]

        if self.enable_output_warp_upscale:
            tmv = get_lmv(data[f'merged_motion_vector_{0}'],
                          ret[f'pred_layer_{1}_smv0_{0}'],
                          ret.get(f'pred_layer_{1}_smv_res_{0}', None),
                          smv_scale=2.0)
            ret['pred_tmv_upscale'] = tmv

        if self.enable_output_warp1 and self.enable_output_warp1_upscale:
            tmv = get_lmv(data[f'merged_motion_vector_{0}'],
                          ret[f'pred_layer_{0}_smv0_{0}'],
                          ret.get(f'pred_layer_{0}_smv_res_{0}', None) if self.enable_no_st_warp1 else None,
                          re_scale=2.0)
            ret[f'pred_layer_{0}_tmv_{0}'] = tmv

        if self.enable_output_warp2:
            ret["pred_smv0"] = ret[f"pred_layer_{0}_smv0_{0}"] + torch.tanh(ret["pred_smv0_raw"])
            if self.enable_sep_smv_res:
                ret["pred_smv_res"] = ret[f"pred_layer_{0}_smv_res_{0}"] + torch.tanh(ret["pred_smv_res_raw"])

            scale = 2.0 if self.enable_output_warp2_upscale else 1.0
            tmv = get_lmv(data[f'merged_motion_vector_{0}'],
                          ret['pred_smv0'],
                          ret.get('pred_smv_res', None),
                          re_scale=scale)
            ret['pred_tmv'] = tmv

        mv_owu = ret['pred_tmv_upscale'] if self.enable_output_warp_upscale else None
        mv_ow1 = ret[f'pred_layer_{0}_tmv_{0}'] if self.enable_output_warp1 else None
        mv_ow2 = ret['pred_tmv'] if self.enable_output_warp2 else None

        ret["residual_item"] = data[self.config["residual_item"]]
        dmdl_color = fix_dmdl_color_zero_value(data['dmdl_color'])
        if self.enable_demodulate_before_warp:
            ret['residual_item'] = get_demodulated_color(inv_tonemap_func(ret['residual_item']), data['history_dmdl_color_0'])
            ret['residual_item'] = tonemap_func(ret['residual_item'])

        ''' no need for output block '''
        mode = "bilinear"
        if self.enable_output_warp_upscale and self.enable_no_st_warp_upscale:
            ret['residual_item'] = warp(
                ret['residual_item'], mv_owu, mode=mode, padding_mode="border", flow_type="mv")
            ret['pred_recurrent_lmv'] = mv_owu
        elif self.enable_output_warp1 and self.enable_no_st_warp1:
            ret['residual_item'] = warp(
                ret['residual_item'], mv_ow1, mode=mode, padding_mode="border", flow_type="mv")
            ret['pred_recurrent_lmv'] = mv_ow1
        elif self.enable_output_warp2 and self.enable_no_st_warp2:
            ret['residual_item'] = warp(
                ret['residual_item'], mv_ow2, mode=mode, padding_mode="border", flow_type="mv")
            ret['pred_recurrent_lmv'] = mv_ow2
        else:
            ret['residual_item'] = warp(
                ret['residual_item'], data['merged_motion_vector_0'], mode=mode,
                padding_mode="border", flow_type="mv")
            ret['pred_recurrent_lmv'] = data['merged_motion_vector_0']

        if self.enable_demodulate_after_warp:
            ret['residual_item'] = get_demodulated_color(inv_tonemap_func(ret['residual_item']), dmdl_color)
            ret['residual_item'] = tonemap_func(ret['residual_item'])

        if self.method == "residual":
            ''' FIXME: temporary solution for grad nan '''
            residual_item_max = ret['residual_item'].max().item()
            ret['residual_output'] = torch.clamp(
                ret['residual_output'], -residual_item_max, residual_item_max)

            if ret['residual_output'].shape[1] == 3:
                ret['pred_scene_light_no_st'] = ret['residual_output'] + \
                    ret['residual_item']
                ret['pred_scene_light_no_st'] = torch.clamp(ret['pred_scene_light_no_st'], 0, residual_item_max)
                ret['pred_scene_light_no_st'] = inv_tonemap_func(ret['pred_scene_light_no_st'])
                if self.enable_demodulate:
                    ret['pred_scene_color_no_st'] = ret['pred_scene_light_no_st'] * dmdl_color
                else:
                    ret['pred_scene_color_no_st'] = ret['pred_scene_light_no_st'].clone()
            else:
                raise Exception("pred_residual's output channel size must be 3, it's {} now".format(
                    ret['residual_output'].shape[1]))

            ret['pred'] = ret['pred_scene_color_no_st'].clone()

        ret['pred_comp_color_before_sky_st'] = ret['pred'].clone()
        color_names = self.config['st_color_names']
        history_names = self.config['st_history_names']
        for ind, name in enumerate(color_names):
            warped_color = None
            warped = False
            if name.startswith("st_"):
                if not self.enable_st:
                    continue
                if self.enable_output_warp1 and self.enable_st_warp1:
                    warped_color = warp(
                        data[history_names[ind]], mv_ow1, mode="bilinear", padding_mode="border")
                    # log.debug(f"warp {name} by mv_ow1")
                    warped = True
                elif self.enable_output_warp2 and self.enable_st_warp2:
                    warped_color = warp(
                        data[history_names[ind]], mv_ow2, mode="bilinear", padding_mode="border")
                    # log.debug(f"warp {name} by mv_ow2")
                    warped = True
                elif self.enable_output_warp_upscale and self.enable_st_warp_upscale:
                    warped_color = warp(
                        data[history_names[ind]], mv_owu, mode="bilinear", padding_mode="border")
                    # log.debug(f"warp {name} by mv_ou")
                    warped = True
                if not warped:
                    warped_color = warp(data[history_names[ind]], data[self.he_mv_name.format(
                        0)], mode="bilinear", padding_mode="border")
            elif name.startswith("sky_"):
                if self.enable_output_warp1 and self.enable_sky_warp1:
                    warped_color = warp(
                        data[history_names[ind]], mv_ow1, mode="bilinear", padding_mode="border")
                    warped = True
                    # log.debug(f"warp {name} by mv_ow1")
                elif self.enable_output_warp2 and self.enable_sky_warp2:
                    warped_color = warp(
                        data[history_names[ind]], mv_ow2, mode="bilinear", padding_mode="border")
                    # log.debug(f"warp {name} by mv_ow2")
                    warped = True
                elif self.enable_output_warp_upscale and self.enable_sky_warp_upscale:
                    warped_color = warp(
                        data[history_names[ind]], mv_owu, mode="bilinear", padding_mode="border")
                    # log.debug(f"warp {name} by mv_ou")
                    warped = True
                if not warped:
                    warped_color = warp(data[history_names[ind]], data[self.he_mv_name.format(
                        0)], mode="bilinear", padding_mode="border")
            assert warped_color is not None

            if name != 'sky_color' and self.enable_st_residual or name == 'sky_color' and self.enable_sky_residual:
                val = data[history_names[ind]].max()
                ret[f'pred_{name}_residual'] = torch.clamp(
                    ret[f'pred_{name}_residual'], -val, val)
                ret[f'pred_{name}'] = warped_color + \
                    ret[f'pred_{name}_residual']
                ret[f'pred_{name}'] = torch.clamp(ret[f'pred_{name}'], warped_color.min(), warped_color.max())
            else:
                ret[f'pred_{name}'] = warped_color

            if not name.endswith('_color'):
                ret[f'pred_{name}'] = torch.clamp(
                    ret[f'pred_{name}'], 0, 1)
            else:
                ret[f'pred_{name}'] = inv_tonemap_func(ret[f'pred_{name}'])

        ret['pred_comp_color_sky'] = ret['pred_comp_color_before_sky_st'] * \
            (1 - data['skybox_mask']) + \
            ret['pred_sky_color'] * data['skybox_mask']
        ret['pred_scene_color_no_st'] = ret['pred_comp_color_sky'].clone()
        
        if self.enable_st:
            ret['pred_comp_color_sky_st'] = ret['pred_comp_color_sky'] * \
                ret['pred_st_alpha'] + ret['pred_st_color']
            ret['pred'] = ret['pred_comp_color_sky_st'].clone()
        else:
            ret['pred'] = ret['pred_comp_color_sky'].clone()
            
        

        if not self.enable_timing:
            if self.enable_debug_no_st:
                ret['gt_comp'] = (inv_tonemap_func(data["scene_light_no_st"]) * dmdl_color *
                                        (1 - data['skybox_mask']) + inv_tonemap_func(data['sky_color']) *
                                        data['skybox_mask'])
            else:
                ret['gt_comp'] = (inv_tonemap_func(data["scene_light_no_st"]) * dmdl_color *
                                        (1 - data['skybox_mask']) + inv_tonemap_func(data['sky_color']) *
                                        data['skybox_mask']) * data['st_alpha'] + inv_tonemap_func(data['st_color'])
                
            ret['residual_item'] = inv_tonemap_func(ret['residual_item'])
            ret['skybox_mask'] = data['skybox_mask']
            if self.enable_debug_no_st:
                ret['gt'] = inv_tonemap_func(data['scene_color_no_st'])
            else:
                ret['gt'] = inv_tonemap_func(data['scene_color'])

        return self.return_output_by_type(ret)


ones_map_cache = {}


def get_ones_map(data, device):
    b, c, h, w = data.shape
    k = f"{b}_{c}_{h}_{w}"
    if k not in ones_map_cache.keys():
        ones_map_cache[k] = torch.ones_like(data, device=data.device)
    return ones_map_cache[k]


zeros_map_cache = {}


def get_zeros_map(data):
    b, c, h, w = data.shape
    k = f"{b}_{c}_{h}_{w}"
    if k not in zeros_map_cache.keys():
        zeros_map_cache[k] = torch.zeros_like(data, device=data.device)
    return zeros_map_cache[k]


class ShadeNetModelV5d4(ModelBaseEXT):
    def __init__(self, config):
        super().__init__(config)
        global mu
        mu = self.trainer_config['buffer_config']['mu']

    def get_dummy_input(self, bs=1):
        H, W = 720, 1280
        dump_input = {}
        input_2d_str = []
        input_2d_str += self.config['input_buffer']
        input_2d_str += self.config['shade_encoder']['input_buffer']
        input_1d_str = []
        for item in input_2d_str:
            tmp_tensor = torch.zeros(1, get_2d_dim(item), H, W)
            if self.use_cuda:
                tmp_tensor = tmp_tensor.cuda()
            dump_input[item] = tmp_tensor
        for item in input_1d_str:
            tmp_tensor = torch.zeros(1, get_1d_dim(item))
            if self.use_cuda:
                tmp_tensor = tmp_tensor.cuda()
            dump_input[item] = tmp_tensor
        return dump_input

    def get_data_to_input(self, data):
        return data_to_input(data, self.config)

    def get_augment_data(self, data):
        return history_extend(data, self.trainer_config)

    def create_model(self):
        history_encoders_config = self.config['history_encoders']
        history_encoders_config['inputs'] = []
        history_encoders_config['output_prefixs'] = []
        for ind in range(history_encoders_config['num']):
            inputs = []
            for name in history_encoders_config['input_template']:
                inputs.append(name.format(
                    history_encoders_config['history_id'][ind]))
            history_encoders_config['inputs'].append(inputs)
            history_encoders_config['output_prefixs'].append(
                history_encoders_config['output_prefix_template'].format(ind))
        self.net = ShadeNet(self.config)

    def calc_loss(self, data, ret):
        net = self.get_net()
        num_he = int(net.num_history_encoder)  # type: ignore
        num_dec = int(net.num_shade_decoder_layer)  # type: ignore
        device = ret['gt'].device

        def add_zero_loss(name, loss_name, loss_pf, loss_fn=flow_loss, scale=1.0):
            name = 'pred_' + name
            if name in ret.keys():
                ret[f'{name}_{loss_pf}'] = loss_fn([ret[name], get_zeros_map(ret[name])]).mean() * scale
                ret[loss_name] += ret[f'{name}_{loss_pf}']

        def add_c1_loss(name, loss_name, loss_pf, loss_fn=flow_loss, scale=1.0, layer=1):
            name = "pred_" + name
            if not name.format(layer) in ret.keys():
                return
            if layer == 3:
                return
            elif layer == 2:
                last_layer = resize(ret[name.format(layer + 1)], 2.0)
                ret[f'{name.format(layer)}_{loss_pf}'] = loss_fn(
                    [last_layer, ret[name.format(layer)] - last_layer]).mean() * scale
                ret[loss_name] += ret[f'{name.format(layer)}_{loss_pf}']
            elif layer == 1 or layer == 0:
                last_layer_0 = resize(ret[name.format(layer + 1)], 2.0)
                last_layer_1 = resize(ret[name.format(layer + 2)], 4.0)
                ret[f'{name.format(layer)}_{loss_pf}'] = loss_fn(
                    [ret[name.format(layer)] - last_layer_0, last_layer_0 - last_layer_1]).mean() * scale
                ret[loss_name] += ret[f'{name.format(layer)}_{loss_pf}']
            else:
                return

        if net.enable_zero_flow_loss or net.enable_zero_flow2_loss or net.enable_c1_flow_loss:
            if net.enable_zero_flow_loss:
                ret['zero_flow_loss'] = 0.0
            if net.enable_zero_flow2_loss:
                ret['zero_flow2_loss'] = 0.0
            if net.enable_c1_flow_loss:
                ret['c1_flow_loss'] = 0.0
            tot = 0
            if net.enable_smv0:
                tot += 1
            if net.enable_sep_smv_res:
                tot += 1
            scale = 1
            for he_id in range(0, num_he):
                for i in range(1, num_dec + 1):
                    layer_id = num_dec - i
                    ratio = 2 ** max(1, layer_id)
                    scale = 16 / ratio
                    if net.enable_c1_flow_loss:
                        c1_flow_ratio = self.config["loss_config"]["c1_flow_ratio"]
                        add_c1_loss(f'layer_{"{}"}_smv0_{he_id}', "c1_flow_loss", "c1f_ls",
                                    loss_fn=flow_loss, layer=layer_id, scale=c1_flow_ratio * scale / tot)
                        add_c1_loss(f'layer_{"{}"}_smv_res_{he_id}', "c1_flow_loss", "c1f_ls",
                                    loss_fn=flow_loss, layer=layer_id, scale=c1_flow_ratio * scale / tot)
                    if net.enable_zero_flow_loss:
                        zero_flow_ratio = self.config["loss_config"]["zero_flow_ratio"]
                        add_zero_loss(f'layer_{layer_id}_smv0_{he_id}', "zero_flow_loss", "zf_ls",
                                      loss_fn=flow_loss, scale=zero_flow_ratio * scale / tot)
                        add_zero_loss(f'layer_{layer_id}_smv_res_{he_id}', "zero_flow_loss",
                                      "zf_ls", loss_fn=flow_loss, scale=zero_flow_ratio * scale / tot)
                    if net.enable_zero_flow2_loss:
                        zero_flow2_ratio = self.config["loss_config"]["zero_flow2_ratio"]
                        add_zero_loss(f'layer_{layer_id}_tmv_{he_id}', "zero_flow2_loss",
                                      "zfl2_ls", loss_fn=flow2_loss, scale=zero_flow2_ratio)

            if net.enable_output_warp2:
                scale = 8
                if net.enable_zero_flow_loss:
                    zero_flow_ratio = self.config["loss_config"]["zero_flow_ratio"]
                    add_zero_loss(f'smv{0}', "zero_flow_loss", "zf_ls", loss_fn=flow_loss, scale=zero_flow_ratio * scale / tot)
                    add_zero_loss('smv_res', "zero_flow_loss", "zf_ls", loss_fn=flow_loss, scale=zero_flow_ratio * scale / tot)
                if net.enable_zero_flow2_loss:
                    zero_flow2_ratio = self.config["loss_config"]["zero_flow2_ratio"]
                    add_zero_loss('tmv', "zero_flow_loss", "zf2_ls", loss_fn=flow2_loss, scale=zero_flow2_ratio * scale)

        if net.enable_rec_occ_loss:
            warp_mode = 'bilinear'
            warp_padding_mode = 'border'
            warped_gt = warp(data[f'history_scene_color_0'], data['merged_motion_vector_0'],
                             mode=warp_mode, padding_mode=warp_padding_mode)
            gt = data['scene_color']
            ret['occ_mask'] = torch.where(torch.abs((warped_gt) - (gt)) > 1, torch.ones_like(gt), torch.zeros_like(gt))
            ret['rec_occ_loss'] = l1_mask_loss([gamma_log(ret['pred']), gamma_log(ret['gt']),
                                               ret['occ_mask']]) / ret['occ_mask'].mean()

    def calc_preprocess_input(self, data):
        data = self.get_data_to_input(data)
        return data

    def inference_for_timing(self, data):
        self.set_eval()
        self.get_net().enable_timing = True  # type: ignore
        with torch.no_grad():
            output = self.forward(data)
        self.get_net().enable_timing = False  # type: ignore
        return output 