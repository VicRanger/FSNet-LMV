
from models.general.unet import UNetDecoder, UNetEncoder
from models.general.unet import single_conv
from models.general.common_structure import NetBase, create_act_func, create_norm_func
import torch.nn as nn
import torch

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
                                 norm_func_name=norm_func_name, act_func_name=act_func_name)
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
            last_c + cat_c, mid_c, act_func_name=act_func_name, norm_func_name=norm_func_name)
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
                                   norm_func_name=norm_func_name, act_func_name=act_func_name)
        self.output1x1 = nn.Conv2d(
            mid_channel, out_channel, kernel_size=1)

    def forward(self, data):
        ret = self.conv1x1(data)
        ret = self.output1x1(ret)
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

class RecurrentEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.compress_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, data):
        ret = self.compress_conv(data)
        ret = self.tanh(ret)
        return ret