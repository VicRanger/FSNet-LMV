import torch.nn as nn
import torch
from utils.str_utils import dict_to_string 
from utils.log import log

class ConvLSTMCellV1(nn.Module):

    def __init__(self, in_channel, hidden_channel, out_channel, kernel_size=1, bias=True):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCellV1, self).__init__()

        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.hidden_channel,
                              out_channels=self.out_channel * 3,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state, c_list=None):
        ''' 
        cc_i + cc_f + cc_g <--> input_tensor + cur_state
        cc_f <--> c_cur <--> cc_i <--> cc_g <--> c_next
        '''
        h_cur, c_cur = cur_state

        combined = torch.cat([h_cur, c_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)

        if c_list is None:
            c_list = [self.out_channel] * 3
        cc_i, cc_f, cc_g = torch.split(combined_conv, c_list, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        # o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * input_tensor + i * g
        return c_next