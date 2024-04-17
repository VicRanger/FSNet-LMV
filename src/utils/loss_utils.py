
import pytorch_msssim.ssim as pyssim
from utils.buffer_utils import aces_tonemapper
from lpips import LPIPS
import torch

lpips_kernel = None


def lpips(pred, gt, normalize=True):
    global lpips_kernel
    if lpips_kernel is None:
        lpips_kernel = LPIPS(net='vgg').cuda()
    kernel_device = next(lpips_kernel.parameters()).device
    if  kernel_device != pred.device:
        pred = pred.to(kernel_device)
    if  kernel_device != gt.device:
        gt = gt.to(kernel_device)
    return lpips_kernel(pred, gt, normalize=normalize)


def lpips_hdr(pred, gt, normalize=True):
    pred = aces_tonemapper(pred)
    gt = aces_tonemapper(gt)
    pred = torch.clamp(pred, 0, 1)
    gt = torch.clamp(gt, 0, 1)
    return lpips(pred, gt, normalize=normalize)


def ssim(pred, gt, size_average=True):
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(gt.shape) == 3:
        gt = gt.unsqueeze(0)
    return pyssim(pred, gt, data_range=1, size_average=size_average)


def ssim_hdr(pred, gt, size_average=True):
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(gt.shape) == 3:
        gt = gt.unsqueeze(0)
    pred = aces_tonemapper(pred)
    gt = aces_tonemapper(gt)
    pred = torch.clamp(pred, 0, 1)
    gt = torch.clamp(gt, 0, 1)
    return pyssim(pred, gt, data_range=1, size_average=size_average)


def psnr_hdr(pred, gt, **kwargs):
    return psnr_image_tonemapping(pred, gt)


def psnr_image_gamma(pred, gt):
    mse = ((torch.clamp(pred, 0, 1) ** (1 / 2.2)) - (torch.clamp(gt, 0, 1) ** (1 / 2.2))).square().mean()
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr


def psnr_image_tonemapping(pred, gt):
    mse = (torch.clamp(aces_tonemapper(pred), 0, 1) - torch.clamp(aces_tonemapper(gt), 0, 1)).square().mean()
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr


def psnr(pred, gt, **kwargs):
    return psnr_image(pred, gt)


def psnr_image(pred, gt):
    mse = (torch.clamp(pred, 0, 1) - torch.clamp(gt, 0, 1)).square().mean()
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr
