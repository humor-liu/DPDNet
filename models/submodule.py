from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import cv2
import numpy as np
from models.Gauss import guss_loss_function

###############################################################################
""" Fundamental Building Blocks """
###############################################################################

def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels)
    )


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
        nn.BatchNorm3d(out_channels)
    )


def convbn_dws(inp, oup, kernel_size, stride, pad, dilation, second_relu=True):
    if second_relu:
        return nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad,
                      dilation=dilation, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=False)
            )
    else:
        return nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad,
                      dilation=dilation, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
            )


class MobileV1_Residual(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(MobileV1_Residual, self).__init__()

        self.stride = stride
        self.downsample = downsample
        self.conv1 = convbn_dws(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = convbn_dws(planes, planes, 3, 1, pad, dilation, second_relu=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

# MobileV4_Residual is AS-DSConv_2D
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6

class SEBlock(nn.Module):
    def __init__(self, inp, reduction=4):
        super(SEBlock, self).__init__()
        hidden_dim = inp // reduction
        self.fc1 = nn.Conv2d(inp, hidden_dim, 1, padding=0)
        self.fc2 = nn.Conv2d(hidden_dim, inp, 1, padding=0)

    def forward(self, x):
        se = torch.mean(x, dim=(2, 3), keepdim=True)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se

class MobileV4_Residual(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, use_se=True, use_hs=True):
        super(MobileV4_Residual, self).__init__()
        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expanse_ratio != 1:
            # pointwise
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(h_swish(inplace=True) if use_hs else nn.ReLU(inplace=True))

        # depthwise
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(h_swish(inplace=True) if use_hs else nn.ReLU(inplace=True))

        # Squeeze-and-Excite (optional)
        if use_se:
            layers.append(SEBlock(hidden_dim))

        # pointwise linear
        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(oup))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# MobileV4_Residual_3D is AS-DSConv_3D
class SEBlock_3D(nn.Module):
    def __init__(self, inp, reduction=4):
        super(SEBlock_3D, self).__init__()
        hidden_dim = inp // reduction
        self.fc1 = nn.Conv3d(inp, hidden_dim, 1, padding=0)
        self.fc2 = nn.Conv3d(hidden_dim, inp, 1, padding=0)

    def forward(self, x):
        se = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se

class MobileV4_Residual_3D(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, use_se=True, use_hs=True):
        super(MobileV4_Residual_3D, self).__init__()
        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = (stride == (1, 1, 1) and inp == oup)

        layers = []
        if expanse_ratio != 1:
            # pointwise
            layers.append(nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm3d(hidden_dim))
            layers.append(h_swish(inplace=True) if use_hs else nn.ReLU(inplace=True))

        # depthwise
        layers.append(nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm3d(hidden_dim))
        layers.append(h_swish(inplace=True) if use_hs else nn.ReLU(inplace=True))

        # Squeeze-and-Excite (optional)
        if use_se:
            layers.append(SEBlock_3D(hidden_dim))

        # pointwise linear
        layers.append(nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm3d(oup))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

###############################################################################
""" Feature Extraction """
###############################################################################

class feature_extraction(nn.Module):
    def __init__(self, add_relus=False):
        super(feature_extraction, self).__init__()

        self.expanse_ratio = 3
        self.inplanes = 32
        if add_relus:
            self.firstconv = nn.Sequential(MobileV4_Residual(3, 32, 2, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV4_Residual(32, 32, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV4_Residual(32, 32, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True)
                                           )
        else:
            self.firstconv = nn.Sequential(MobileV4_Residual(3, 32, 2, self.expanse_ratio),
                                           MobileV4_Residual(32, 32, 1, self.expanse_ratio),
                                           MobileV4_Residual(32, 32, 1, self.expanse_ratio)
                                           )

        self.layer1 = self._make_layer(MobileV1_Residual, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(MobileV1_Residual, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(MobileV1_Residual, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(MobileV1_Residual, 128, 3, 1, 1, 2)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
                # Domain_2d(planes),
            )

        layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        feature_volume = torch.cat((l2, l3, l4), dim=1)

        return feature_volume
#
# ###############################################################################
# """ Cost Volume Related Functions """
# ###############################################################################
#
#
def interweave_tensors(refimg_fea, targetimg_fea):
    B, C, H, W = refimg_fea.shape
    interwoven_features = refimg_fea.new_zeros([B, 2 * C, H, W])
    interwoven_features[:,::2,:,:] = refimg_fea
    interwoven_features[:,1::2,:,:] = targetimg_fea
    interwoven_features = interwoven_features.contiguous()
    return interwoven_features


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

###############################################################################
""" Disparity Regression Function """
###############################################################################


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)

###############################################################################
""" Loss Function """
###############################################################################

def model_loss(disp_ests, disp_gt, mask,mode,gass_volume):
    if mode == 'train':
        weights = [0.5, 0.5, 0.7, 1.0]
        all_losses = []
        for disp_est, weight, gv in zip(disp_ests[:4], weights, gass_volume):
            all_losses.append(weight * F.l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
            all_losses.append(weight * guss_loss_function(gv, disp_gt, max_disparity=192, sigma=0.5))
        loss = sum(all_losses)
        if torch.isnan(loss): loss.data = torch.Tensor([0.001]).cuda()
        return loss

    elif mode == 'test':
        weights = [0.5, 0.5, 0.7, 1.0]
        all_losses = []
        for disp_est, weight in zip(disp_ests, weights):
            all_losses.append(weight * F.l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
        loss = sum(all_losses)
        if torch.isnan(loss): loss.data = torch.Tensor([0.])

        return loss
