from __future__ import print_function, division
import torch.nn.parallel
import torch.utils.data
from models import __models__
from utils import *
from thop import profile

C = 3
H = 256
W = 512
# F = 40

"""
For input size: (3, 256, 512):

    Feature size: (320, 64, 128)

    [2D model] Feature size after channel reduction: (32, 64, 128)

    [2D model] Cost size: (48, 64, 128)

    [3D model] Cost size: (40, 48, 64, 128)
        For this, change (1, C, H, W) to (1, F, C, H, W).
"""


def input_constructor(input_shape):
    # For Flops-Counter method
    # Notice the input naming
    inputs = {'L': torch.ones(input_shape), 'R': torch.ones(input_shape)}
    return inputs

with torch.cuda.device(0):
    ################# Using Flops-Counter #################
    #
    # net = __models__['DPDNet2D'](192)
    # macs2D, params2D = get_model_complexity_info(net, (1, C, H, W), as_strings=True,
    #                                              print_per_layer_stat=False, verbose=False,
    #                                              input_constructor=input_constructor)
    #
    # net = __models__['DPDNet3D'](192)
    # macs3D, params3D = get_model_complexity_info(net, (1, C, H, W), as_strings=True,
    #                                              print_per_layer_stat=False, verbose=False,
    #                                              input_constructor=input_constructor)
    #
    # print("==========================\n", '2D-DPDNet', "\n==========================")
    # print('{:<30}  {:<8}'.format('Number of operations: ', macs2D))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params2D))
    #
    # print("==========================\n", '3D-DPDNet', "\n==========================")
    # print('{:<30}  {:<8}'.format('Number of operations: ', macs3D))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params3D))

    ################# Using THOP (OpCounter) #################

    L = torch.randn(1, C, H, W)
    R = L

    macs2D, params2D = profile(__models__['DPDNet2D'](192), inputs=(L, R))
    macs3D, params3D = profile(__models__['DPDNet3D'](192), inputs=(L, R))

    print("==========================\n", '2D-DPDNet', "\n==========================")
    print('{:<30}  {:<8}'.format('Number of operations: ', np.round(macs2D / 1000000000), 5))
    print('{:<30}  {:<8}'.format('Number of parameters: ', np.round(params2D / 1000000, 5)))

    print("==========================\n", '3D-DPDNet', "\n==========================")
    print('{:<30}  {:<8}'.format('Number of operations: ', np.round(macs3D / 1000000000), 2))
    print('{:<30}  {:<8}'.format('Number of parameters: ', np.round(params3D / 1000000, 3)))
