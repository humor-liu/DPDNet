import math

import torch
import torch.nn.functional as F
import numpy as np

def gaussian_kernel(disparity, ground_truth, sigma=1.0):
    gaussian = torch.exp(-((disparity - ground_truth) ** 2) / (2 * sigma ** 2)) / (math.sqrt(2*np.pi) * sigma)
    # #guiyihua
    # gaussian_sum = torch.sum(gaussian)
    # gaussian_norm = gaussian / gaussian_sum

    # return gaussian_norm

    return gaussian


# define cos_loss
def guss_loss_function(predicted_cost_volume, ground_truth_disparity, max_disparity, sigma=1.0):

    B, D, H, W = predicted_cost_volume.size()

    disparity_range = torch.arange(0, D).view(1, D, 1, 1).expand(B, D, H, W).to(ground_truth_disparity.device)
    ground_truth_disparity = ground_truth_disparity.unsqueeze(1).expand(B, D, H, W)

    gaussian = gaussian_kernel(disparity_range, ground_truth_disparity, sigma)

    p = predicted_cost_volume
    q = gaussian

    up = torch.sum(p * q, dim=[1, 2, 3])
    bottom = torch.sqrt(torch.sum((p ** 2), dim=[1, 2, 3])) * torch.sqrt(torch.sum((q ** 2), dim=[1, 2, 3]))
    cos_loss = -torch.mean(up / (bottom + 1e-8))
    return 0.5 * cos_loss

