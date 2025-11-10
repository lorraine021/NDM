import numpy as np
import torch
from torch.nn import functional as F

from utils.gaussian_smoothing import GaussianSmoothing

def fn_get_topk(attention_map, K=1):
    H, W = attention_map.size()
    attention_map_detach = attention_map.detach().view(H * W)
    topk_value, topk_index = attention_map_detach.topk(K, dim=0, largest=True, sorted=True)
    topk_coord_list = []

    for index in topk_index:
        index = index.cpu().numpy()
        coord = index // W, index % W
        topk_coord_list.append(coord)
    return topk_coord_list, topk_value


def fn_smoothing_func(attention_map):
    smoothing = GaussianSmoothing().to(attention_map.device)
    attention_map = F.pad(attention_map.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
    attention_map = smoothing(attention_map).squeeze(0).squeeze(0)
    return attention_map


def fn_show_attention(
    cross_attention_maps,
    self_attention_maps,
    indices,
    K=1,
    attention_res=16,
    smooth_attentions=True,
):

    cross_attention_map_list, self_attention_map_list = [], []

    # cross attention map preprocessing
    cross_attention_maps = cross_attention_maps[:, :, 1:-1]
    cross_attention_maps = cross_attention_maps * 100
    cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-1)

    # Shift indices since we removed the first token
    indices = [index - 1 for index in indices]

    for i in indices:
        cross_attention_map_per_token = cross_attention_maps[:, :, i]
        if smooth_attentions: cross_attention_map_per_token = fn_smoothing_func(cross_attention_map_per_token)
        cross_attention_map_list.append(cross_attention_map_per_token)

    all_topk_coord_list = []
    topk_value_list = []
    for i in indices:
        cross_attention_map_per_token = cross_attention_maps[:, :, i]
        topk_coord_list, topk_value = fn_get_topk(cross_attention_map_per_token, K=K)
        # print(i,"topk:",topk_coord_list,topk_value)
        all_topk_coord_list.append(topk_coord_list)
        topk_value_list.append(topk_value)

        self_attention_map_per_token_list = []
        for coord_x, coord_y in topk_coord_list:

            self_attention_map_per_token = self_attention_maps[coord_x, coord_y]
            self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()
            self_attention_map_per_token_list.append(self_attention_map_per_token)

        if len(self_attention_map_per_token_list) > 0:
            self_attention_map_per_token = sum(self_attention_map_per_token_list) / len(self_attention_map_per_token_list)
            if smooth_attentions: self_attention_map_per_token = fn_smoothing_func(self_attention_map_per_token)
        else:
            self_attention_map_per_token = torch.zeros_like(self_attention_maps[0, 0])
            self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()

        norm_self_attention_map_per_token = (self_attention_map_per_token - self_attention_map_per_token.min()) / \
            (self_attention_map_per_token.max() - self_attention_map_per_token.min() + 1e-6)
        
        self_attention_map_list.append(norm_self_attention_map_per_token)

    # tensor to numpy
    cross_attention_map_numpy       = torch.cat(cross_attention_map_list, dim=0).cpu().detach().numpy()
    self_attention_map_numpy        = torch.cat(self_attention_map_list, dim=0).cpu().detach().numpy()

    # obvious_list = []
    # for i in range(len(topk_value_list)):
        # if topk_value_list[i] >= 0.1:
            # print(i,topk_value_list[i],all_topk_coord_list[i])
            # obvious_list.append(i)
    return cross_attention_map_numpy, self_attention_map_numpy



def fn_get_otsu_mask(x: torch.Tensor) -> torch.Tensor:
    x_float = x.float()  # Converts to float32 if needed
    threshold = torch.quantile(x_float.view(-1), 0.8)
    otsu_mask = (x >= threshold).to(dtype=x.dtype)  # Keep original dtype
    return otsu_mask


def fn_clean_mask(otsu_mask: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    GPU-friendly approximation of connected component cleaning.
    Keeps only pixels within a circular region around (x, y).
    
    Args:
        otsu_mask: (H, W) binary mask on GPU (values 0 or 1)
        x: scalar tensor (row index, height direction)
        y: scalar tensor (col index, width direction)
    
    Returns:
        ret_otsu_mask: (H, W) cleaned mask, same dtype/device as input
    """
    H, W = otsu_mask.shape
    device = otsu_mask.device
    dtype = otsu_mask.dtype

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    dy = grid_y - x.float()
    dx = grid_x - y.float()
    dist_sq = dx * dx + dy * dy
    radius = 8.0  # e.g., for 64x64 or 96x96 maps
    circular_mask = (dist_sq <= radius * radius).to(dtype=dtype)
    ret_otsu_mask = otsu_mask * circular_mask
    return ret_otsu_mask