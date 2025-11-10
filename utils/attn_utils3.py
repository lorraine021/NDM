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

    obvious_list = []
    for i in range(len(topk_value_list)):
        if topk_value_list[i] >= 0.1:
            print(i,topk_value_list[i],all_topk_coord_list[i])
            obvious_list.append(i)
    return cross_attention_map_numpy, self_attention_map_numpy


import cv2


def fn_get_otsu_mask(x: torch.Tensor) -> torch.Tensor:
    x_float = x.float()  # Converts to float32 if needed
    threshold = torch.quantile(x_float.view(-1), 0.8)
    otsu_mask = (x >= threshold).to(dtype=x.dtype)  # Keep original dtype
    return otsu_mask

# def fn_get_otsu_mask(x):

#     x_numpy = x
#     x_numpy = x_numpy.cpu().detach().numpy()
#     x_numpy = x_numpy * 255
#     x_numpy = x_numpy.astype(np.uint16)

#     opencv_threshold, _ = cv2.threshold(x_numpy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     opencv_threshold = opencv_threshold * 1. / 255.

#     otsu_mask = torch.where(
#         x < opencv_threshold,
#         torch.tensor(0, dtype=x.dtype, device=x.device),
#         torch.tensor(1, dtype=x.dtype, device=x.device))
    
#     return otsu_mask

"""
def fn_clean_mask(otsu_mask, x, y):
    
    H, W = otsu_mask.size()
    direction = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    def dfs(cur_x, cur_y):
        if cur_x >= 0 and cur_x < H and cur_y >= 0 and cur_y < W and otsu_mask[cur_x, cur_y] == 1:
            otsu_mask[cur_x, cur_y] = 2
            for delta_x, delta_y in direction:
                dfs(cur_x + delta_x, cur_y + delta_y)
    
    dfs(x, y)
    ret_otsu_mask = torch.where(
        otsu_mask < 2,
        torch.tensor(0, dtype=otsu_mask.dtype, device=otsu_mask.device),
        torch.tensor(1, dtype=otsu_mask.dtype, device=otsu_mask.device))

    return ret_otsu_mask
"""

# def fn_clean_mask(otsu_mask: torch.Tensor, x0: int, y0: int) -> torch.Tensor:
#     """
#     连通域保留：
#     以 (x0, y0) 为种子，只保留与之相连且值==1 的像素区域，其余清零。
#     * 采用显式栈，避免 Python 递归深度限制。 *

#     Args
#     ----
#     otsu_mask : (H, W) 0/1 tensor
#     x0, y0    : 起始坐标

#     Returns
#     -------
#     ret_mask  : (H, W) 0/1 tensor，dtype 与输入一致
#     """
#     H, W = otsu_mask.shape
#     device = otsu_mask.device
#     dtype  = otsu_mask.dtype          # 保持数据类型

#     visited = torch.zeros_like(otsu_mask, dtype=torch.bool)
#     stack   = [(x0, y0)]              # 显式栈

#     # 4-邻域方向
#     dirs = [(1,0), (-1,0), (0,1), (0,-1)]

#     while stack:
#         cx, cy = stack.pop()
#         if (0 <= cx < H and 0 <= cy < W
#                 and otsu_mask[cx, cy] == 1
#                 and not visited[cx, cy]):
#             visited[cx, cy] = True
#             for dx, dy in dirs:
#                 stack.append((cx + dx, cy + dy))

#     # visited==True 的位置保留为 1，其余置 0
#     ret_mask = visited.to(dtype)

#     return ret_mask

def fn_clean_mask(otsu_mask: torch.Tensor, x0: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
    """
    GPU-friendly approximation of connected component:
    Keep only pixels within a circular region around (x0, y0).
    
    Args:
        otsu_mask: (H, W) tensor on GPU, binary (0/1)
        x0: scalar tensor (row index, height direction)
        y0: scalar tensor (col index, width direction)
    
    Returns:
        ret_mask: (H, W) tensor, same dtype/device as input
    """
    H, W = otsu_mask.shape
    device = otsu_mask.device
    dtype = otsu_mask.dtype

    # Create coordinate grids
    y_coords = torch.arange(H, device=device, dtype=torch.float32)
    x_coords = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (H, W)

    # Compute squared distance to seed (x0, y0)
    # Note: x0 is row (y), y0 is col (x)
    dy = yy - x0.float()
    dx = xx - y0.float()
    dist_sq = dx * dx + dy * dy

    # Use circular mask with adaptive or fixed radius
    # Option 1: fixed radius (e.g., 8 pixels)
    radius = 8.0
    local_mask = (dist_sq <= radius * radius).to(dtype=dtype)

    # Option 2 (advanced): adaptive radius based on attention spread
    # But fixed is simpler and works well

    # Combine with Otsu mask
    ret_mask = otsu_mask * local_mask
    return ret_mask