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


def fn_get_otsu_mask(x: torch.Tensor) -> torch.Tensor:
    x_float = x.float()  # Converts to float32 if needed
    threshold = torch.quantile(x_float.view(-1), 0.8)
    otsu_mask = (x >= threshold).to(dtype=x.dtype)  # Keep original dtype
    return otsu_mask
    

def fn_clean_mask(otsu_mask: torch.Tensor, x0: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
    """
    Replace CPU flood-fill with GPU circular mask around (x0, y0).
    Input x0, y0 must be 0-D tensors (not int!).
    
    Args:
        otsu_mask: (H, W) tensor on GPU
        x0: scalar tensor (y-coordinate, row index)
        y0: scalar tensor (x-coordinate, col index)
    
    Returns:
        ret_mask: (H, W) cleaned mask on same device/dtype
    """
    H, W = otsu_mask.shape
    device = otsu_mask.device
    dtype = otsu_mask.dtype

    # Create coordinate grids
    y_coords = torch.arange(H, device=device, dtype=torch.float32)
    x_coords = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (H, W)

    dy = yy - x0.float()  # x0 is row index → y-coordinate
    dx = xx - y0.float()  # y0 is col index → x-coordinate
    dist_sq = dx * dx + dy * dy

    # Circular mask with radius 8 (adjustable)
    radius = 8.0
    circular_mask = (dist_sq <= radius * radius).to(dtype=dtype)

    # Combine with Otsu mask
    ret_mask = otsu_mask * circular_mask
    return ret_mask

