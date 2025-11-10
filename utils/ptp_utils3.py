import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F

# from diffusers.models.attention_processor import Attention
from diffusers.models.attention_processor import Attention, AttentionProcessor

class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if self.cur_att_layer >= 0:
            # print(attn.shape[1],np.prod(self.attn_res))
            # Check if the attention map matches the expected resolution
            H, W = self.attn_res          # (96, 96)
            H = H / 4
            W = W / 4
            # print(H,W,attn.shape[1])
            if attn.shape[1] == H * W:    # 只有 Q == 9216 才保存
                self.step_store[key].append(attn)
            # if attn.shape[1] == np.prod(self.attn_res):
                # self.step_store[key].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        # print("get:",average_attention)
        return average_attention

  
   
    def aggregate_attention(self, from_where: List[str], is_cross: bool = True) -> torch.Tensor:
        """Aggregates the attention with debugging info."""
        out = []
        attention_maps = self.get_average_attention()
        
        # print("Debugging attention map sizes:")
        for location in from_where:
            key = f"{location}_{'cross' if is_cross else 'self'}"
            if key in attention_maps and len(attention_maps[key]) > 0:
                for i, item in enumerate(attention_maps[key]):
                    # print(f"Map {i}: shape={item.shape}, total_elements={item.numel()}")
                    
                    batch_heads, seq_len, target_len = item.shape
                    H = W = int(seq_len ** 0.5)
                    
                    if H * W == seq_len:
                        cross_maps = item.reshape(batch_heads, H, W, target_len)
                    else:
                        # print(f"Warning: seq_len {seq_len} is not a perfect square")
                        for dim in [64, 128, 256]:
                            if seq_len % dim == 0:
                                H, W = dim, seq_len // dim
                                break
                        else:
                            H, W = 1, seq_len
                        
                        cross_maps = item.reshape(batch_heads, H, W, target_len)
                    
                    out.append(cross_maps)
                    # print(f"Reshaped to: {cross_maps.shape}")
        
        if len(out) == 0:
            raise ValueError(f"No attention maps found for keys: {from_where}")
        
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res):
        """
        Initialize an empty AttentionStore.
        :param attn_res: The expected resolution of the attention maps (height, width).
        """
        self.num_att_layers = -1  # Will be set dynamically based on the number of attention layers
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res  # Expected resolution of attention maps

    def step_end(self): # 新添加
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_step_store_keys(self):
        """Returns a list of all valid keys in the current step_store."""
        return list(self.step_store.keys())

class AttendExciteAttnProcessor:
    def __init__(self, attnstore, place_in_unet, is_cross=False):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.is_cross = is_cross

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        # print("=== DEBUG: Attention Scores Calculation ===")
        # print(f"query shape: {query.shape}")      # 应该是 (batch * heads, seq_len, head_dim)
        # print(f"key shape: {key.shape}")          # 应该是 (batch * heads, seq_len, head_dim)
        # print(f"attention_mask shape: {attention_mask.shape if attention_mask is not None else 'None'}")
        # print(f"query dtype: {query.dtype}, key dtype: {key.dtype}")
        # print(f"query ndim: {query.ndim}, key ndim: {key.ndim}")
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Store attention maps if they require gradients (during Attend and Excite process)
        if attention_probs.requires_grad:
            self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
