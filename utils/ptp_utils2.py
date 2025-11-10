import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F

from diffusers.models.attention_processor import Attention, AttentionProcessor

class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if self.cur_att_layer >= 0:
            # Check if the attention map matches the expected resolution
            H, W = self.attn_res          # (96, 96)
            if attn.shape[1] == H * W:   
                self.step_store[key].append(attn)
           
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def aggregate_attention(self, from_where, *, is_cross=True):
        H, W = self.attn_res
        running_sum = None    # (H, W, T)
        count = 0

        for place in from_where:
            key = f"{place}_{'cross' if is_cross else 'self'}"
            for m in self.attention_store.get(key, []):
                m = m.reshape(-1, H, W, m.shape[-1])   # (B⋅head, H, W, T)
                if running_sum is None:
                    running_sum = m.sum(0)
                else:
                    running_sum = running_sum + m.sum(0)
                count += m.shape[0]

        if running_sum is None:
            raise ValueError(f"No attention maps for {from_where}")

        return running_sum / count       


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

    def step_end(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()


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
