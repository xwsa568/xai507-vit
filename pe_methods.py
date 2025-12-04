# pe_methods.py
import torch
import torch.nn as nn
import math

# (A) Baseline: 2D Sinusoidal Positional Encoding [cite: 32]
# 1D 시퀀스지만 2D 그리드 정보를 반영하여 생성
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True):
    """
    grid_size: int of the grid height and width
    return: pos_embed: [grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return torch.from_numpy(pos_embed).float()

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: [H, W] of positions to be encoded
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

import numpy as np

# (B) RoPE: Rotary Positional Embedding logic [cite: 33]
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: [batch, seq_len, head_dim]
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # [seq_len, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1) # [seq_len, dim]
        return emb[None, :, :] # [1, seq_len, dim] for broadcasting

def apply_rotary_pos_emb(x, freqs):
    # x: [B, N, H, D]
    # freqs: [1, N, 1, D] (RotaryEmbedding에서 이미 [freqs, freqs]로 2배 불려진 상태)
    
    d = x.shape[-1]
    
    # x를 절반으로 나눔 (예: 32 -> 16, 16)
    x1, x2 = x[..., :d//2], x[..., d//2:]
    
    # [수정된 부분] freqs도 x1의 크기에 맞춰 절반만 가져옴
    # freqs는 앞뒤가 똑같은 값으로 복사되어 있으므로 앞부분만 쓰면 됩니다.
    freqs_half = freqs[..., :d//2]
    
    sin_emb = freqs_half.sin()
    cos_emb = freqs_half.cos()
    
    # 회전 변환 적용
    # (x1 * cos - x2 * sin, x1 * sin + x2 * cos)
    return torch.cat([x1 * cos_emb - x2 * sin_emb, x1 * sin_emb + x2 * cos_emb], dim=-1)

# (C) Custom: Learnable Positional Encoding [cite: 36]
# 제안 방법 예시: 학습 가능한 파라미터로 위치 정보를 처음부터 학습 (BERT 방식)
class LearnablePE(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        # Initialize with small random noise
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * .02)

    def forward(self, x):
        return x + self.pos_embed