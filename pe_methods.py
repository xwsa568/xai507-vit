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

# pe_methods.py 하단에 추가

class MultiScalePE(nn.Module):
    def __init__(self, embed_dim, grid_size, block_size=2):
        super().__init__()
        assert embed_dim % 2 == 0
        
        # P1, P2 투영을 위한 Linear Layer (D -> D/2)
        # 위치 정보를 학습하여 압축
        self.proj_p1 = nn.Linear(embed_dim, embed_dim // 2)
        self.proj_p2 = nn.Linear(embed_dim, embed_dim // 2)
        
        # --- Grid 생성 ---
        # numpy grid 생성
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        
        # 1) P1: Fine-grained Grid (기존과 동일)
        grid_p1 = np.stack(grid, axis=0) # [2, H, W]
        grid_p1 = grid_p1.reshape([2, 1, grid_size, grid_size])
        
        # 2) P2: Coarse-grained Grid (Block 단위)
        # 좌표를 block_size로 나누고 내림하여 같은 블럭끼리 같은 좌표 공유
        grid_p2 = np.floor(np.stack(grid, axis=0) / block_size)
        grid_p2 = grid_p2.reshape([2, 1, grid_size, grid_size])

        # --- Sinusoidal Encoding 생성 ---
        # 기존 함수 재활용 (D 차원 생성)
        emb_p1 = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_p1) # [N, D]
        emb_p2 = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_p2) # [N, D]

        # --- CLS 토큰 (0으로 채움) 및 병합 ---
        emb_p1 = np.concatenate([np.zeros([1, embed_dim]), emb_p1], axis=0)
        emb_p2 = np.concatenate([np.zeros([1, embed_dim]), emb_p2], axis=0)

        # Buffer 등록 (학습되지 않는 고정 텐서지만, GPU 이동은 자동)
        self.register_buffer('p1', torch.from_numpy(emb_p1).float())
        self.register_buffer('p2', torch.from_numpy(emb_p2).float())

    def forward(self, x):
        # x: [B, N, D]
        
        # 1. Projection (D -> D/2)
        # 학습 가능한 가중치로 중요한 위치 정보 추출
        p1_feat = self.proj_p1(self.p1)
        p2_feat = self.proj_p2(self.p2)
        
        # 2. Concatenate (D/2 + D/2 -> D)
        pe = torch.cat([p1_feat, p2_feat], dim=-1) # [N, D]
        
        # 3. Add to input
        return x + pe.unsqueeze(0)

# (D) Polar PE: Cartesian + Polar Coordinate Positional Encoding
class PolarPE(nn.Module):
    def __init__(self, embed_dim, grid_size):
        super().__init__()
        assert embed_dim % 4 == 0  # 4등분 해야 하므로
        
        quarter_dim = embed_dim // 4
        
        # 1. Grid 생성 (x, y)
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(grid_w, grid_h) # x, y
        
        # 2. Polar 변환 (r, theta)
        center = (grid_size - 1) / 2.0
        x_centered = grid_x - center
        y_centered = grid_y - center
        
        r = np.sqrt(x_centered**2 + y_centered**2)
        theta = np.arctan2(y_centered, x_centered)
        
        # 3. 각각 인코딩 (x, y, r, theta)
        # x, y는 기존 방식대로
        # get_1d_sincos_pos_embed_from_grid는 [H, W] 입력을 받아 [H*W, D]를 반환함
        emb_x = get_1d_sincos_pos_embed_from_grid(quarter_dim, grid_x)
        emb_y = get_1d_sincos_pos_embed_from_grid(quarter_dim, grid_y)
        
        # r, theta는 Polar 방식대로
        emb_r = get_1d_sincos_pos_embed_from_grid(quarter_dim, r)
        emb_theta = get_1d_sincos_pos_embed_from_grid(quarter_dim, theta)
        
        # 4. 모두 합치기 (Concat)
        # [N, D/4 * 4] = [N, D]
        emb = np.concatenate([emb_x, emb_y, emb_r, emb_theta], axis=1)
        
        # CLS 토큰 추가
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
        
        self.register_buffer('pos_embed', torch.from_numpy(emb).float())

    def forward(self, x):
        return x + self.pos_embed.unsqueeze(0)