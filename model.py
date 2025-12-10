# model.py
import torch
import torch.nn as nn
from config import cfg
from pe_methods import get_2d_sincos_pos_embed, RotaryEmbedding, apply_rotary_pos_emb, LearnablePE, MultiScalePE, PolarPE

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, use_rope=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        if self.use_rope:
            self.rope = RotaryEmbedding(head_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, N, H, D_head]

        if self.use_rope:
            # RoPE expects [B, N, H, D] but applies on N dimension
            # Generate freqs based on N (sequence length)
            freqs = self.rope(q) # [1, N, D_head]
            # Broadcast freqs to heads
            freqs = freqs.unsqueeze(2) # [1, N, 1, D_head]
            
            q = apply_rotary_pos_emb(q, freqs)
            k = apply_rotary_pos_emb(k, freqs)

        # Transpose for attention calculation: [B, H, N, D]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., use_rope=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, use_rope=use_rope)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, pe_type='baseline'):
        super().__init__()
        self.pe_type = pe_type
        
        # Patch Embed params
        self.img_size = cfg.img_size
        self.patch_size = cfg.patch_size
        self.num_patches = (cfg.img_size // cfg.patch_size) ** 2
        self.embed_dim = cfg.embed_dim

        # Patch Embedding Layer
        self.patch_embed = nn.Conv2d(cfg.in_chans, cfg.embed_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        
        # Positional Encoding Initialization
        if pe_type == 'baseline':
            # 2D Sinusoidal (Fixed)
            self.pos_embed = nn.Parameter(
                get_2d_sincos_pos_embed(self.embed_dim, self.img_size//self.patch_size, cls_token=True), 
                requires_grad=False
            )
        elif pe_type == 'multiscale':
            block_size = getattr(cfg, 'block_size', 2) 
            self.pe_module = MultiScalePE(self.embed_dim, self.img_size//self.patch_size, block_size=block_size)
        elif pe_type == 'polar':
            # DCPE (Cartesian + Polar)
            self.pe_module = PolarPE(self.embed_dim, self.img_size//self.patch_size)
        elif pe_type == 'custom':
            # Learnable PE
            self.pe_module = LearnablePE(self.num_patches, self.embed_dim)
        elif pe_type == 'rope':
            # RoPE adds nothing at the embedding stage
            pass
        
        self.pos_drop = nn.Dropout(p=cfg.dropout)

        # Transformer Blocks
        use_rope_in_block = (pe_type == 'rope')
        self.blocks = nn.ModuleList([
            Block(dim=cfg.embed_dim, num_heads=cfg.num_heads, mlp_ratio=cfg.mlp_ratio, 
                  drop=cfg.dropout, use_rope=use_rope_in_block)
            for _ in range(cfg.depth)
        ])

        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

    def forward(self, x):
        # x: [B, C, H, W]
        B = x.shape[0]
        x = self.patch_embed(x) # [B, D, H/p, W/p]
        x = x.flatten(2).transpose(1, 2) # [B, N, D]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # [B, N+1, D]

        # Apply PE (If not RoPE)
        if self.pe_type == 'baseline':
            x = x + self.pos_embed
        elif self.pe_type == 'custom':
            x = self.pe_module(x)
        elif self.pe_type == 'multiscale':
            x = self.pe_module(x)
        elif self.pe_type == 'polar':
            x = self.pe_module(x)
        
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # Use CLS token for classification
        cls_out = x[:, 0]
        return self.head(cls_out)