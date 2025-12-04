# config.py
import torch

class Config:
    # Data params
    img_size = 32
    patch_size = 4
    in_chans = 3
    num_classes = 10
    
    # Model params
    embed_dim = 256
    depth = 8
    num_heads = 4
    mlp_ratio = 4.
    dropout = 0.0
    
    # Training params
    batch_size = 512
    epochs = 100
    learning_rate = 5e-4
    weight_decay = 5e-2
    seed = 42
    
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'

cfg = Config()