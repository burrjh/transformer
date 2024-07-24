import torch
import math
import torch.nn as nn
from torch.nn import functional as F

class GELU(nn.Module):
    """
    Gaussian Error Linear Units (GELU) 
    from https://arxiv.org/abs/1606.08415
    basically gives the ReLu a bump around zero which helps convergence in transformer models
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class MLPBlock(nn.Module):
    """
    MLP applied per Token after the self Attention operation
    The biggest amount of parameters lie here encoding world knowledge
    This is the GPT3 configuration upsampling to 4x dimensionality
    """
    def __init__(self, cfg) -> None:
        super().__init__()
        self.fc_up = nn.Linear(cfg.dim, 4*cfg.dim)
        self.fc_proj = nn.Linear(4 * cfg.dim, cfg.dim)
        self.dropout = nn.Dropout(cfg.mlp_dropout_rate)
        self.act = GELU()

    def forward(self, x):

        x = self.act(self.fc_up(x))         # upsample
        x = self.dropout(self.fc_proj(x))   # project down
        return x

class MLP_small(nn.Module):
    """ 
    MLP applied per Token
    preserving dimensionality [b,s,d] -> [b,s,d] 
    """
    def __init__(self, cfg) -> None:
        super().__init__()

        self.fc1 = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.fc2 = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.mlp_dropout_rate)
        self.act = GELU()

    def forward(self, x):

        x = self.act(self.fc1(x))               
        x = self.dropout(self.act(self.fc2(x)))

        return x       # [b,s,d]
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, idx):
        
        return self.pe[:,:idx.size(1),:]


class ClauseEncoding(nn.Module):

    def __init__(self, num_node_types: int, model_dim: int) -> None:
        super().__init__()

        clause_enc = nn.Embedding(num_node_types, model_dim)
        self.model_dim = model_dim
        self.register_buffer("clause_enc", clause_enc)

    def forward(self, node_idx: list):
        # node_idx is tensor of shape [bsize, S, num_nodes] , num_nodes will need to be padded

        # embed last dim and sum along second

        clause_encodings = self.clause_enc(node_idx)
        assert clause_encodings.shape[-1]== self.model_dim
        assert clause_encodings.dim == 4

        clause_encodings = clause_encodings.sum(-2)  # sum along the num_nodes dim

        return clause_encodings


class NodeEncoding(nn.Module):

    def __init__(self, num_node_types: int, enc_dim: int) -> None:
        super().__init__()

        node_enc = nn.Embedding(num_node_types, enc_dim)
        self.enc_dim = enc_dim
        self.register_buffer('node_enc', node_enc)

    def forward(self, node_idx: list):
        # node_idx is tensor of shape [bsize, S, num_nodes] , num_nodes will need to be padded

        # embed last dim and sum along second
        node_encodings = self.node_enc(node_idx)
        assert node_encodings.shape[-1]== self.enc_dim
        assert node_encodings.dim == 4

        node_encodings = node_encodings.sum(-2)  # sum along the num_nodes dim

        return node_encodings  #todo normalize?
