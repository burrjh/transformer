import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from utils import MLP_small, NodeEncoding, PositionalEncoding


class SelfAttention(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        assert cfg.dim % cfg.n_heads == 0

        self.dim = cfg.dim
        self.n_heads = cfg.n_heads

        self.attention_fc = nn.Linear(cfg.dim, 3*cfg.dim)

        self.resid_dropout = nn.Dropout(cfg.resid_dropout_rate)
        self.attn_dropout = nn.Dropout(cfg.attn_dropout_rate)
        self.proj = nn.Linear(cfg.dim, cfg.dim)

    def forward(self, x):
        b, s, d = x.size() # batchsize, sequence length, dimensionality

        # big matrix multiplication, creating K,V,Q at once
        Q,K,V = self.attention_fc(x).split(self.dim, dim=2) # split along hidden dim

        # split each matrix into heads and take appropriate view
        Q = Q.view(b,s,self.n_heads, d // self.n_heads).transpose(1,2)
        K = K.view(b,s,self.n_heads, d // self.n_heads).transpose(1,2)
        V = V.view(b,s,self.n_heads, d // self.n_heads).transpose(1,2) # -> [b,n_heads,s,hidden]

        attention = (Q@K.transpose(-2,-1)) * (1.0/math.sqrt(K.size(-1)))

        ## attention is  [b, n_heads, S, S]; quadratic is sequence length,  first two dims are broadcasted
        attention = F.softmax(attention, dim=-1) # row wise softmax
        attention = self.attn_dropout(attention)

        # matmul along last two dimensions broadcasting the first two
        y = attention @ V # [b,n_heads, S,S] x [B,n_heads, S, hidden] -> [b,n_heads, S, hidden] 
        y = y.transpose(1,2).contiguous().view(b,s,d) # re-assemble heads side by side
        y = self.resid_dropout(self.proj(y))

        return y




class Block(nn.Module):
    "basic Transformer Block"

    def __init__(self, cfg):
        super().__init__()

        self.attention = SelfAttention(cfg)
        self.mlp = MLP_small(cfg)  

        self.layernorm1 = torch.nn.LayerNorm(cfg.dim)
        self.layernorm2 = torch.nn.LayerNorm(cfg.dim)

    def forward(self, x):
        # split into self attention ("looking around")
        # and MLP "pondering about what I saw"
        # each time with residual connection
        # and pre-norm (modern version of layernorm)

        x = x + self.attention(self.layernorm1(x)) # Attention
        x = x + self.mlp(self.layernorm2(x)) # MLP

        return x


class VanillaTransformer(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()

        assert cfg.vocab_size is not None
        assert cfg.block_size is not None
        
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx=1)
        self.positional_enc = PositionalEncoding(cfg.dim, cfg.block_size)
        self.dropout_embd = nn.Dropout(cfg.embd_dropout_rate)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.decoder = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):  # resid projection layer of self attention module gets different init
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * 6))
        print("Transformer Model with ", {self.get_num_params()/1e6}, "Million Parameters initialized.")

    def forward(self, idx):

        token_embd = self.token_embedding(idx)              # (b, s, dim)
        positional_embd = self.positional_enc(idx)          # (1, s, dim)

        X = token_embd + positional_embd
        X = self.dropout_embd(X) # maybe only on positional encoding? #todo

        for block in self.blocks:
            X = block(X)

        return self.decoder(X)

        
    def get_num_params(self):

        n_params = sum(p.numel() for p in self.blocks.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)



    def configure_optimizer(self, train_config):
        """
        Seperare parameters in two groups: A:[biases,embeddings, layernom] B:[layers where weight decay is applied]
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn 
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer



class AttentionBias(nn.Module):
    def __init__(self, num_node_types, enc_dim, num_heads) -> None:
        super().__init__()

        self.EncodeNodes = NodeEncoding(num_node_types=num_node_types, enc_dim=int(enc_dim*num_heads))
        self.num_heads = num_heads
        self.enc_dim   = enc_dim

    def forward(self, node_idx):

        node_enc = self.EncodeNodes(node_idx) # [b, S, enc_dim*num_heads]

        b, S, _ = node_enc.shape

        node_enc.view(b, S, self.num_heads, self.enc_dim).transpose(1,2) # b,nheads, S, d

        attention_bias = (node_enc @ node_enc.transpose(-2,-1)) * (1.0/math.sqrt(node_enc.size(-1)))

        attention_bias = F.softmax(attention_bias, dim=-1)

        return attention_bias 




class BiasedSelfAttention(nn.Module):
    """
    Attention Mechanism whith a bias term derived from a embedding of each token
    """
    def __init__(self, cfg) -> None:
        super().__init__()

        assert cfg.dim % cfg.n_heads == 0

        self.dim = cfg.dim
        self.n_heads = cfg.n_heads

        self.attention_fc = nn.Linear(cfg.dim, 3*cfg.dim)
        self.proj = nn.Linear(cfg.dim, cfg.dim)


    def forward(self, x, attn_B):

        b, s, d = x.size() # batchsize, sequence length, dimensionality

        # big matrix multiplication, creating K,V,Q at once
        Q,K,V = self.attention_fc(x).split(self.dim, dim=2) # split along hidden dim

        # split each matrix into heads and take appropriate view
        Q = Q.view(b,s,self.n_heads, d // self.n_heads).transpose(1,2)
        K = K.view(b,s,self.n_heads, d // self.n_heads).transpose(1,2)
        V = V.view(b,s,self.n_heads, d // self.n_heads).transpose(1,2) # -> [b,n_heads,s,hidden]

        attention = (Q@K.transpose(-2,-1)) * (1.0/math.sqrt(K.size(-1)))

        ## attention is  [b, n_heads, S, S]; quadratic is sequence length,  first two dims are broadcasted
        attention = F.softmax(attention, dim=-1) # row wise softmax

        attention = attention + attn_B

        # matmul along last two dimensions broadcasting the first two
        y = attention @ V # [b,n_heads, S,S] x [B,n_heads, S, hidden] -> [b,n_heads, S, hidden] 

        y = y.transpose(1,2).contiguous().view(b,s,d) # re-assemble heads side by side

        return y


class BiasedBlock(nn.Module):
    "Transformer Block with SyntaxBiased Self Attention"

    def __init__(self, cfg):
        super().__init__()

        self.attention = BiasedSelfAttention(cfg)
        self.mlp = MLP_small(cfg)  

        self.layernorm1 = torch.nn.LayerNorm(cfg.dim)
        self.layernorm2 = torch.nn.LayerNorm(cfg.dim)

    def forward(self, x, attention_bias):
        
        x = x + self.attention(self.layernorm1(x), attention_bias) # Attention
        x = x + self.mlp(self.layernorm2(x)) # MLP

        return x

class BiasSyntaxBERT(nn.Module):
    """
    BERT-style Transformer Encoder which is biased to focus on structural changes via
    a biased attention mechanism
    """
    def __init__(self, cfg) -> None:
        super().__init__()

        assert cfg.vocab_size is not None
        assert cfg.block_size is not None
        
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx=1)
        self.clause_encoding = nn.Embedding(cfg.num_node_types, cfg.dim, padding_idx=1)
        self.positional_enc = PositionalEncoding(cfg.dim, cfg.block_size)
        self.dropout_embd = nn.Dropout(cfg.embd_dropout_rate)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])

        self.attentionBias = AttentionBias(cfg.num_node_types, cfg.dim, cfg.num_heads, )

        self.decoder = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):  # resid proejection layer of self attention module gets different init
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * 6))
        print("Transformer Model with ", {self.get_num_params()/1e6}, "Million Parameters initialized.")

    def forward(self, token_idx, node_idx):

        token_embd = self.token_embedding(token_idx)   
        clause_encoding = self.clause_encoding(node_idx).sum(-2)    # sum along node_types dim to get (b, s, dim)
        positional_embd = self.positional_enc(token_idx)                # (1, s, dim)


        # maybe dropout some of node_encoding?
        X = token_embd + clause_encoding + positional_embd
        X = self.dropout_embd(X) 

        for block in self.blocks:
            X = block(X)

        return self.decoder(X)

        
    def get_num_params(self):

        n_params = sum(p.numel() for p in self.blocks.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)



    def configure_optimizer(self, train_config):
        """
        Seperare parameters in two groups: A:[biases,embeddings, layernom] B:[layers where weight decay is applied]
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn 
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

