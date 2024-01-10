import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_min_frequency=1.0, embedding_max_frequency=1000.0, embedding_dims=32):
        super(SinusoidalEmbedding, self).__init__()

        frequencies = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(embedding_min_frequency)),
                torch.log(torch.tensor(embedding_max_frequency)),
                embedding_dims // 2
            ))

        self.register_buffer('angular_speeds', 2.0 * torch.pi * frequencies)

    def forward(self, x):
        angular_speeds = self.angular_speeds

        embeddings = torch.cat([torch.sin(angular_speeds * x),
                                torch.cos(angular_speeds * x)], dim=-1)
        return embeddings

class MHAttention(nn.Module):
    def __init__(self, is_causal=False, dropout_level=0, n_heads=4):
        super().__init__()
        self.is_causal = is_causal
        self.dropout_level = dropout_level
        self.n_heads = n_heads

    def forward(self, q, k, v, attn_mask=None):

        assert q.size(-1) == k.size(-1)
        assert k.size(-2) == v.size(-2)

        q, k, v = [rearrange(x, 'bs n (d h) -> bs h n d', h=self.n_heads) for x in [q,k,v]]
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        out = nn.functional.scaled_dot_product_attention(q, k, v,
                                                          attn_mask=attn_mask,
                                                          is_causal=self.is_causal,
                                                          dropout_p=self.dropout_level if self.training else 0)

        out = rearrange(out, 'bs h n d -> bs n (d h)', h=self.n_heads)

        return out

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, is_causal=False, dropout_level=0, n_heads=4):
        super().__init__()
        self.qkv_linear = nn.Linear(embed_dim, 3*embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)
        #self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=dropout_level) - doesn't work?

    def forward(self, x):
        q, k, v = self.qkv_linear(x).chunk(3, dim=2)
        return self.mha(q, k, v)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, is_causal=False, dropout_level=0, n_heads=4):
        super().__init__()
        self.kv_linear = nn.Linear(embed_dim, 2*embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)

    def forward(self, x, y):
        q = self.q_linear(x)
        k, v = self.kv_linear(y).chunk(2, dim=2)
        return self.mha(q,k,v)


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_multiplier * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout_level)
        )

    def forward(self, x):
        return self.mlp(x)

class MLPSepConv(nn.Module):
    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, mlp_multiplier*embed_dim, kernel_size=1, padding='same'),
            nn.Conv2d(mlp_multiplier*embed_dim, mlp_multiplier*embed_dim, kernel_size=3,
                      padding='same', groups=mlp_multiplier*embed_dim), #<- depthwise conv
            nn.GELU(),
            nn.Conv2d(mlp_multiplier*embed_dim, embed_dim, kernel_size=1, padding='same'),
            nn.Dropout(dropout_level)
            )

    def forward(self, x):
        w = h = int(np.sqrt(x.size(1))) #only square images for now
        x = rearrange(x, 'bs (h w) d -> bs d h w', h=h, w=w)
        x = self.mlp(x)
        x = rearrange(x, 'bs d h w -> bs (h w) d')
        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, is_causal, mlp_multiplier, dropout_level, mlp_class=MLP):
        super().__init__()
        self.self_attention = SelfAttention(embed_dim, is_causal, dropout_level, n_heads=embed_dim//64)
        self.mlp = mlp_class(embed_dim, mlp_multiplier, dropout_level)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.self_attention(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, is_causal, mlp_multiplier, dropout_level, mlp_class=MLP):
        super().__init__()
        self.self_attention = SelfAttention(embed_dim, is_causal, dropout_level, n_heads=embed_dim//64)
        self.cross_attention = CrossAttention(embed_dim, is_causal=False, dropout_level=0, n_heads=4)
        self.mlp = mlp_class(embed_dim, mlp_multiplier, dropout_level)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, y):
        x = self.norm1(self.self_attention(x) + x)
        x = self.norm2(self.cross_attention(x, y) + x)
        x = self.norm3(self.mlp(x) + x)
        return x

class Tower(nn.Module):
    # input is (bs,n,d) n sequences of dim d (e.g. word embeddings, or flattened image patches)
    # output is (bs, n,d) OR (bs,d) if global_pool is True

    def __init__(self, embed_dim, seq_len, n_layers, use_pos_embeddings,
                 dropout=0, n_heads=4, n_class=1, mlp_multiplier=2, is_causal=False, global_pool=False, 
                 block_class=EncoderBlock,
                 mlp_class=MLP):
        super().__init__()
        self.use_pos_embeddings = use_pos_embeddings
        self.global_pool = global_pool

        self.tower = nn.Sequential(*[block_class(embed_dim=embed_dim,
                                    dropout_level=dropout,
                                    mlp_multiplier=mlp_multiplier,
                                    is_causal=is_causal,
                                    mlp_class=mlp_class) for i in range(n_layers)])

        if use_pos_embeddings:
            #simple fixed learned positional encodings for now:
            self.pos_embed = nn.Embedding(seq_len, embed_dim)
            self.register_buffer('precomputed_pos_enc', torch.arange(0, seq_len).long())

    def forward(self, x):

        if self.use_pos_embeddings:
            pos_enc = self.precomputed_pos_enc[:x.size(1)].expand(x.size(0), -1)
            out = self.tower(x+self.pos_embed(pos_enc))
        else:
            out = self.tower(x)

        if self.global_pool:
            return torch.mean(out, dim=1)
        else:
            return out


class GPTmodel(nn.Module):
    def __init__(self, embed_dim, seq_len, n_layers, dropout, vocab_size):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.tower = Tower(embed_dim, 
                           seq_len, 
                           n_layers, 
                           use_pos_embeddings=True,
                           dropout=dropout,
                           n_heads=embed_dim//64,
                           mlp_multiplier=4, 
                           is_causal=True, 
                           global_pool=False, 
                           block_class=EncoderBlock, 
                           mlp_class=MLP)

        self.out_proj = nn.Sequential(nn.LayerNorm(embed_dim),
                                      nn.Linear(embed_dim, vocab_size))

    def forward(self, x):
        x = self.tower(self.embed(x))
        return self.out_proj(x)


def test_dims():
    b = Block(embed_dim=256, n_heads=2, dropout=0, mlp_multiplier=2, is_causal=False)
    t = Tower(embed_dim=256, n_heads=2, dropout=0, mlp_multiplier=2, n_layers=4,
            seq_len=64, use_pos_embeddings=True, global_pool=True)

    mha = MHAttention()
    multihead_attn = torch.nn.MultiheadAttention(256, 4, batch_first=True)
    q,k,v = torch.randn(32, 64, 256), torch.randn(32, 64, 256), torch.randn(32, 64, 256)


    assert mha(q,k,v).shape == multihead_attn(q,k,v).shape

    q,k,v = torch.randn(32, 32, 128), torch.randn(32, 64, 128), torch.randn(32, 64, 256)
    assert mha(q,k,v).shape == (32, 32, 256)


    x = torch.randn(32, 64, 256)

    assert b(x).shape == (32, 64, 256)
    assert t(x).shape == (32, 256)