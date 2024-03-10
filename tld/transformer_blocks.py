import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class SinusoidalEmbedding(nn.Module):
    def __init__(self, emb_min_freq=1.0, emb_max_freq=1000.0, embedding_dims=32):
        super(SinusoidalEmbedding, self).__init__()

        frequencies = torch.exp(
            torch.linspace(np.log(emb_min_freq), np.log(emb_max_freq), embedding_dims // 2)
        )

        self.register_buffer("angular_speeds", 2.0 * torch.pi * frequencies)

    def forward(self, x):
        embeddings = torch.cat(
            [torch.sin(self.angular_speeds * x), torch.cos(self.angular_speeds * x)], dim=-1
        )
        return embeddings


class MHAttention(nn.Module):
    def __init__(self, is_causal=False, dropout_level=0.0, n_heads=4):
        super().__init__()
        self.is_causal = is_causal
        self.dropout_level = dropout_level
        self.n_heads = n_heads

    def forward(self, q, k, v, attn_mask=None):
        assert q.size(-1) == k.size(-1)
        assert k.size(-2) == v.size(-2)

        q, k, v = [rearrange(x, "bs n (d h) -> bs h n d", h=self.n_heads) for x in [q, k, v]]

        out = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=self.is_causal,
            dropout_p=self.dropout_level if self.training else 0,
        )

        out = rearrange(out, "bs h n d -> bs n (d h)", h=self.n_heads)

        return out


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, is_causal=False, dropout_level=0.0, n_heads=4):
        super().__init__()
        self.qkv_linear = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)

    def forward(self, x):
        q, k, v = self.qkv_linear(x).chunk(3, dim=2)
        return self.mha(q, k, v)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, is_causal=False, dropout_level=0, n_heads=4):
        super().__init__()
        self.kv_linear = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mha = MHAttention(is_causal, dropout_level, n_heads)

    def forward(self, x, y):
        q = self.q_linear(x)
        k, v = self.kv_linear(y).chunk(2, dim=2)
        return self.mha(q, k, v)


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_multiplier * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout_level),
        )

    def forward(self, x):
        return self.mlp(x)


class MLPSepConv(nn.Module):
    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        """see: https://github.com/ofsoundof/LocalViT"""
        super().__init__()
        self.mlp = nn.Sequential(
            # this Conv with kernel size 1 is equivalent to the Linear layer in a "regular" transformer MLP
            nn.Conv2d(embed_dim, mlp_multiplier * embed_dim, kernel_size=1, padding="same"),
            nn.Conv2d(
                mlp_multiplier * embed_dim,
                mlp_multiplier * embed_dim,
                kernel_size=3,
                padding="same",
                groups=mlp_multiplier * embed_dim,
            ),  # <- depthwise conv
            nn.GELU(),
            nn.Conv2d(mlp_multiplier * embed_dim, embed_dim, kernel_size=1, padding="same"),
            nn.Dropout(dropout_level),
        )

    def forward(self, x):
        w = h = int(np.sqrt(x.size(1)))  # only square images for now
        x = rearrange(x, "bs (h w) d -> bs d h w", h=h, w=w)
        x = self.mlp(x)
        x = rearrange(x, "bs d h w -> bs (h w) d")
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        is_causal: bool,
        mlp_multiplier: int,
        dropout_level: float,
        mlp_class: type[MLP] | type[MLPSepConv],
    ):
        super().__init__()
        self.self_attention = SelfAttention(embed_dim, is_causal, dropout_level, n_heads=embed_dim // 64)
        self.cross_attention = CrossAttention(
            embed_dim, is_causal=False, dropout_level=0, n_heads=4
        )
        self.mlp = mlp_class(embed_dim, mlp_multiplier, dropout_level)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, y):
        x = self.norm1(self.self_attention(x) + x)
        x = self.norm2(self.cross_attention(x, y) + x)
        x = self.norm3(self.mlp(x) + x)
        return x
