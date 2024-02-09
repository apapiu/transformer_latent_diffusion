"""transformer based denoiser"""

import torch
from torch import nn
from einops.layers.torch import Rearrange

from tld.transformer_blocks import DecoderBlock, MLPSepConv, SinusoidalEmbedding


class DenoiserTransBlock(nn.Module):
    def __init__(self, patch_size, img_size, embed_dim, dropout, n_layers, mlp_multiplier=4, n_channels=4):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.mlp_multiplier = mlp_multiplier

        seq_len = img_size**2#int((self.img_size/self.patch_size)*((self.img_size/self.patch_size)))
        patch_dim = self.n_channels*self.patch_size*self.patch_size

        self.first_proj = nn.Sequential(Rearrange('bs d h w -> bs (h w) d'),
                                        nn.Linear(n_channels, self.embed_dim//4),
                                        nn.LayerNorm(self.embed_dim//4),
                                        )
        
        self.cond_proj = nn.Linear(self.embed_dim, self.embed_dim//4)

        self.patchify_and_embed = nn.Sequential(
                                       Rearrange('bs (h w) d ->bs d h w', h = self.img_size),
                                       nn.Conv2d(self.embed_dim//4, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size),
                                       Rearrange('bs d h w -> bs (h w) d'),
                                       nn.LayerNorm(self.embed_dim),
                                       nn.Linear(self.embed_dim, self.embed_dim),
                                       nn.LayerNorm(self.embed_dim)
                                       )

        self.rearrange2 = Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                                   h=int(self.img_size/self.patch_size),
                                   p1=self.patch_size, p2=self.patch_size)


        self.pos_embed = nn.Embedding(seq_len, self.embed_dim//4)
        self.register_buffer('precomputed_pos_enc', torch.arange(0, seq_len).long())

        self.first_decoder_block = DecoderBlock(embed_dim=self.embed_dim//4, ##aplied on priginal image
                                            mlp_multiplier=self.mlp_multiplier,
                                            is_causal=False,
                                            dropout_level=self.dropout,
                                            mlp_class=MLPSepConv)

        self.decoder_blocks = nn.ModuleList([DecoderBlock(embed_dim=self.embed_dim,
                                            mlp_multiplier=self.mlp_multiplier,
                                            #note that this is a non-causal block since we are 
                                            #denoising the entire image no need for masking
                                            is_causal=False,
                                            dropout_level=self.dropout,
                                            mlp_class=MLPSepConv)
                                              for _ in range(self.n_layers)])

        self.out_proj = nn.Sequential(nn.Linear(self.embed_dim, patch_dim),
                                                self.rearrange2)


    def forward(self, x, cond):
        x = self.first_proj(x) ### 4,h,w -> h*w, 4 -> h*w, d/4

        pos_enc = self.precomputed_pos_enc[:x.size(1)].expand(x.size(0), -1)
        x = x+self.pos_embed(pos_enc)

        ##have to project cond:
        cond_projected = self.cond_proj(cond)
        x = self.first_decoder_block(x, cond_projected) ###same dim as original data in and out.
        x = self.patchify_and_embed(x) ## h*w, d/4 -> d/4, h, w -> h/2*w/2, d

        for block in self.decoder_blocks:
            x = block(x, cond)

        ## should I add another attention layer here at 1024 tokens?

        #h/2*w/2, d -> d, h/2, w/2 -> upsample -> attenion block -> h*w, d -> h*w, 4 ->


        return self.out_proj(x)

class Denoiser(nn.Module):
    def __init__(self,
                 image_size, noise_embed_dims, patch_size, embed_dim, dropout, n_layers,
                 text_emb_size=768):
        super().__init__()

        self.image_size = image_size
        self.noise_embed_dims = noise_embed_dims
        self.embed_dim = embed_dim

        self.fourier_feats = nn.Sequential(SinusoidalEmbedding(embedding_dims=noise_embed_dims),
                                           nn.Linear(noise_embed_dims, self.embed_dim),
                                           nn.GELU(),
                                           nn.Linear(self.embed_dim, self.embed_dim)
                                           )

        self.denoiser_trans_block = DenoiserTransBlock(patch_size, image_size, embed_dim, dropout, n_layers)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.label_proj = nn.Linear(text_emb_size, self.embed_dim)

    def forward(self, x, noise_level, label):

        noise_level = self.fourier_feats(noise_level).unsqueeze(1)

        label = self.label_proj(label).unsqueeze(1)

        noise_label_emb = torch.cat([noise_level, label], dim=1) #bs, 2, d
        noise_label_emb = self.norm(noise_label_emb)

        x = self.denoiser_trans_block(x,noise_label_emb)

        return x
    
def test_outputs():
    model = Denoiser(image_size=16, noise_embed_dims=128, patch_size=2, embed_dim=256, dropout=0.1, n_layers=6)
    x = torch.rand(8, 4, 16, 16)
    noise_level = torch.rand(8, 1)
    label = torch.rand(8, 768)

    with torch.no_grad():
        output = model(x, noise_level, label)

    assert output.shape == torch.Size([8, 4, 16, 16])
    print("Basic tests passed.")