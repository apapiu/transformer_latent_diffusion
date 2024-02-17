"""transformer based denoiser"""

import torch
from torch import nn
from einops.layers.torch import Rearrange

from tld.transformer_blocks import DecoderBlock, MLPSepConv, SinusoidalEmbedding


class DenoiserTransBlock(nn.Module):
    def __init__(self, patch_size, img_size, embed_dim, dropout, n_layers, mlp_multiplier=4, n_channels=4, scale_factor=1):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size ##size the model was trained on -> output is this * scale factor
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.mlp_multiplier = mlp_multiplier
        self.scale_factor = scale_factor

        seq_len = int((self.img_size/self.patch_size)*((self.img_size/self.patch_size)))
        lat_h = lat_w = int(self.img_size/self.patch_size)
        self.seq_len = seq_len
        patch_dim = self.n_channels*self.patch_size*self.patch_size


        self.pos_enc_down_sampling = nn.Sequential(Rearrange('bs (h w) d -> bs d h w', h=lat_h, w=lat_w),
                                                   nn.AvgPool2d(kernel_size=self.scale_factor),
                                                   Rearrange('bs d h w -> bs (h w) d'))
        
        self.pos_enc_upsampling = nn.Sequential(Rearrange('bs (h w) d -> bs d h w', h=lat_h, w=lat_w),
                                                   nn.Upsample(scale_factor=self.scale_factor, mode='bilinear'),
                                                   Rearrange('bs d h w -> bs (h w) d'))


        self.patchify_and_embed = nn.Sequential(
                                       nn.Conv2d(self.n_channels, patch_dim, kernel_size=self.patch_size, stride=self.patch_size),
                                       Rearrange('bs d h w -> bs (h w) d'),
                                       nn.LayerNorm(patch_dim),
                                       nn.Linear(patch_dim, self.embed_dim),
                                       nn.LayerNorm(self.embed_dim)
                                       )

        self.rearrange2 = Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                                   h=int(self.img_size/self.patch_size)*self.scale_factor,
                                   w=int(self.img_size/self.patch_size)*self.scale_factor,
                                   p1=self.patch_size, p2=self.patch_size)


        self.pos_embed = nn.Embedding(seq_len, self.embed_dim)
        self.register_buffer('precomputed_pos_enc', torch.arange(0, seq_len).long())

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
        x = self.patchify_and_embed(x)
        #pos_enc = self.precomputed_pos_enc[:x.size(1)].expand(x.size(0), -1)
        pos_enc = self.precomputed_pos_enc.expand(x.size(0), -1) 
        pos_enc = self.pos_embed(pos_enc) ##bs, 256, embed_dim 

        if self.seq_len > x.size(1):
            ##-> embed_dim, 16, 16 -> down/upsample:
            #downsample_size = self.seq_len//x.size(1)    
            pos_enc = self.pos_enc_down_sampling(pos_enc)
        elif self.seq_len < x.size(1):
            pos_enc = self.pos_enc_upsampling(pos_enc)
            
        x = x+pos_enc

        for block in self.decoder_blocks:
            x = block(x, cond)

        return self.out_proj(x)

class Denoiser(nn.Module):
    def __init__(self,
                 image_size, noise_embed_dims, patch_size, embed_dim, dropout, n_layers,
                 text_emb_size=768, scale_factor=1):
        super().__init__()

        self.image_size = image_size
        self.noise_embed_dims = noise_embed_dims
        self.embed_dim = embed_dim

        self.fourier_feats = nn.Sequential(SinusoidalEmbedding(embedding_dims=noise_embed_dims),
                                           nn.Linear(noise_embed_dims, self.embed_dim),
                                           nn.GELU(),
                                           nn.Linear(self.embed_dim, self.embed_dim)
                                           )

        self.denoiser_trans_block = DenoiserTransBlock(patch_size, image_size, embed_dim, dropout, n_layers, scale_factor=scale_factor)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.label_proj = nn.Linear(text_emb_size, self.embed_dim)

    def forward(self, x, noise_level, label):

        noise_level = self.fourier_feats(noise_level).unsqueeze(1)

        label = self.label_proj(label).unsqueeze(1)

        noise_label_emb = torch.cat([noise_level, label], dim=1) #bs, 2, d
        noise_label_emb = self.norm(noise_label_emb)

        x = self.denoiser_trans_block(x,noise_label_emb)

        return x
    
def test_outputs(num_imgs = 1):
    import time

    model = Denoiser(image_size=32, noise_embed_dims=128, patch_size=2, embed_dim=768, dropout=0.1, n_layers=12)
    x = torch.rand(num_imgs, 4, 32, 32)
    noise_level = torch.rand(num_imgs, 1)
    label = torch.rand(num_imgs, 768)

    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    with torch.no_grad():
        start_time = time.time()
        output = model(x, noise_level, label)
        end_time = time.time()

    execution_time = end_time - start_time  
    print(f"Model execution took {execution_time:.4f} seconds.")

    assert output.shape == torch.Size([num_imgs, 4, 32, 32])
    print("Basic tests passed.")

    model = Denoiser(image_size=16, noise_embed_dims=128, patch_size=2, embed_dim=256, dropout=0.1, n_layers=6, scale_factor=2)
    x = torch.rand(8, 4, 32, 32)
    noise_level = torch.rand(8, 1)
    label = torch.rand(8, 768)

    with torch.no_grad():
        output = model(x, noise_level, label)

    assert output.shape == torch.Size([8, 4, 32, 32])
    print("Uspscale tests passed.")