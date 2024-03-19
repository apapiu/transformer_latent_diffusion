from dataclasses import dataclass
import torch

# = "https://huggingface.co/apapiu/small_ldt/resolve/main/state_dict_378000.pth"

@dataclass
class DenoiserConfig:
    image_size: int = 16
    noise_embed_dims: int = 256
    patch_size: int = 2
    embed_dim: int = 128
    dropout: float = 0
    n_layers: int = 3
    text_emb_size: int = 768
    dtype: torch.dtype = torch.float32
    file_url: str | None = None  #if not None will atempt to load state dict.
    local_filename: str | None = None

    #n_channels: int = 4 #not propoagated
    #mlp_multiplier: int = 4 #not propagated

@dataclass
class VaeConfig:
    vae_scale_factor: float = 8
    vae_name: str = "madebyollin/sdxl-vae-fp16-fix"
    vae_dtype: torch.dtype = torch.float32

@dataclass
class ClipConfig:
    clip_model_name: str = "ViT-L/14"
    vae_dtype: torch.dtype = torch.float16

@dataclass
class DataConfig:
    latent_path: str  
    text_emb_path: str
    val_path: str

@dataclass
class TrainConfig:
    batch_size: int = 128 #train..
    lr: float = 3e-4
    n_epoch: int = 100
    alpha: float = 0.999
    from_scratch: bool = True
    beta_a: float = 0.75
    beta_b: float = 0.75
    save_and_eval_every_iters: int = 1000
    run_id: str = ""
    model_name: str = ""


@dataclass
class LTDConfig:
    """used for diffusion generation"""
    denoiser_config: DenoiserConfig = DenoiserConfig()
    vae_cfg: VaeConfig = VaeConfig()
    clip_cfg: ClipConfig = ClipConfig()



@dataclass
class ModelConfig:
    data_config: DataConfig 
    denoiser_config: DenoiserConfig = DenoiserConfig()
    train_config: TrainConfig = TrainConfig()
    vae_cfg: VaeConfig = VaeConfig()
    clip_cfg: ClipConfig = ClipConfig()






if __name__=='__main__':
    cfg = ModelConfig(**{"beta_b":1000})
    print(cfg)
