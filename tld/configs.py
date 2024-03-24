from dataclasses import dataclass, field
import torch

@dataclass
class DenoiserConfig:
    image_size: int = 16
    noise_embed_dims: int = 256
    patch_size: int = 2
    embed_dim: int = 128
    dropout: float = 0
    n_layers: int = 3
    text_emb_size: int = 768
    n_channels: int = 4 #not propoagated
    mlp_multiplier: int = 4 #not propagated

@dataclass
class DenoiserLoad:
    dtype: torch.dtype = torch.float32
    file_url: str | None = None
    local_filename: str | None = None

@dataclass
class VaeConfig:
    vae_scale_factor: float = 8
    vae_name: str = "madebyollin/sdxl-vae-fp16-fix"
    vae_dtype: torch.dtype = torch.float32

@dataclass
class ClipConfig:
    clip_model_name: str = "ViT-L/14"
    clip_dtype: torch.dtype = torch.float16


@dataclass
class LTDConfig:
    """main config for inference"""
    denoiser_cfg: DenoiserConfig = field(default_factory=DenoiserConfig)
    denoiser_load: DenoiserLoad = field(default_factory=DenoiserLoad)
    vae_cfg: VaeConfig = field(default_factory=VaeConfig)
    clip_cfg: ClipConfig = field(default_factory=ClipConfig)

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
    compile: bool = True
    save_model: bool = True
    use_wandb: bool = True


@dataclass
class ModelConfig:
    """main config for training and inference"""
    data_config: DataConfig 
    denoiser_config: DenoiserConfig = field(default_factory=DenoiserConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    vae_cfg: VaeConfig = field(default_factory=VaeConfig)
    clip_cfg: ClipConfig = field(default_factory=ClipConfig)



if __name__=='__main__':
    cfg = LTDConfig()
    print(cfg)
