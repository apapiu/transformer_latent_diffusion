import os
import sys
from dataclasses import asdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from tld.denoiser import Denoiser
from tld.diffusion import DiffusionGenerator, DiffusionTransformer
from tld.configs import LTDConfig, DenoiserConfig, TrainConfig
from PIL.Image import Image

to_pil = transforms.ToPILImage()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

denoiser_cfg = DenoiserConfig(n_channels=4)
cfg = LTDConfig(denoiser_cfg=denoiser_cfg)


def test_denoiser_outputs(num_imgs=4):
    model = Denoiser(**asdict(denoiser_cfg))
    img_size = denoiser_cfg.image_size

    x = torch.rand(num_imgs, denoiser_cfg.n_channels, img_size, img_size)
    noise_level = torch.rand(num_imgs, 1)
    label = torch.rand(num_imgs, cfg.denoiser_cfg.text_emb_size)

    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    with torch.no_grad():
        start_time = time.time()
        output = model(x, noise_level, label)
        end_time = time.time()

    execution_time = end_time - start_time
    print(f"Model execution took {execution_time:.4f} seconds.")

    assert output.shape == torch.Size([num_imgs, denoiser_cfg.n_channels, img_size, img_size])
    print("Basic tests passed.")


def test_diffusion_generator():
    num_imgs = 1
    nrow = int(np.sqrt(num_imgs))
    model_dtype = cfg.denoiser_load.dtype

    denoiser = Denoiser(**asdict(denoiser_cfg))

    print(f"Model has {sum(p.numel() for p in denoiser.parameters())} parameters")

    denoiser.to(model_dtype)

    vae = AutoencoderKL.from_pretrained(
        cfg.vae_cfg.vae_name, torch_dtype=cfg.vae_cfg.vae_dtype
    ).to(device)

    labels = torch.rand(num_imgs, cfg.denoiser_cfg.text_emb_size)

    diffuser = DiffusionGenerator(denoiser, vae, device, model_dtype)

    out, _ = diffuser.generate(
        labels=labels,
        num_imgs=num_imgs,
        img_size=cfg.denoiser_cfg.image_size,
        class_guidance=3,
        seed=1,
        n_iter=5,
        exponent=1,
        scale_factor=8,
        sharp_f=0,
        bright_f=0,
    )

    out = to_pil(
        (vutils.make_grid((out + 1) / 2, nrow=nrow, padding=4)).float().clip(0, 1)
    )
    out.save("test.png")
    print("Images generated at test.png")


def test_full_generation_pipeline():
    diffusion_transformer = DiffusionTransformer(cfg)

    out = diffusion_transformer.generate_image_from_text(prompt="a cute cat")
    print(out)
    assert type(out) == Image


def test_training():
    from tld.train import main
    from tld.configs import ModelConfig, DataConfig
    import numpy as np

    data_config = DataConfig(
        latent_path="latents.npy", text_emb_path="text_emb.npy", val_path="val_emb.npy"
    )

    model_cfg = ModelConfig(
        data_config=data_config,
        train_config=TrainConfig(n_epoch=2, save_model=False, compile=False, use_wandb=False),
    )

    n = 200
    img_size = model_cfg.denoiser_config.image_size
    text_emb_size = model_cfg.denoiser_config.text_emb_size
    x = torch.randn(n, model_cfg.denoiser_config.n_channels, img_size, img_size).numpy()
    text_emb = torch.randn(n, text_emb_size).numpy()
    val_emb = torch.randn(8, text_emb_size).numpy()

    np.save("latents.npy", x)
    np.save("text_emb.npy", text_emb)
    np.save("val_emb.npy", val_emb)

    main(model_cfg)

# def test_data_processing():
#     from tld.data import main
#     from tld.configs import DataDownloadConfig

#     data_link = "https://huggingface.co/datasets/zzliang/GRIT/resolve/main/grit-20m/coyo_0_snappy.parquet?download=true"

#     data_config = DataDownloadConfig(
#         data_link=data_link,
#         latent_save_path="latent_folder",
#         raw_imgs_save_path="raw_imgs_folder",
#         download_data=True,
#         number_sample_per_shard=2,
#         batch_size=1,
#         first_n_rows=6,
#         use_wandb=False
#         )

#     if data_config.use_wandb:
#         os.environ["WANDB_API_KEY"] = "key"

#     main(data_config)

