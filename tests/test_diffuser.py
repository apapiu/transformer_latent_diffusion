import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from diffusers import AutoencoderKL

from tld.denoiser import Denoiser
from tld.diffusion import DiffusionGenerator, DiffusionTransformer, LTDConfig
from PIL.Image import Image

to_pil = transforms.ToPILImage()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_outputs(num_imgs=4):
    model = Denoiser(
        image_size=32, noise_embed_dims=128, patch_size=2, embed_dim=768, dropout=0.1, n_layers=12
    )
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

    # model = Denoiser(image_size=16, noise_embed_dims=128, patch_size=2, embed_dim=256, dropout=0.1, n_layers=6)
    # x = torch.rand(8, 4, 32, 32)
    # noise_level = torch.rand(8, 1)
    # label = torch.rand(8, 768)

    # with torch.no_grad():
    #     output = model(x, noise_level, label)

    # assert output.shape == torch.Size([8, 4, 32, 32])
    # print("Uspscale tests passed.")


def test_diffusion_generator():
    model_dtype = torch.float32  ##float 16 will not work on cpu
    num_imgs = 1
    nrow = int(np.sqrt(num_imgs))

    denoiser = Denoiser(
        image_size=32, noise_embed_dims=128, patch_size=2, embed_dim=256, dropout=0.1, n_layers=3
    )
    print(f"Model has {sum(p.numel() for p in denoiser.parameters())} parameters")

    denoiser.to(model_dtype)

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=model_dtype).to(device)

    labels = torch.rand(num_imgs, 768)

    diffuser = DiffusionGenerator(denoiser, vae, device, model_dtype)

    out, _ = diffuser.generate(
        labels=labels,
        num_imgs=num_imgs,
        class_guidance=3,
        seed=1,
        n_iter=5,
        exponent=1,
        scale_factor=8,
        sharp_f=0,
        bright_f=0,
    )

    out = to_pil((vutils.make_grid((out + 1) / 2, nrow=nrow, padding=4)).float().clip(0, 1))
    out.save("test.png")
    print("Images generated at test.png")


def test_full_generation_pipeline():
    ltdconfig = LTDConfig()
    diffusion_transformer = DiffusionTransformer(ltdconfig)

    out = diffusion_transformer.generate_image_from_text(prompt="a cute cat")
    print(out)
    assert type(out) == Image


# TODO: should add tests for train loop and data processing
