#!/usr/bin/env python3
import os 
import sys
import copy
import numpy as np
from tqdm import tqdm

from dataclasses import dataclass


import wandb
import torchvision.utils as vutils

from accelerate import Accelerator
import torchvision

import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from safetensors.torch import load_model, save_model
from diffusers import AutoencoderKL

sys.path.append('transformer_latent_diffusion')
from tld.denoiser import Denoiser
from tld.diffusion import DiffusionGenerator

#!pip install torchview torchmetrics einops wandb diffusers datasets accelerate
#!git clone https://github.com/apapiu/transformer_latent_diffusion.git

def eval_gen(diffuser, labels):
    class_guidance=4.5
    seed=10
    out, out_latent = diffuser.generate(labels=torch.repeat_interleave(labels, 8, dim=0),
                                        num_imgs=64,
                                        class_guidance=class_guidance,
                                        seed=seed,
                                        n_iter=40,
                                        exponent=1,
                                        sharp_f=0.1,
                                        )

    out = to_pil((vutils.make_grid((out+1)/2, nrow=8, padding=4)).float().clip(0, 1))
    out.save(f'emb_val_cfg:{class_guidance}_seed:{seed}.png')

    return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_per_layer(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")

to_pil = torchvision.transforms.ToPILImage()

def update_ema(ema_model, model, alpha=0.999):
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(model_param.data, alpha=1-alpha)


@dataclass
class TrainConfig:
    embed_dim: int
    n_layers: int
    n_epoch: int

def main(config, data_path, val_path):
    ## see this for more: https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    
    accelerator = Accelerator(mixed_precision="fp16", log_with="wandb")

    ####DATA LOADING:
    latent_train_data = np.load(os.path.join(data_path, 'image_latents.npy'))
    train_label_embeddings = np.load(os.path.join(data_path, 'text_encodings.npy'))

    emb_val = torch.tensor(np.load(os.path.join(val_path, 'val_encs.npy')), dtype=torch.float32)

    train_label_embeddings = torch.tensor(train_label_embeddings, dtype=torch.float32)
    latent_train_data = torch.tensor(latent_train_data, dtype=torch.float32)
    dataset = TensorDataset(latent_train_data, train_label_embeddings)

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                    torch_dtype=torch.float16)
    
    if accelerator.is_local_main_process:
        vae = vae.to(accelerator.device)

    ###config: TODO: put all this in a dataclass:

    from_scratch = True
    run_id = '3ix3i6qp'
    model_name = 'curr_state_dict.pth/model.safetensors'

    embed_dim = config.embed_dim
    n_layers = config.n_layers

    clip_embed_size = 768
    scaling_factor = 8
    patch_size = 2
    image_size = img_size = 32
    n_channels = 4
    dropout = 0
    mlp_multiplier = 4

    batch_size = 128
    class_guidance = 3
    lr=3e-4

    alpha = 0.999

    noise_embed_dims = 128
    diffusion_n_iter = 35

    n_epoch = config.n_epoch

    #end config:


    ###model definition

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    accelerator.print("loading model")

    model = Denoiser(image_size=image_size, noise_embed_dims=noise_embed_dims,
                 patch_size=patch_size, embed_dim=embed_dim, dropout=dropout,
                 n_layers=n_layers)

    if not from_scratch:
        wandb.restore(model_name, run_path=f"apapiu/cifar_diffusion/runs/{run_id}",
                      replace=True)
        load_model(model, model_name)

    accelerator.print("model loaded")

    if accelerator.is_local_main_process:
        ema_model = copy.deepcopy(model).to(accelerator.device)
        diffuser = DiffusionGenerator(ema_model, vae,  accelerator.device, torch.float32)

    config = {k: v for k, v in locals().items() if k in ['embed_dim', 'n_layers', 'clip_embed_size', 'scaling_factor',
                                                         'image_size', 'noise_embed_dims', 'dropout',
                                                         'mlp_multiplier', 'diffusion_n_iter', 'batch_size', 'lr',
                                                         'patch_size']}


    #opt stuff:
    global_step = 0
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    accelerator.print("model prep")
    model, train_loader, optimizer = accelerator.prepare(
        model, train_loader, optimizer
    )

    accelerator.init_trackers(
    project_name="cifar_diffusion",
    config=config
    )

    accelerator.print(count_parameters(model))
    accelerator.print(count_parameters_per_layer(model))

    ### and train:

    for i in range(1, n_epoch+1):
        accelerator.print(f'epoch: {i}')            

        for x, y in tqdm(train_loader):
            x = x/scaling_factor

            noise_level = torch.tensor(np.random.beta(1, 2.7, len(x)), device=accelerator.device)
            signal_level = 1 - noise_level
            noise = torch.randn_like(x)

            x_noisy = noise_level.view(-1,1,1,1)*noise + signal_level.view(-1,1,1,1)*x

            x_noisy = x_noisy.float()
            noise_level = noise_level.float()
            label = y

            prob = 0.15
            mask = torch.rand(y.size(0), device=accelerator.device) < prob
            label[mask] = 0 # OR replacement_vector

            if global_step % 500 == 0 and accelerator.is_local_main_process:
                out = eval_gen(diffuser=diffuser, labels=emb_val)
                out.save('img.jpg')
                accelerator.log({f"step: {global_step}": wandb.Image("img.jpg")})

                ###todo: add dict here to save the opt state:
                state_dict_path = 'curr_state_dict.pth'
                accelerator.save_model(ema_model, state_dict_path)
                wandb.save('curr_state_dict.pth/model.safetensors')

            model.train()

            ###train loop:
            optimizer.zero_grad()

            pred = model(x_noisy, noise_level.view(-1,1), label)
            loss = loss_fn(pred, x)
            accelerator.log({"train_loss":loss.item()}, step=global_step)
            accelerator.backward(loss)
            optimizer.step()

            if accelerator.is_local_main_process:
                update_ema(ema_model, model, alpha=alpha)

            global_step += 1
    accelerator.end_training()
            
# args = (vae, "fp16")
# notebook_launcher(training_loop, num_processes=1)