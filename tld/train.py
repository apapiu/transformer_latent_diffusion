#!/usr/bin/env python3

import copy
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict

import wandb
import torchvision.utils as vutils

from accelerate import Accelerator
import torchvision

import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from diffusers import AutoencoderKL

from tld.denoiser import Denoiser
from tld.diffusion import DiffusionGenerator


def eval_gen(diffuser, labels, masked_latents):
    class_guidance=4.5
    seed=10
    out, _ = diffuser.generate(labels=torch.repeat_interleave(labels, 8, dim=0),
                                        num_imgs=64,
                                        class_guidance=class_guidance,
                                        seed=seed,
                                        n_iter=30,
                                        exponent=1,
                                        sharp_f=0.1,
                                        use_ddpm_plus=True,
                                        masked_latents=masked_latents
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
class ModelConfig:
    embed_dim: int = 512
    n_layers: int = 6
    clip_embed_size: int = 768
    scaling_factor: int = 8
    patch_size: int = 2
    image_size: int = 32 
    n_channels: int = 4+4
    dropout: float = 0
    mlp_multiplier: int = 4
    batch_size: int = 128
    class_guidance: int = 3
    lr: float = 3e-4
    n_epoch: int = 100
    alpha: float = 0.999
    noise_embed_dims: int = 128
    diffusion_n_iter: int = 35
    from_scratch: bool = True
    run_id: str = None
    model_name: str = None
    beta_a: float = 0.75
    beta_b: float = 0.75
    save_and_eval_every_iters: int = 1000

@dataclass
class DataConfig:
    latent_path: str #path to a numpy file containing latents
    text_emb_path: str
    val_path: str

def main(config: ModelConfig, dataconfig: DataConfig):
    """main train loop to be used with accelerate"""

    accelerator = Accelerator(mixed_precision="fp16", log_with="wandb")

    accelerator.print("Loading Data:")
    latent_train_data = torch.tensor(np.load(dataconfig.latent_path), dtype=torch.float32)

    masked_latents = latent_train_data[:64].clone()
    masked_latents[:, :, :, 16:] = 0

    train_label_embeddings = torch.tensor(np.load(dataconfig.text_emb_path), dtype=torch.float32)
    emb_val = torch.tensor(np.load(dataconfig.val_path), dtype=torch.float32)
    dataset = TensorDataset(latent_train_data, train_label_embeddings)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix",
                                    torch_dtype=torch.float16)
    
    if accelerator.is_main_process:
        vae = vae.to(accelerator.device)
   
    model = Denoiser(image_size=config.image_size, noise_embed_dims=config.noise_embed_dims,
                 patch_size=config.patch_size, embed_dim=config.embed_dim, dropout=config.dropout,
                 n_layers=config.n_layers, n_channels=config.n_channels)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if not config.from_scratch:
        accelerator.print("Loading Model:")
        wandb.restore(config.model_name, run_path=f"apapiu/cifar_diffusion/runs/{config.run_id}",
                      replace=True)
        full_state_dict = torch.load(config.model_name)

        state_dict = full_state_dict['model_ema']
        #initialize weights with zero - only first time we fine-tune:
        print("initalizing zero weight in conv patch:")
        new_weights = torch.zeros((16, 8, 2, 2))
        new_weights[:, :4, :, :] = state_dict['denoiser_trans_block.patchify_and_embed.0.weight']
        state_dict['denoiser_trans_block.patchify_and_embed.0.weight'] = new_weights

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(full_state_dict['opt_state'])
        global_step = full_state_dict['global_step']
    else:
        global_step = 0

    if accelerator.is_local_main_process:
        ema_model = copy.deepcopy(model).to(accelerator.device)
        diffuser = DiffusionGenerator(ema_model, vae,  accelerator.device, torch.float32)

    accelerator.print("model prep")
    model, train_loader, optimizer = accelerator.prepare(
        model, train_loader, optimizer
    )

    accelerator.init_trackers(
    project_name="cifar_diffusion",
    config=asdict(config)
    )

    accelerator.print(count_parameters(model))
    accelerator.print(count_parameters_per_layer(model))

    ### Train:
    for i in range(1, config.n_epoch+1):
        accelerator.print(f'epoch: {i}')            

        for x, y in tqdm(train_loader):
            x = x/config.scaling_factor

            noise_level = torch.tensor(np.random.beta(config.beta_a, config.beta_b, len(x)),
                                        device=accelerator.device)
            signal_level = 1 - noise_level
            noise = torch.randn_like(x)

            x_noisy = noise_level.view(-1,1,1,1)*noise + signal_level.view(-1,1,1,1)*x

            x_noisy = x_noisy.float()

            #the input is the noisy latent and the unnoised first half of latent:
            first_half = x.clone()
            first_half[:, :, :, 16:] = 0
            
            x_noisy = torch.cat([x_noisy, first_half], dim=1) #(bs, 4+4, h, w)

            noise_level = noise_level.float()
            label = y

            prob = 0.15
            mask = torch.rand(y.size(0), device=accelerator.device) < prob
            label[mask] = 0 # OR replacement_vector

            if global_step % config.save_and_eval_every_iters == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ##eval and saving:
                    out = eval_gen(diffuser=diffuser, labels=emb_val, masked_latents=masked_latents)
                    out.save('img.jpg')
                    accelerator.log({f"step: {global_step}": wandb.Image("img.jpg")})
                    
                    opt_unwrapped = accelerator.unwrap_model(optimizer)
                    full_state_dict = {'model_ema':ema_model.state_dict(),
                                    'opt_state':opt_unwrapped.state_dict(),
                                    'global_step':global_step
                                    }
                    accelerator.save(full_state_dict, config.model_name)
                    wandb.save(config.model_name)

            model.train()

            with accelerator.accumulate():
                ###train loop:
                optimizer.zero_grad()

                pred = model(x_noisy, noise_level.view(-1,1), label)
                loss = loss_fn(pred, x)
                accelerator.log({"train_loss":loss.item()}, step=global_step)
                accelerator.backward(loss)
                optimizer.step()

                if accelerator.is_main_process:
                    update_ema(ema_model, model, alpha=config.alpha)

            global_step += 1
    accelerator.end_training()
            
# args = (config, data_path, val_path)
# notebook_launcher(training_loop)