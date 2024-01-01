from denoiser import Denoiser
from diffusion import diffusion


from accelerate import Accelerator, notebook_launcher
import wandb
import torchvision.utils as vutils
from safetensors.torch import load_model, save_model
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch 



#config:
def training_loop(vae, mixed_precision, emb_val):
    ## see this for more: https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    ## can't get some stuff to work in multi-gpu :(  diffusion does not work, EMA does not work, full dataset doesn't work
    accelerator = Accelerator(mixed_precision=mixed_precision, log_with="wandb")
    
    if accelerator.is_local_main_process:
        vae = vae.to(accelerator.device)
    
    from_scratch = False
    run_id = '3skree8h'
    model_name = 'curr_state_dict.pth/model.safetensors'

    embed_dim = 256
    n_layers = 4

    clip_embed_size = 768
    scaling_factor = 8
    patch_size = 1
    image_size = img_size = 16
    n_channels = 4
    dropout = 0
    mlp_multiplier = 4

    batch_size = 256
    class_guidance = 3
    lr=3e-4

    alpha = 0.999

    noise_embed_dims = 128
    diffusion_n_iter = 35

    n_epoch = 10

    #end config:

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

    config = {k: v for k, v in locals().items() if k in ['embed_dim', 'n_layers', 'clip_embed_size', 'scaling_factor',
                                                         'image_size', 'noise_embed_dims', 'dropout',
                                                         'mlp_multiplier', 'diffusion_n_iter', 'batch_size', 'lr']}

    
    #opt stuff:
    global_step = 0
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
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

    for i in range(1, n_epoch+1):
        accelerator.print(f'epoch: {i}')
        
        if accelerator.is_local_main_process:
            state_dict_path = 'curr_state_dict.pth'
            accelerator.save_model(ema_model, state_dict_path)
            wandb.save('curr_state_dict.pth/model.safetensors')

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
                out, out_latent = diffusion(
                                        model=ema_model,
                                        vae=vae,
                                        device=accelerator.device,
                                        labels=torch.repeat_interleave(emb_val, 8, dim=0),
                                        num_imgs=64, n_iter=35,
                                        class_guidance=3,
                                        scale_factor=scaling_factor,
                                        dyn_thresh=True)
                to_pil((vutils.make_grid((out.float()+1)/2, nrow=8)).clip(0, 1)).save('img.jpg')
                wandb.log({f"step: {global_step}": wandb.Image("img.jpg")})

            model.train()
    
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
# notebook_launcher(training_loop, args, num_processes=1)
    

##utils:
import torchvision
from torch import nn

    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_per_layer(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")

to_pil = torchvision.transforms.ToPILImage()

from torch.optim.lr_scheduler import _LRScheduler

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_warmup_steps, initial_lr, final_lr, last_epoch=-1):
        self.total_warmup_steps = total_warmup_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.total_warmup_steps:
            # Calculate the learning rate based on the current epoch
            lr = self.initial_lr + (self.final_lr - self.initial_lr) * (self.last_epoch / self.total_warmup_steps)
            return [lr for _ in self.base_lrs]
        else:
            # After warm-up, continue with the base learning rate
            return self.base_lrs
        

class CustomDataset(Dataset):
    def __init__(self, latent_data, label_embeddings1, label_embeddings2):
        self.latent_data = latent_data
        self.label_embeddings1 = label_embeddings1
        self.label_embeddings2 = label_embeddings2

    def __len__(self):
        return len(self.latent_data)

    def __getitem__(self, idx):
        x = self.latent_data[idx]
        if random.random() < 0.5:
            y = self.label_embeddings1[idx]
        else:
            y = self.label_embeddings2[idx]
        return x, y

def update_ema(ema_model, model, alpha=0.999):
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(model_param.data, alpha=1-alpha)


def set_dropout_to_zero(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
            print(module)

    for module in model.modules():
        if isinstance(module, MHAttention):
            #module.p = 0.0
            module.dropout_level = 0
            print(module.dropout_level)