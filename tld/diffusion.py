import torch
from tqdm import tqdm

@torch.no_grad()
def diffusion(model,
              vae,
              device,
              model_dtype=torch.float32,
              n_iter=30,
              labels=None, #embeddings to condition on
              num_imgs=16,
              class_guidance=3,
              seed=10, #for reproducibility
              scale_factor=8, #latent scaling before decoding - should be ~ std of latent space
              img_size=32, #size of latent
              sharp_f = 0.1,
              bright_f = 0.1,
              exponent = 1, #this control the curve of the noise trajectory
              seeds = None #can input own random latents of size (n_imgs, 4, img_size, img_size)
              ):
    """function to generate images via reverese diffusion - includes decoding the latents."""
              
    noise_levels = (1 - torch.pow(torch.arange(0, 1, 1 / n_iter), exponent)).tolist()

    if seeds is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        new_img = torch.randn(num_imgs,4,img_size,img_size, dtype=model_dtype, device=device, generator=generator)
    else:
        new_img = seeds.to(model_dtype, device)

    labels = torch.cat([labels, torch.zeros_like(labels)])
    model.eval()

    for i in tqdm(range(len(noise_levels) - 1)):

        curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]

        noises = torch.full((2*num_imgs,1), curr_noise)

        #predict conditional and unconditional latents at the same time:
        x0_pred = model(torch.cat([new_img, new_img]),
                        noises.to(device, model_dtype),
                        labels.to(device, model_dtype)
                        )

        x0_pred_label, x0_pred_no_label = x0_pred[:num_imgs], x0_pred[num_imgs:]

        # classifier free guidance:
        x0_pred = class_guidance * x0_pred_label + (1 - class_guidance) * x0_pred_no_label

        # new image at next_noise level is a weighted average of old image and predicted x0:
        new_img = ((curr_noise - next_noise) * x0_pred + next_noise * new_img) / curr_noise

    # TODO: predict with model one more time to get x0
    x0_pred[:, 3, :, :] += sharp_f
    x0_pred[:, 0, :, :] += bright_f

    x0_pred_img = vae.decode((x0_pred*scale_factor).half())[0].cpu()
    return x0_pred_img, x0_pred

torch.manual_seed(1)
torch.randn(1)

torch.randn(32)
torch.randn(32)