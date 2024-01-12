import torch
from tqdm import tqdm

@torch.no_grad()
def diffusion(model,
              vae,
              device,
              model_dtype=torch.float32,
              n_iter=30,
              labels=None,
              num_imgs=64,
              class_guidance=3,
              seed=10,
              scale_factor=8,
              img_size=16,
              sharp_f = 0.1,
              bright_f = 0.1,
              exponent = 1,
              seeds = None
              ):

    noise_levels = 1 - torch.pow(torch.arange(0.0001, 0.99, 1 / n_iter), exponent)
    #noise_levels  = 1 - torch.arange(0, 1, 1 / n_iter)
    noise_levels = noise_levels.tolist()

    if seeds is None:
        torch.manual_seed(seed)
        seeds = torch.randn(num_imgs,4,img_size,img_size)
        seeds = seeds.to(device)
    else:
        seeds.to(device)

    new_img = seeds

    empty_labels = torch.zeros_like(labels)
    labels = torch.cat([labels, empty_labels])

    model.eval()

    for i in tqdm(range(len(noise_levels) - 1)):

        curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]

        noises = torch.full((num_imgs,1), curr_noise)
        noises = torch.cat([noises, noises])

        new_img = new_img.to(model_dtype)
        x0_pred = model(torch.cat([new_img, new_img]),
                        noises.to(device, model_dtype),
                        labels.to(device, model_dtype)
                        )

        x0_pred_label = x0_pred[:num_imgs]
        x0_pred_no_label = x0_pred[num_imgs:]

        # classifier free guidance:
        x0_pred = class_guidance * x0_pred_label + (1 - class_guidance) * x0_pred_no_label

        # new image at next_noise level is a weighted average of old image and predicted x0:
        new_img = ((curr_noise - next_noise) * x0_pred + next_noise * new_img) / curr_noise

    #TODO: predict with model one more time to get x0

    x0_pred[:, 3, :, :] += sharp_f
    x0_pred[:, 0, :, :] += bright_f

    x0_pred_img = vae.decode((x0_pred*scale_factor).half())[0].cpu()

    return x0_pred_img, x0_pred