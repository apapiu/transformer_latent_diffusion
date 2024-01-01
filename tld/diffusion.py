import torch
from tqdm import tqdm

@torch.no_grad()
def diffusion(model,
              vae,
              device,
              n_iter=30,
              labels=None,
              num_imgs=64,
              class_guidance=3,
              seed=10,
              scale_factor=8,
              dyn_thresh=False,
              img_size=16,
              ):
    
    """diffusion in the latent space with vae decoding"""

    noise_levels = 1 - torch.pow(torch.arange(0.0001, 0.99, 1 / n_iter), 1 / 3)
    noise_levels = noise_levels.tolist()

    torch.manual_seed(seed)
    seeds = torch.randn(num_imgs,4,img_size,img_size).to(device)
    new_img = seeds

    empty_labels = torch.zeros_like(labels)
    labels = torch.cat([labels, empty_labels])

    model.eval()


    for i in tqdm(range(len(noise_levels) - 1)):

        curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]

        noises = torch.full((num_imgs,1), curr_noise)
        noises = torch.cat([noises, noises])


        x0_pred = model(torch.cat([new_img, new_img]),
                        noises.to(device),
                        labels.to(device)
                        )

        x0_pred_label = x0_pred[:num_imgs]
        x0_pred_no_label = x0_pred[num_imgs:]

        # classifier free guidance:
        x0_pred = class_guidance * x0_pred_label + (1 - class_guidance) * x0_pred_no_label

        # new image at next_noise level is a weighted average of old image and predicted x0:
        new_img = ((curr_noise - next_noise) * x0_pred + next_noise * new_img) / curr_noise
        #new_img = (np.sqrt(1 - next_noise**2)) * x0_pred + next_noise * (new_img - np.sqrt(1 - curr_noise**2)* x0_pred)/ curr_noise

        if dyn_thresh:
            s = x0_pred.abs().float().quantile(0.99)
            x0_pred = x0_pred.clip(-s, s)/(s/2) #rescale to -2,2

    #should predict one more time here

    x0_pred_img = vae.decode((x0_pred*scale_factor).half())[0].cpu()

    return x0_pred_img, x0_pred