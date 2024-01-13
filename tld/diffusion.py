import torch
from tqdm import tqdm

class DiffusionGenerator:
    def __init__(self, model, vae, device, model_dtype=torch.float32):
        self.model = model
        self.vae = vae
        self.device = device
        self.model_dtype = model_dtype


    @torch.no_grad()
    def generate(self, 
                 n_iter=30, 
                 labels=None, #embeddings to condition on
                 num_imgs=16, 
                 class_guidance=3,
                 seed=10,  #for reproducibility
                 scale_factor=8, #latent scaling before decoding - should be ~ std of latent space
                 img_size=32, #height, width of latent
                 sharp_f=0.1, 
                 bright_f=0.1, 
                 exponent=1,
                 seeds=None):
        """Generate images via reverse diffusion."""
        noise_levels = (1 - torch.pow(torch.arange(0, 1, 1 / n_iter), exponent)).tolist()
        new_img = self.initialize_image(seeds, num_imgs, img_size, seed)

        labels = torch.cat([labels, torch.zeros_like(labels)])
        self.model.eval()

        for i in tqdm(range(len(noise_levels) - 1)):
            curr_noise, next_noise = noise_levels[i], noise_levels[i + 1]
            noises = torch.full((2*num_imgs, 1), curr_noise)

            x0_pred = self.model(torch.cat([new_img, new_img]),
                                 noises.to(self.device, self.model_dtype),
                                 labels.to(self.device, self.model_dtype))

            x0_pred = self.apply_classifier_free_guidance(x0_pred, num_imgs, class_guidance)

            new_img = ((curr_noise - next_noise) * x0_pred + next_noise * new_img) / curr_noise

        x0_pred[:, 3, :, :] += sharp_f
        x0_pred[:, 0, :, :] += bright_f

        x0_pred_img = self.vae.decode((x0_pred*scale_factor).half())[0].cpu()
        return x0_pred_img, x0_pred

    def initialize_image(self, seeds, num_imgs, img_size, seed):
        """Initialize the seed tensor."""
        if seeds is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            return torch.randn(num_imgs, 4, img_size, img_size, dtype=self.model_dtype, 
                               device=self.device, generator=generator)
        else:
            return seeds.to(self.model_dtype, self.device)

    def apply_classifier_free_guidance(self, x0_pred, num_imgs, class_guidance):
        """Apply classifier-free guidance to the predictions."""
        x0_pred_label, x0_pred_no_label = x0_pred[:num_imgs], x0_pred[num_imgs:]
        return class_guidance * x0_pred_label + (1 - class_guidance) * x0_pred_no_label