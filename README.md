# Transformer Latent Diffusion
Text to Image Latent Diffusion using a Transformer core in PyTorch.

**Try with own inputs**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VaCe01YG9rnPwAfwVLBKdXEX7D_tk1U5?usp=sharing)

Below are some random examples (at 256 resolution) from a 100MM model trained from scratch for 260k iterations (about 32 hours on 1 A100):

<img width="760" alt="image" src="https://github.com/apapiu/transformer_latent_diffusion/assets/13619417/e01e3094-2487-4c04-bc0f-d9b03eeaed00">

#### Clip interpolation Examples:

a photo of a cat → an anime drawing of a super saiyan cat, artstation:

<img width="1361" alt="image" src="https://github.com/apapiu/transformer_latent_diffusion/assets/13619417/a079458b-9bd5-4557-aa7a-5a3e78f31b53">

a cute great gray owl → starry night by van gogh:

<img width="1399" alt="image" src="https://github.com/apapiu/transformer_latent_diffusion/assets/13619417/8731d87a-89fa-43a2-847d-c7ff772de286">

Note that the model has not converged yet and could use more training. 

#### High(er) Resolution: 
By upsampling the positional encoding the model can also generate 512 or 1024 px images with minimal fine-tuning. See below for some examples of model fine-tuned on 100k extra 512 px images and 30k 1024 px images for about 2 hours on an A100. The images do sometimes lack global coherence at 1024 px - more to come here:

<img width="600" alt="image" src="https://github.com/apapiu/transformer_latent_diffusion/assets/13619417/adba64f0-b43c-423e-9a7d-033a4afea207">
<img width="600" alt="image" src="https://github.com/apapiu/transformer_latent_diffusion/assets/13619417/5a94515b-313e-420d-89d4-6bdc376d9a00">



### Intro: 

The main goal of this project is to build an accessible diffusion model in PyTorch that is: 
- fast (close to real time generation)
- small (~100MM params)
- reasonably good (of course not SOTA)
- can be trained in a reasonable amount of time on a single GPU (under 50 hours on an A100 or equivalent).
- simple self-contained codebase (model + train loop is about ~400 lines of PyTorch with little dependencies)
- uses ~ 1 million images with a focus on data quality over quantity

This is part II of a previous [project](https://github.com/apapiu/guided-diffusion-keras) I did where I trained a pixel level diffusion model in Keras. Even though this model outputs 4x higher resolution images (256px vs 64px), it's actually faster to both train and sample from, which shows the power of training in the latent space and speed of transformer architectures.

## Table of Contents:
- [Codebase](#codebase)
- [Usage](#usage)
- [Examples](#examples)
- [Data Processing](#data-processing)
- [Architecture](#architecture)
- [TO-DOs](#todos)


## Codebase:
The code is written in pure PyTorch with as few dependencies as possible.

- [transformer_blocks.py](https://github.com/apapiu/transformer_latent_diffusion/blob/main/tld/transformer_blocks.py) - basic transformer building blocks relevant to the transformer denoiser
- [denoiser.py](https://github.com/apapiu/transformer_latent_diffusion/blob/main/tld/denoiser.py) - the architecture of the denoiser transformer
- [train.py](https://github.com/apapiu/transformer_latent_diffusion/blob/main/tld/train.py). The train loop uses `accelerate` so its training can scale to multiple GPUs if needed.
- [diffusion.py](https://github.com/apapiu/transformer_latent_diffusion/blob/main/tld/diffusion.py). Class to generate image from noise using reverse diffusion. Short (~60 lines) and self-contained.
- [data.py](https://github.com/apapiu/transformer_latent_diffusion/blob/main/tld/data.py). Data utils to download images/text and process necessary features for the diffusion model.

### Usage:
If you have your own dataset of URLs + captions, the process to train a model on the data consists of two steps:

1. Use `train.download_and_process_data` to obtain the latent and text encodings as numpy files. See [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BPDFDBdsP9SSKBNEFJysmlBjfoxKK13r?usp=sharing) for a notebook example downloading and processing 2000 images from this HuggingFace [dataset](https://huggingface.co/datasets/zzliang/GRIT).

2. use the `train.main` function in an accelerate `notebook_launcher` - see [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sKk0usxEF4bmdCDcNQJQNMt4l9qBOeAM?usp=sharing) for a colab notebook that trains a model on 100k images from scratch. Note that this downloads already pre-preprocessed latents and embeddings from [here](https://huggingface.co/apapiu/small_ldt/tree/main) but you could just use whatever `.npy` files you had saved from step 1.

#### Fine-Tuning - TODO but it is the same as step 2 above except you train on a pre-trained model.

```python
!wandb login
import os
from tld.train import main, DataConfig, ModelConfig
from accelerate import notebook_launcher

data_config = DataConfig(latent_path='path/to/image_latents.npy',
                         text_emb_path='path/to/text_encodings.npy',
                         val_path='path/to/val_encodings.npy')

model_config = ModelConfig(embed_dim=512, n_layers=6) #see ModelConfig for more params

#run the training process on 2 GPUs:
notebook_launcher(main, (model_config, data_config), num_processes=2)
```

### Dependencies:
- `PyTorch` `numpy` `einops` for model building
- `wandb` `tqdm` for logging + progress bars
- `accelerate` for train loop and multi-GPU support
- `img2dataset` `webdataset` `torchvision` for data downloading and image processing
- `diffusers` `clip` for pretrained VAE and CLIP text model

### Codebases used for inspiration:
- [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha)
- [k-diffusion](https://github.com/crowsonkb/k-diffusion)
- [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master)
- [LocalViT](https://github.com/ofsoundof/LocalViT)

#### Speed:

I try to speed up training and inference as much as possible by:
- using mixed precision for training + [sdpa]
- precompute all latent and text embeddings
- using float16 precision for inference
- using [sdpa] for the attention natively + torch.compile() (compile doesn't always work).
- use a highly performant sampler (DPM-Solver++(2M)) that gets good results in ~ 15 steps.
- TODO: would distillation or something like LCM work here?
- TODO: use flash-attention2?
- TODO: use smaller vae?

The time to generate a batch of 36 images (15 iterations) on a: 
- T4: ~ 3.5 seconds
- A100: ~ 0.6 seconds
In fact on an A100 the vae becomes the bottleneck even though it is only used once.


## Examples:

More examples generated with the 100MM model - click the photo to see the prompt and other params like cfg and seed:
![image](tld/img_examples/a%20cute%20grey%20great%20owl_cfg_8_seed_11.png)
![image](tld/img_examples/watercolor%20of%20a%20cute%20cat%20riding%20a%20motorcycle_cfg_7_seed_11.png)
![image](tld/img_examples/painting%20of%20a%20cyberpunk%20market_cfg_7_seed_11.png)
![image](tld/img_examples/isometric%20view%20of%20small%20japanese%20village%20with%20blooming%20trees_cfg_7_seed_11.png)
![image](tld/img_examples/a%20beautiful%20woman%20with%20blonde%20hair%20in%20her%2050s_cfg_7_seed_11.png)
![image](tld/img_examples/painting%20of%20a%20cute%20fox%20in%20a%20suit%20in%20a%20field%20of%20poppies_cfg_8_seed_11.png)
![image](tld/img_examples/an%20aerial%20view%20of%20manhattan%2C%20isometric%20view%2C%20as%20pantinted%20by%20mondrian_cfg_7_seed_11.png)

## Outpainting model:

I also fine-tuned an outpaing model on top of the original 101MM model. I had to modify the original input conv2d patch to 8 channel and initialize the mask channels parameters to zero. The rest of the architecture remained the same.

Below I apply the outpainting model repatedly to generate a somewhat consistent scenery based on the prompt "a cyberpunk marketplace":

<img width="1440" alt="image" src="https://github.com/apapiu/transformer_latent_diffusion/assets/13619417/4451719f-d45a-4a86-a7bb-06c021b34996">

## Data Processing:

In [data.py](https://github.com/apapiu/transformer_latent_diffusion/blob/main/tld/data.py), I have some helper functions to process images and captions. The flow is as follows:
- Use `img2dataset` to download images from a dataframe containing URLs and captions.
- Use `CLIP` to encode the prompts and the `VAE` to encode images to latents on a web2dataset data generator.
- Save the latents and text embedding for future training.

There are two advantages to this approach. One is that the VAE encoding is somewhat expensive, so doing it every epoch would affect training times. The other is that we can discard the images after processing. For `3*256*256` images, the latent dimension is `4*32*32`, so every latent is around 4KB (when quantized in uint8; see [here](https://pub.towardsai.net/stable-diffusion-based-image-compresssion-6f1f0a399202?gi=1f45c6522d3b)). This means that 1 million latents will be "only" 4GB in size, which is easy to handle even in RAM. Storing the raw images would have been 48x larger in size.

## Architecture:

See [here](https://github.com/apapiu/transformer_latent_diffusion/blob/main/tld/denoiser.py) for the denoiser class.

The denoiser model is a Transformer-based model based on the archirtecture in [DiT](https://arxiv.org/abs/2203.02378) and [Pixart-Alpha](https://pixart-alpha.github.io/), albeit with quite a few modifications and simplifications. Using a Transformer as the denoiser is different from most diffusion models in that most other models used a CNN-based U-NET as the denoising backbone. I decided to use a Transformer for a few reasons. One was I just wanted to experiment and learn how to build and train Transformers from the ground up. Secondly, Transformers are fast both to train and to do inference on, and they will benefit most from future advances (both in hardware and in software) in performance.

Transformers are not natively built for spatial data and at first I found a lot of the outputs to be very "patchy". To remediy that I added a depth-wise convolution in the FFN layer of the transformer (this was introduced in the [Local ViT](https://arxiv.org/abs/2104.05707) paper. This allows the model to mix pixels that are close to each other with very little added compute cost.


### Img+Text+Noise Encoding:

The image latent inputs are `4*32*32` and we use a patch size of 2 to build 256 flattened `4*2*2=16` dimensional input "pixels". These are then projected into the embed dimensions are are fed through the transformer blocks. 

The text and noise conditioning is very simple - we concatenate a pooled CLIP text embedding (`ViT/L14` - 768-dimensional) and the sinusoidal noise embedding and feed it as input in the cross-attention layer in each transformer block. No unpooled CLIP embeddings are used.

### Training:
The base model is 101MM parameters and has 12 layers and embedding dimension = 768. I train it with a batch size of 256 on a A100 and learning rate  of `3e-4`. I used 1000 steps for warmup. Due to computational contraints I did not do any ablations for this configuration.


## Train and Diffusion Setup:

We train a denoising transformer that takes the following three inputs:
- `noise_level` (sampled from 0 to 1 with more values concentrated close to 0 - I use a beta distribution)
- Image latent (x) corrupted with a level of random noise
  - For a given `noise_level` between 0 and 1, the corruption is as follows:
    - `x_noisy = x*(1-noise_level) + eps*noise_level where eps ~ np.random.normal(0, 1)`
- CLIP embeddings of a text prompt
  - You can think of this as a numerical representation of a text prompt.
  - We use the pooled text embedding here (768 dimensional for `ViT/L14`)

The output is a prediction of the denoised image latent - call it `f(x_noisy)`.

The model is trained to minimize the mean squared error `|f(x_noisy) - x|` between the prediction and actual image
(you can also use absolute error here). Note that I don't reparameterize the loss in terms of the noise here to keep things simple.

Using this model, we then iteratively generate an image from random noise as follows:
    
         for i in range(len(self.noise_levels) - 1):

            curr_noise, next_noise = self.noise_levels[i], self.noise_levels[i + 1]

            # Predict original denoised image:
            x0_pred = predict_x_zero(new_img, label, curr_noise)

            # New image at next_noise level is a weighted average of old image and predicted x0:
            new_img = ((curr_noise - next_noise) * x0_pred + next_noise * new_img) / curr_noise

The `predict_x_zero` method uses classifier free guidance by combining the conditional and unconditional
prediction: `x0_pred = class_guidance * x0_pred_conditional + (1 - class_guidance) * x0_pred_unconditional`

A bit of math: The approach above falls within the VDM parametrization see 3.1 in [Kingma et al.](https://arxiv.org/pdf/2107.00630.pdf):

$$z_t = \alpha_t x + \sigma_t \epsilon,  \epsilon \sim \mathcal{N}(0,1)$$

Where $z_t$ is the noisy version of $x$ at time $t$.

Generally, $\alpha_t$ is chosen to be $\sqrt{1-\sigma_t^2}$ so that the process is variance preserving. Here, I chose $\alpha_t=1-\sigma_t$ so that we linearly interpolate between the image and random noise. Why? For one, it simplifies the updating equation quite a bit, and it's easier to understand what the noise to signal ratio will look like. I also found that the model produces sharper images faster - more validation here is needed. The updating equation above is the DDIM model for this parametrization, which simplifies to a simple weighted average. Note that the DDIM model deterministically maps random normal noise to images - this has two benefits: we can interpolate in the random normal latent space, and it generally takes fewer steps to achieve decent image quality.

## TODOS:
- better config in the train file
- how to speed up generation even more - LCMs or other sampling strategies?
- add script to compute FID




