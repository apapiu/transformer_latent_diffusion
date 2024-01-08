# transformer_latent_diffusion
Text to Image Latent Diffusion using a Transformer core.

Codebase to train a Text to Image Latent Diffusion Transformer based model in Pytorch. See below for notebooks and examples with prompts. The model generates 256*256 resolution images.

This is part II of aprevious project I did where I trained a pixel level diffusion model in keras. Here I train a latent diffusion model in pytorch. 

The main goal of this project is to showcase a model in Pytorch that is: 
- fast(almost real time generation)
- small (100MM params)
- reasonably good (of course not Sota)
- can be trained in a reasonable ammount of time on a single GPU (under 50 hours on a A100 or equivalent).
- uses ~ 1 million images with a focus on data quality over quantity

## Speed:

I try to speed up training and inference as much as possible by:
- using mixed precision for training + sdpa
- precompute all latent and text embeddings
- using float16 precision for inference
- using sdpa for the attention natively + torch.compile
- use a deterministic denoising process (DDIM) for fewer steps
- TODO: would distillation or something like LCM work here?

The time to generate 16 images on a T4: A100:


### Code:


## Data Processing:

In (data.py)[https://github.com/apapiu/transformer_latent_diffusion/blob/main/tld/data.py] I have some helper functions to process images and captions. The flow is as follows:
- use `img2dataset` to download images from a dataframe containing urls and captions
- use `clip` to encode the pormpts and the `vae`  to encode images to latents on a web2dataset data generator.
- save the latents and text embedding for future training.

There are two advantages to this approach. One is that the vae encoding is somewhat expensive so doing it every epoch would affect training times. The other is that we can discard the images after processing. For `3*256*256` images the latent dimension is `4*32*32` so every latent is around 4KB (when quantized in uint8 see [here](https://pub.towardsai.net/stable-diffusion-based-image-compresssion-6f1f0a399202?gi=1f45c6522d3b)). This means that 1 million latents will be "only" 4GB in size which is easy to handle even in RAM. Storing the raw images would have been 48x larger in size.

## Architecture:

The code for the architecture is (here)[https://github.com/apapiu/transformer_latent_diffusion/blob/main/tld/denoiser.py]

The denoiser model is a transformer based model insipired by [DeIT] and [Pixart Alpha] albeit with quite a few modifications and simplifications. Using a transformer as the denoiser is different from most diffusion models in that most other models used a CNN based U-NET as the denoising backbone. I decided to use a transformer for a few reasons. One was I just wanted to experiment and learn how to build and train transformers. Secondly transformers are fast both to train and to do inference on and they will benefit most from future advances (both in hardware and in software) in performance. 

Transformers are not natively built for spatial data and at first I found a lot of the outputs to be very "patchy". To remediy that I added a depth-wise convolution in the FFN layer of the transformer (this was introduced in the [Local ViT](https://arxiv.org/abs/2104.05707) paper. This allows the model to mix pixels that are close to each other with very little added compute cost.

### Img+Text+Noise Encoding:

The image latent inputs are `4*32*32` and we use a patch size of 2 to build 256 flattened `4*2*2=16` dimensional input "pixels". These are then projected into the embed dimensions are are fed through the transformer blocks. 

The text and noise conditioning is very simple - we concatenate a pooled CLIP text embedding (`ViT/L14` - 768-dimensional) and the sinsuoidal noise embedding and feed it as input in the cross-attention layer in each transformer block. No unpooled CLIP embeddings are used.

### Params:
The base model is 101MM parameters and has 12 layers and embedding dimension = 768. We trained it with a batch size of 256 on a A100 for 10 hours and learning rate  of `3e-4`. I used 1000 steps for warmup. Due to computational contraints I did not do any ablations for this configuration.



##  Diffusion Schedule:




