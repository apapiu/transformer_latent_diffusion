####data util to get and preprocess data from a text and image pair to latents and text embeddings.
### all that is required is a csv file with an image url and text caption:
#!pip install datasets img2dataset accelerate diffusers
#!pip install git+https://github.com/openai/CLIP.git

import os
import pandas as pd
import numpy as np

import torch
import torchvision.transforms as transforms
from tqdm import tqdm

import webdataset as wds
from img2dataset import download

#models:
import clip
from diffusers import AutoencoderKL


@torch.no_grad()
def encode_text(label):
    text_tokens = clip.tokenize(label, truncate=True).cuda()
    text_encoding = model.encode_text(text_tokens)
    return text_encoding.cpu()

@torch.no_grad()
def encode_image(img):
    x = img.to('cuda').to(torch.float16)

    x = x*2 - 1 #to make it between -1 and 1.
    encoded = vae.encode(x, return_dict=False)[0].sample()
    return encoded.cpu()

@torch.no_grad()
def decode_latents(out_latents):
    #expected to be in the unscaled latent space 
    out = vae.decode(out_latents.cuda())[0].cpu()

    return ((out + 1)/2).clip(0,1)


def quantize_latents(lat, clip_val=20):
    """scale and quantize latents to unit8"""
    lat_norm = lat.clip(-clip_val, clip_val)/clip_val
    return (((lat_norm + 1)/2)*255).to(torch.uint8)

def dequantize_latents(lat, clip_val=20):
    lat_norm = (lat.to(torch.float16)/255)*2 - 1
    return lat_norm*clip_val

def get_text_and_latent_embedings(dataloader):
    """dataloader that outputs an img tensor and text_prompts"""
    text_encodings = []
    img_encodings = []

    for img, label in tqdm(dataloader):

        #encode text:
        text_encodings.append(encode_text(label).cpu())
        ##encode images:
        img_encodings.append(encode_image(img).cpu())

    img_encodings = torch.cat(img_encodings)
    text_encodings = torch.cat(text_encodings)
    return img_encodings, text_encodings

if __name__ == '__main__':
    from google.colab import drive
    drive.mount('/content/drive')

    #get a parquet file from somehwere and save it as csv (optionally clean the dataframe):
    def get_csv():
        df = pd.read_parquet('https://huggingface.co/datasets/wanng/midjourney-v5-202304-clean/resolve/main/data/upscaled_prompts_df.parquet')
        #clean and filter data here...
        df.to_csv("imgs.csv", index=None)

    folder_name = 'test'
    csv_path = 'imgs.csv'
    drive_save_path = '/content/drive/MyDrive/midjourney_test'
    download_data = True
    save_data = True

    bs = 64
    image_size = 256

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    if not os.path.exists(drive_save_path):
        os.mkdir(drive_save_path)

    to_pil = transforms.ToPILImage()

    caption_col = "clean_prompts"
    url_col = "Attachments"

    get_csv()

    if download_data:

        download(
            processes_count=8,
            thread_count=64,
            url_list=csv_path,
            image_size=image_size,
            output_folder=folder_name,
            output_format="webdataset",
            input_format="csv",
            url_col=url_col,
            caption_col=caption_col,
            enable_wandb=False,
            number_sample_per_shard=10000,
            distributor="multiprocessing",
            resize_mode="center_crop"
        )

    files = os.listdir(folder_name)
    tar_files = [os.path.join(folder_name, file) for file in files if file.endswith('.tar')]

    dataset = wds.WebDataset(tar_files)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = dataset.decode("pil").to_tuple("jpg;png", "json").map_tuple(transform, lambda x: x["caption"])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)

    model, preprocess = clip.load("ViT-L/14")

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    vae = vae.to('cuda')
    model.to('cuda')

    image_latents, text_encodings = get_text_and_latent_embedings(dataloader)

    if save_data:
        np.save(os.path.join(drive_save_path, 'image_latents.npy'), image_latents.numpy())
        np.save(os.path.join(drive_save_path, 'text_encodings.npy'), text_encodings.numpy())
