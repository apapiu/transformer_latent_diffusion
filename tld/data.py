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
def encode_text(label, model):
    text_tokens = clip.tokenize(label, truncate=True).cuda()
    text_encoding = model.encode_text(text_tokens)
    return text_encoding.cpu()

@torch.no_grad()
def encode_image(img, vae):
    x = img.to('cuda').to(torch.float16)

    x = x*2 - 1 #to make it between -1 and 1.
    encoded = vae.encode(x, return_dict=False)[0].sample()
    return encoded.cpu()

@torch.no_grad()
def decode_latents(out_latents, vae):
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

def get_text_and_latent_embedings(dataloader, vae, model, drive_save_path):
    """dataloader that outputs an img tensor and text_prompts"""
    text_encodings = []
    img_encodings = []

    i = 0
    for img, label in tqdm(dataloader):

        #encode text:
        text_encodings.append(encode_text(label, model).cpu())
        ##encode images:
        img_encodings.append(encode_image(img, vae).cpu())

        if i%100 == 1:
            print("Saving")
            np.save(os.path.join(drive_save_path, 'image_latents.npy'), torch.cat(img_encodings).numpy())
            np.save(os.path.join(drive_save_path, 'text_encodings.npy'), torch.cat(text_encodings).numpy())

    img_encodings = torch.cat(img_encodings)
    text_encodings = torch.cat(text_encodings)
    return img_encodings, text_encodings

def get_csv():
        df = pd.read_parquet('https://huggingface.co/datasets/wanng/midjourney-v5-202304-clean/resolve/main/data/upscaled_prompts_df.parquet')
        #clean and filter data here...
        df = df.iloc[:2000]
        df.to_csv("imgs.csv", index=None)
        
        
def download_and_process_data(drive_save_path='/kaggle/working/saved_data',
                              image_size = 256,
                              bs = 64,
                              folder_name='test',
                              caption_col = "clean_prompts",
                              url_col = "Attachments",
                              clean_csv_function = get_csv
                             ):
    
    csv_path = 'imgs.csv'
    download_data = True
    save_data = True

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    if not os.path.exists(drive_save_path):
        os.mkdir(drive_save_path)

    to_pil = transforms.ToPILImage()

    #clean_csv_function()

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
            number_sample_per_shard=1000,
            distributor="multiprocessing",
            resize_mode="center_crop"
        )

    files = os.listdir(folder_name)
    tar_files = [os.path.join(folder_name, file) for file in files if file.endswith('.tar')]
    print(tar_files)

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

    image_latents, text_encodings = get_text_and_latent_embedings(dataloader, vae, model, drive_save_path)

    if save_data:
        np.save(os.path.join(drive_save_path, 'image_latents.npy'), image_latents.numpy())
        np.save(os.path.join(drive_save_path, 'text_encodings.npy'), text_encodings.numpy())

if __name__ == '__main__':
    df = pd.read_parquet('https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-00021-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet')
    df = df.iloc[:3000]
    df.to_csv("imgs.csv", index=None)

    data_path = '/kaggle/working/saved_data'
    caption_col = 'text'
    url_col = 'url'

    download_and_process_data(drive_save_path=data_path,
                              folder_name=caption_col,
                              caption_col=url_col,
                              url_col="url",
                              clean_csv_function = get_csv)
