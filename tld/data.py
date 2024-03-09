####data util to get and preprocess data from a text and image pair to latents and text embeddings.
### all that is required is a csv file with an image url and text caption:
#!pip install datasets img2dataset accelerate diffusers
#!pip install git+https://github.com/openai/CLIP.git

import json
import os
from dataclasses import dataclass
from typing import List, Union

import clip
import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import webdataset as wds
from diffusers import AutoencoderKL
from img2dataset import download
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def encode_text(label: Union[str, List[str]], model: nn.Module, device: str) -> Tensor:
    text_tokens = clip.tokenize(label, truncate=True).to(device)
    text_encoding = model.encode_text(text_tokens)
    return text_encoding.cpu()


@torch.no_grad()
def encode_image(img: Tensor, vae: AutoencoderKL) -> Tensor:
    x = img.to("cuda").to(torch.float16)

    x = x * 2 - 1  # to make it between -1 and 1.
    encoded = vae.encode(x, return_dict=False)[0].sample()
    return encoded.cpu()


@torch.no_grad()
def decode_latents(out_latents: torch.FloatTensor, vae: AutoencoderKL) -> Tensor:
    # expected to be in the unscaled latent space
    out = vae.decode(out_latents.cuda())[0].cpu()

    return ((out + 1) / 2).clip(0, 1)


def quantize_latents(lat: Tensor, clip_val: float = 20) -> Tensor:
    """scale and quantize latents to unit8"""
    lat_norm = lat.clip(-clip_val, clip_val) / clip_val
    return (((lat_norm + 1) / 2) * 255).to(torch.uint8)


def dequantize_latents(lat: Tensor, clip_val: float = 20) -> Tensor:
    lat_norm = (lat.to(torch.float16) / 255) * 2 - 1
    return lat_norm * clip_val


def append_to_dataset(dataset: h5py.File, new_data: Tensor) -> None:
    """Appends new data to an HDF5 dataset."""
    new_size = dataset.shape[0] + new_data.shape[0]
    dataset.resize(new_size, axis=0)
    dataset[-new_data.shape[0] :] = new_data


def get_text_and_latent_embeddings_hdf5(
    dataloader: DataLoader, vae: AutoencoderKL, model: nn.Module, drive_save_path: str
) -> None:
    """Process img/text inptus that outputs an latent and text embeddings and text_prompts, saving encodings as float16."""

    img_latent_path = os.path.join(drive_save_path, "image_latents.hdf5")
    text_embed_path = os.path.join(drive_save_path, "text_encodings.hdf5")
    metadata_csv_path = os.path.join(drive_save_path, "metadata.csv")

    with h5py.File(img_latent_path, "a") as img_file, h5py.File(text_embed_path, "a") as text_file:
        if "image_latents" not in img_file:
            img_ds = img_file.create_dataset(
                "image_latents",
                shape=(0, 4, 32, 32),
                maxshape=(None, 4, 32, 32),
                dtype="float16",
                chunks=True,
            )
        else:
            img_ds = img_file["image_latents"]

        if "text_encodings" not in text_file:
            text_ds = text_file.create_dataset(
                "text_encodings", shape=(0, 768), maxshape=(None, 768), dtype="float16", chunks=True
            )
        else:
            text_ds = text_file["text_encodings"]

        for img, (label, url) in tqdm(dataloader):
            text_encoding = encode_text(label, model).cpu().numpy().astype(np.float16)
            img_encoding = encode_image(img, vae).cpu().numpy().astype(np.float16)

            append_to_dataset(img_ds, img_encoding)
            append_to_dataset(text_ds, text_encoding)

            metadata_df = pd.DataFrame({"text": label, "url": url})
            if os.path.exists(metadata_csv_path):
                metadata_df.to_csv(metadata_csv_path, mode="a", header=False, index=False)
            else:
                metadata_df.to_csv(metadata_csv_path, mode="w", header=True, index=False)


def download_and_process_data(
    latent_save_path="latents",
    raw_imgs_save_path="raw_imgs",
    csv_path="imgs.csv",
    image_size=256,
    bs=64,
    caption_col="captions",
    url_col="url",
    download_data=True,
    number_sample_per_shard=10000,
):
    if not os.path.exists(raw_imgs_save_path):
        os.mkdir(raw_imgs_save_path)

    if not os.path.exists(latent_save_path):
        os.mkdir(latent_save_path)

    if download_data:
        download(
            processes_count=8,
            thread_count=64,
            url_list=csv_path,
            image_size=image_size,
            output_folder=raw_imgs_save_path,
            output_format="webdataset",
            input_format="csv",
            url_col=url_col,
            caption_col=caption_col,
            enable_wandb=False,
            number_sample_per_shard=number_sample_per_shard,
            distributor="multiprocessing",
            resize_mode="center_crop",
        )

    files = os.listdir(raw_imgs_save_path)
    tar_files = [os.path.join(raw_imgs_save_path, file) for file in files if file.endswith(".tar")]
    print(tar_files)
    dataset = wds.WebDataset(tar_files)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # output is (img_tensor, (caption , url_col)) per batch:
    dataset = (
        dataset.decode("pil")
        .to_tuple("jpg;png", "json")
        .map_tuple(transform, lambda x: (x["caption"], x[url_col]))
    )

    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)

    model, _ = clip.load("ViT-L/14")

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    vae = vae.to("cuda")
    model.to("cuda")

    print("Starting to encode latents and text:")
    get_text_and_latent_embeddings_hdf5(dataloader, vae, model, latent_save_path)
    print("Finished encode latents and text:")


@dataclass
class DataConfiguration:
    data_link: str
    caption_col: str = "caption"
    url_col: str = "url"
    latent_save_path: str = "latents_folder"
    raw_imgs_save_path: str = "raw_imgs_folder"
    use_drive: bool = False
    initial_csv_path: str = "imgs.csv"
    number_sample_per_shard: int = 10000
    image_size: int = 256
    batch_size: int = 64
    download_data: bool = True


if __name__ == "__main__":
    use_wandb = False

    if use_wandb:
        import wandb

        os.environ["WANDB_API_KEY"] = "key"
        #!wandb login

    data_link = "https://huggingface.co/datasets/zzliang/GRIT/resolve/main/grit-20m/coyo_0_snappy.parquet?download=true"

    data_config = DataConfiguration(
        data_link=data_link,
        latent_save_path="latent_folder",
        raw_imgs_save_path="raw_imgs_folder",
        download_data=False,
        number_sample_per_shard=1000,
    )

    if use_wandb:
        wandb.init(project="image_vae_processing", entity="apapiu", config=data_config)

    if not os.path.exists(data_config.latent_save_path):
        os.mkdir(data_config.latent_save_path)

    config_file_path = os.path.join(data_config.latent_save_path, "config.json")
    with open(config_file_path, "w") as f:
        json.dump(data_config.__dict__, f)

    print("Config saved to:", config_file_path)

    df = pd.read_parquet(data_link)
    ###add additional data cleaning here...should I
    df = df.iloc[:3000]
    df[["key", "url", "caption"]].to_csv("imgs.csv", index=None)

    if data_config.use_drive:
        from google.colab import drive

        drive.mount("/content/drive")

    download_and_process_data(
        latent_save_path=data_config.latent_save_path,
        raw_imgs_save_path=data_config.raw_imgs_save_path,
        csv_path=data_config.initial_csv_path,
        image_size=data_config.image_size,
        bs=data_config.batch_size,
        caption_col=data_config.caption_col,
        url_col=data_config.url_col,
        download_data=data_config.download_data,
        number_sample_per_shard=data_config.number_sample_per_shard,
    )

    if use_wandb:
        wandb.finish()
