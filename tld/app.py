import io
from typing import Optional

import clip
import numpy as np
import requests
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from diffusers import AutoencoderKL
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from tld.denoiser import Denoiser
from tld.diffusion import DiffusionGenerator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
to_pil = transforms.ToPILImage()


def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


@torch.no_grad()
def encode_text(label, model):
    text_tokens = clip.tokenize(label, truncate=True).to(device)
    text_encoding = model.encode_text(text_tokens)
    return text_encoding.cpu()


def generate_image_from_text(prompt, class_guidance=6, seed=11, num_imgs=1, img_size=32):
    n_iter = 15
    nrow = int(np.sqrt(num_imgs))

    cur_prompts = [prompt] * num_imgs
    labels = encode_text(cur_prompts, clip_model)
    out, out_latent = diffuser.generate(
        labels=labels,
        num_imgs=num_imgs,
        class_guidance=class_guidance,
        seed=seed,
        n_iter=n_iter,
        exponent=1,
        scale_factor=8,
        sharp_f=0,
        bright_f=0,
    )

    out = to_pil((vutils.make_grid((out + 1) / 2, nrow=nrow, padding=4)).float().clip(0, 1))
    return out


###config:
vae_scale_factor = 8
img_size = 32
model_dtype = torch.float32

file_url = "https://huggingface.co/apapiu/small_ldt/resolve/main/state_dict_378000.pth"
local_filename = "state_dict_378000.pth"
download_file(file_url, local_filename)

denoiser = Denoiser(
    image_size=32,
    noise_embed_dims=256,
    patch_size=2,
    embed_dim=768,
    dropout=0,
    n_layers=12,
)

state_dict = torch.load("state_dict_378000.pth", map_location=torch.device("cpu"))

denoiser = denoiser.to(model_dtype)
denoiser.load_state_dict(state_dict)
denoiser = denoiser.to(device)

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=model_dtype).to(device)

clip_model, preprocess = clip.load("ViT-L/14")
clip_model = clip_model.to(device)

diffuser = DiffusionGenerator(denoiser, vae, device, model_dtype)

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def validate_token(token: str = Depends(oauth2_scheme)):
    if token != "key_here":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


class ImageRequest(BaseModel):
    prompt: str
    class_guidance: Optional[int] = 6
    seed: Optional[int] = 11
    num_imgs: Optional[int] = 1
    img_size: Optional[int] = 32


@app.post("/generate-image/")
async def generate_image(request: ImageRequest, token: str = Depends(validate_token)):
    try:
        img = generate_image_from_text(
            prompt=request.prompt,
            class_guidance=request.class_guidance,
            seed=request.seed,
            num_imgs=request.num_imgs,
            img_size=request.img_size,
        )
        # Convert PIL image to byte stream suitable for HTTP response
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# add API test to pytests as a separate file (this will also be ran as a github action)
# test authentification
# build a docker file and docker image to use for gitub actions
# build job to deploy the API on a docker image (maybe in Azure?)
