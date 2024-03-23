import io
import os

import torch
import torchvision.transforms as transforms
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from tld.diffusion import DiffusionTransformer
from tld.configs import LTDConfig


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
to_pil = transforms.ToPILImage()

cfg = LTDConfig()
diffusion_transformer = DiffusionTransformer(cfg)

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def validate_token(token: str = Depends(oauth2_scheme)):
    if token != os.getenv("API_TOKEN"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


class ImageRequest(BaseModel):
    prompt: str
    class_guidance: int = 6
    seed: int = 11
    num_imgs: int = 1
    img_size: int = 32


@app.get("/")
def read_root():
    return {"message": "Welcome to Image Generator"}


@app.post("/generate-image/")
async def generate_image(request: ImageRequest, token: str = Depends(validate_token)):
    try:
        img = diffusion_transformer.generate_image_from_text(
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


# build job to test and deploy the API on a docker image (maybe in Azure?)
