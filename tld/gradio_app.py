import os
from io import BytesIO

import gradio as gr
import requests
from PIL import Image

# runpod_id = os.environ['RUNPOD_ID']
# token_id = os.environ['AUTH_TOKEN']
# url = f'https://{runpod_id}-8000.proxy.runpod.net/generate-image/'

url = os.getenv("API_URL")
token_id = os.getenv("API_TOKEN")


def generate_image_from_text(prompt, class_guidance):
    headers = {"Authorization": f"Bearer {token_id}"}

    data = {"prompt": prompt, "class_guidance": class_guidance, "seed": 11, "num_imgs": 1, "img_size": 32}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
    else:
        print("Failed to fetch image:", response.status_code, response.text)

    return image


iface = gr.Interface(
    fn=generate_image_from_text,
    inputs=["text", "slider"],
    outputs="image",
    title="Text-to-Image Generator",
    description="Enter a text prompt to generate an image.",
)

# Launch the app
iface.launch()
