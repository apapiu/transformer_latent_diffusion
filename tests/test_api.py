import os

from fastapi.testclient import TestClient
from tld.app import app
import PIL
from PIL import Image
from io import BytesIO


client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Image Generator"}


def test_generate_image_unauthorized():
    response = client.post("/generate-image/", json={})
    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}


def test_generate_image_authorized():
    api_token = os.getenv("API_TOKEN")
    response = client.post(
        "/generate-image/", json={"prompt": "a cute cat"}, headers={"Authorization": f"Bearer {api_token}"}
    )
    assert response.status_code == 200

    image = Image.open(BytesIO(response.content))
    assert type(image) == PIL.JpegImagePlugin.JpegImageFile
