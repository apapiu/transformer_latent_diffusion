from setuptools import setup, find_packages


def load_requirements(filename="requirements.txt"):
    with open(filename, "r") as file:
        lines = [line.strip() for line in file.readlines() if line.strip() and not line.startswith("#")]
    return lines


setup(
    name="tld",
    version="0.1.0",
    author="Alexandru Papiu",
    author_email="alex.papiu@gmail.com",
    description="Transformer Latent Diffusion",
    url="https://github.com/apapiu/transformer_latent_diffusion",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch",
        "numpy",
        "einops",
        "torchvision",
        "tqdm",
        "diffusers",
        "accelerate",
        "transformers",
        "Pillow",
        "clip @ git+https://github.com/openai/CLIP.git",
    ],
)
