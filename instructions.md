## after cloning the repo run following commands to install dependencies



pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121


pip install einops diffusers timm

pip install av=13.1.0

# Login to hugging face
huggingface-cli login


# Download the Models
huggingface-cli download Etched/oasis-500m oasis500m.safetensors

huggingface-cli download Etched/oasis-500m vit-l-20.safetensors



## Run generate.py with --oasis-ckpt "<path to dit>" --vae-ckpt "<path to vit>"


