import numpy as np 
from PIL import Image
from diffusers.models import AutoencoderKL
import torch
def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    # Create a new black image with size 2 * image_size
    new_size = 2 * image_size
    new_image = Image.new('RGB', (new_size, new_size), (0, 0, 0))

    # Calculate the position to paste the original image at the center
    left = (new_size - pil_image.width) // 2
    top = (new_size - pil_image.height) // 2
    new_image.paste(pil_image, (left, top))

    pil_image = new_image
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

path = '/root/DIT/DiT/fast-DiT/features/imagenet256_features/80000.npy'
features = np.load(path)
print(features)
print(features.shape)  
print(features.max()) 
print(features.min()) 

vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{"ema"}").to('cuda')
features = torch.tensor(features).to('cuda')
samples = vae.decode(features / 0.18215).sample
print(samples.shape) 
print(samples.max()) 
print(samples.min()) 

samples = (samples + 1) / 2  
from torchvision.utils import save_image
save_image(samples, "decoded_image.png")
print("图片保存成功: decoded_image.png")

