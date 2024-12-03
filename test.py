import numpy as np 
from PIL import Image
from diffusers.models import AutoencoderKL
import torch
from models import DiT_models
from diffusion import create_diffusion
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
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

    # scale = image_size / min(*pil_image.size)
    # pil_image = pil_image.resize(
    #     tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    # )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

path = '/root/DIT/DiT/fast-DiT/features/imagenet256_features/51200.npy'
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

# def load_checkpoint(checkpoint_path, device):   
#     checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
#     if "train_steps" in checkpoint:
#         train_steps = checkpoint["train_steps"]
#     else:
#         train_steps = 4000
#     args = checkpoint["args"]
#     model = DiT_models[args.model](
#         input_size=args.image_size // 8,
#         num_classes=args.num_classes,
#         class_dropout_prob=0.0
#     ).to(device)
#     model.load_state_dict(checkpoint["model"])
#     opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
#     opt.load_state_dict(checkpoint["opt"])
#     vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
#     return model, opt, train_steps, args, vae
# checkpoint_path = '/remote-home/fast-DiT/results/001-DiT-XL-4/checkpoints/0004000.pt'
# model, opt, train_steps, args, vae = load_checkpoint(checkpoint_path, 'cuda')
# model.eval()
# diffusion = create_diffusion("100")

# n = 1  # Number of images to generate
# z = torch.randn(1, 4, 32, 32, device='cuda')

# # Define fixed class labels (all 0):
# class_labels = torch.tensor([0], device='cuda')
# model_kwargs = dict(y=class_labels)

# # Sample images:
# samples = diffusion.p_sample_loop(
#     model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device='cuda'
# )
# print(f"Latent samples before decode - min: {samples.min()}, max: {samples.max()}, mean: {samples.mean()}")

# samples = vae.decode(samples / 0.18215).sample 
# print(f"samples min: {samples.min()}, max: {samples.max()}, mean: {samples.mean()}")

# # Save and display images:
# samples = (samples + 1) / 2  
# save_image(samples, "sample3.png")
# print(f"Generated {n} images and saved to sample.png")
