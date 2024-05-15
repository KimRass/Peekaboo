# References:
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py

import torch
import torch.nn as nn
import torchvision.transforms as T
import einops
from diffusers import DiffusionPipeline # 0.27.2
from PIL import Image
import random

from utils import unload_from_gpu


class LearnableAlphaMask(nn.Module):
    """
    "We use a randomly initialized alpha mask ($\alpha$) to alpha-blend the
    puppy image ($x$) with different backgrounds ($b$), generating new composite
    images ($\hat{x}$): $\hat{x} = \alpha x + (1 - \alpha)b$"
    """
    def __init__(self, h, w):
        super().__init__()

        self.alpha = nn.Parameter(torch.randn(h, w))

    def forward(self, channels):
        return torch.sigmoid(
            einops.repeat(self.alpha, pattern="h w -> 1 c h w", c=channels),
        )


def get_alpha_reg_loss(alpha_mask):
    """
    "$\mathcal{L}_{\alpha} = \sum_{i}\alpha_{i}$, $i$ indexes pixel location in
    $\alpha$.
    """
    return torch.sum(alpha_mask, dim=(-2, -1))


def get_rand_uint8_color():
    return (
        random.randint(0, 255), random.randint(0, 255), random.randint(0, 255),
    )


def get_rand_uint8_color_pil_image(h, w):
    return Image.new("RGB", size=(w, h), color=get_rand_uint8_color())


def compose(image, alpha_mask):
    transform = T.Compose(
        [T.ToTensor(), T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))],
    )
    _, _, h, w = image.shape
    bg = get_rand_uint8_color_pil_image(h=h, w=w)
    bg = transform(bg)
    bg = bg[None, ...]
    return torch.lerp(
        image, bg.to(image.device), alpha_mask(channels=3).to(image.device),
    )

if __name__ == "__main__":
    # alpha_mask = torch.rand((1, 2, 3))
    device = torch.device("mps")
    # device = torch.device("cuda")
    sd = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        # torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    # unload_from_gpu(sd)
    
    """
    "We first degrade latent vector $z$ of composite image $x$ using forward
    diffusion, introducing Gaussian noise $\epsilon \sim N$ to $z$, resulting in
    a noisy $\tilde{z}$. We then perform diffusion denoising conditioned on the
    text embedding with pre-trained $\mathcal{D}$. Our loss $\mathcal{L}_{s}$ is measured
    as the reconstruction error of noise $\epsilon$, given noisy $\tilde{z}$ and
    text embedding $\mathcal{T}(p)$: $\mathcal{L}_{s}
    = \text{MSE}(\epsilon, \mathcal{D}(\tilde{z}, \mathcal{T}(p)))$ where
    $\text{MSE}$ refers to mean-squared loss."
    """
    sd.config
    
    img_size = 224
    comp_image = torch.randn((1, 3, img_size, img_size), dtype=torch.float16, device=device) # $x$
    comp_image_latent = sd.vae.encode(comp_image).latent_dist.mode() # $z$
    comp_image_latent.shape

    alpha_mask = LearnableAlphaMask(h=img_size, w=img_size).to(device)
    comp_image = compose(image, alpha_mask)

    sd.encode_image(comp_image)
    
    batch_size = 1
    num_images_per_prompt = 1
    num_channels_latents = sd.unet.config.in_channels
    height = sd.unet.config.sample_size * sd.vae_scale_factor
    width = sd.unet.config.sample_size * sd.vae_scale_factor
    generator = None
    latents = None

    prompt = "Harry potter"
    do_classifier_free_guidance = False
    negative_prompt = None
    prompt_embeds = None
    negative_prompt_embeds = None
    cross_attention_kwargs = None
    lora_scale = None
    clip_skip = None
    prompt_embed, _ = sd.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=clip_skip,
    )

    latents = sd.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embed.dtype,
        device,
        generator,
        latents,
    )
    latents.shape