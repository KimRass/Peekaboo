# References:
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
import sys
sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/Peekaboo")
import torch
import torch.nn as nn
from torch.optim import SGD
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import einops
from diffusers import DiffusionPipeline # 0.27.2
from PIL import Image
import random
from tqdm import tqdm

from utils import unload_from_gpu, image_to_grid


class LearnableAlphaMask(nn.Module):
    """
    "We use a randomly initialized alpha mask ($\alpha$) to alpha-blend the
    puppy image ($x$) with different backgrounds ($b$), generating new composite
    images ($\hat{x}$): $\hat{x} = \alpha x + (1 - \alpha)b$"
    "At the initial iteration it is entirely random Gaussian noise."
    """
    def __init__(self, h, w):
        super().__init__()

        self.alpha = nn.Parameter(torch.randn(h, w))

    def forward(self, channels):
        """
        "We apply a bilateral blur to that matrix and then clip it between 0
        and 1 using a sigmoid function."
        """
        return einops.repeat(
            torch.sigmoid(self.alpha), pattern="h w -> 1 c h w", c=channels,
        )


class Peekaboo(nn.Module):
    def __init__(self, sd):
        super().__init__()

        self.sd = sd
        self.sd.vae.eval()
        self.sd.unet.eval()

        self.img_size = sd.unet.config.sample_size * sd.vae_scale_factor
        self.alpha_mask = LearnableAlphaMask(h=self.img_size, w=self.img_size)
        self.transform = T.Compose(
            [
                T.Resize((self.img_size), antialias=True),
                T.CenterCrop((self.img_size)),
            ]
        )
        self.to_tensor_and_norm = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def get_alpha_reg_loss(self):
        """
        "$\mathcal{L}_{\alpha} = \sum_{i}\alpha_{i}$, $i$ indexes pixel location in
        $\alpha$.
        """
        return torch.sum(self.alpha_mask.alpha, dim=(-2, -1))

    def get_score_distil_loss(self, comp_image, prompt_embed):
        """
        "We first degrade latent vector $z$ of composite image $x$ using forward
        diffusion, introducing Gaussian noise $\epsilon \sim N$ to $z$,
        resulting in a noisy $\tilde{z}$. We then perform diffusion denoising
        conditioned on the text embedding with pre-trained $\mathcal{D}$. Our
        loss $\mathcal{L}_{s}$ is measured as the reconstruction error of noise
        $\epsilon$, given noisy $\tilde{z}$ and text embedding
        $\mathcal{T}(p)$: $\mathcal{L}_{s}
        = \text{MSE}(\epsilon, \mathcal{D}(\tilde{z}, \mathcal{T}(p)))$ where
        $\text{MSE}$ refers to mean-squared loss."
        """
        with torch.no_grad():
            comp_image_latent = self.sd.vae.encode(
                comp_image.to(dtype=sd.dtype, device=self.sd.device),
            ).latent_dist.mode() # "$z$"
        rand_noise = self.sd.prepare_latents(
            batch_size=1,
            num_channels_latents=self.sd.unet.config.in_channels,
            height=self.img_size,
            width=self.img_size,
            dtype=self.sd.dtype,
            device=self.sd.device,
            generator=None,
            latents=None,
        ) # "$\epsilon$"
        rand_timestep = torch.randint(
            0, self.sd.scheduler.config.num_train_timesteps, size=(1,), device=self.sd.device,
        )
        noisy_comp_image_latent = self.sd.scheduler.add_noise(
            original_samples=comp_image_latent,
            noise=rand_noise,
            timesteps=rand_timestep,
        ) # "$\tilde{z}$"
        # with torch.inference_mode():
        with torch.no_grad():
            pred_noise = self.sd.unet(
                sample=noisy_comp_image_latent,
                timestep=rand_timestep,
                encoder_hidden_states=prompt_embed,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
            )[0] # "$\hat{\epsilon}$"
        return torch.sum((rand_noise - pred_noise) ** 2)

    @staticmethod
    def get_rand_uint8_color_bg(h, w):
        return Image.new(
            mode="RGB",
            size=(w, h),
            color=(
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            ),
        )

    def get_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return self.to_tensor_and_norm(image)[None, ...]

    # @torch.inference_mode()
    @torch.no_grad()
    def get_prompt_embed(self, prompt):
        prompt_embed, _ = self.sd.encode_prompt( # "$\mathcal{T}(p)$"
            prompt=prompt, # "$p$"
            device=self.sd.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )
        return prompt_embed.detach()

    def get_composite_image(self, image):
        _, _, h, w = image.shape
        bg = self.get_rand_uint8_color_bg(h=h, w=w)
        bg = self.to_tensor_and_norm(bg)[None, ...]        
        return torch.lerp(image, bg, self.alpha_mask(channels=3))

    def optimize(self, img_path, prompt, num_steps, lr):
        image = self.get_image(img_path)
        prompt_embed = self.get_prompt_embed(prompt)

        optim = SGD(self.alpha_mask.parameters(), lr=lr)
        for _ in tqdm(range(num_steps)):
            reg_loss = self.get_alpha_reg_loss()
            comp_image = self.get_composite_image(image)
            score_distil_loss = self.get_score_distil_loss(
                comp_image=comp_image, prompt_embed=prompt_embed,
            )
            loss = reg_loss + score_distil_loss

            log = f"[ {reg_loss.item():.3f} ]"
            log += f"[ {score_distil_loss.item():.3f} ]"
            log += f"[ {loss.item():.3f} ]"
            print(log)

            optim.zero_grad()
            loss.backward()
            optim.step()
        # return self.alpha_mask.alpha
        return self.get_composite_image(image)


if __name__ == "__main__":
    device = torch.device("mps")
    torch_dtype = torch.float32
    sd = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch_dtype,
        use_safetensors=True,
    ).to(device)
    # unload_from_gpu(sd)

    peekaboo = Peekaboo(sd)
    img_path = "/Users/jongbeomkim/Desktop/workspace/Peekaboo/resources/harry_potter.webp"
    prompt = "Harry Potter"
    comp_image = peekaboo.optimize(
        img_path=img_path, prompt=prompt, num_steps=50, lr=0.00001,
    )
    image_to_grid(comp_image, 1).show()

    # temp = torch.sigmoid(peekaboo.alpha_mask.alpha)
    # TF.to_pil_image(temp).show()
