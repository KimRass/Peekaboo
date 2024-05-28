# References:
    # https://huggingface.co/stabilityai/sd-vae-ft-mse
    # https://github.com/richzhang/PerceptualSimilarity
    # https://wandb.ai/capecape/ddpm_clouds/reports/Using-Stable-Diffusion-VAE-to-encode-satellite-images--VmlldzozNDA2OTgx

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from diffusers.models import AutoencoderKL
from lpips import LPIPS
import einops
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--n_cpus", type=int, required=True)
    parser.add_argument("--n_warmup_steps", type=int, required=True)
    parser.add_argument("--img_size", type=int, required=True)

    parser.add_argument("--seed", type=int, default=223, required=False)

    args = parser.parse_args()

    args_dict = vars(args)
    new_args_dict = dict()
    for k, v in args_dict.items():
        new_args_dict[k.upper()] = v
    args = argparse.Namespace(**new_args_dict)
    return args


class SDVAE(object):
    def __init__(self, vae):
        self.vae = vae

        self.scaling_factor = vae.config.scaling_factor
        self.sample_size = vae.config.sample_size
        self.lpips_fn = LPIPS(net="alex")

    def get_ori_image(
        self, img_path, device=torch.device("cpu"), img_size=None,
    ):
        image = Image.open(img_path).convert("RGB")
        if img_size is not None:
            image = TF.resize(image, size=img_size, antialias=True)
            image = TF.center_crop(image, output_size=img_size)
        image = TF.to_tensor(image)
        return image[None, ...].to(device)

    @torch.inference_mode()
    def reconstruct(self, ori_image):
        latent = self.scaling_factor * self.vae.encode(
            ori_image,
        ).latent_dist.sample()
        out = self.vae.decode((1 / self.scaling_factor) * latent)
        return latent, out.sample

    def vis_image(self, recon_image):
        tensor = recon_image.detach().cpu()
        grid = make_grid(tensor, nrow=1)
        grid.clamp_(0, 1)
        return TF.to_pil_image(grid)

    def vis_latent(self, latent, h, w):
        latent = F.interpolate(latent.detach().cpu(), size=(h, w))
        minim = torch.min(
            torch.min(latent, dim=-2, keepdim=True)[0], dim=-1, keepdim=True,
        )[0]
        maxim = torch.max(
            torch.max(latent, dim=-2, keepdim=True)[0], dim=-1, keepdim=True,
        )[0]
        latent = (latent - minim) / (maxim - minim)
        grid = make_grid(
            latent.permute(1, 0, 2, 3), nrow=latent.size(1), padding=0,
        )
        return TF.to_pil_image(grid)

    def vis(self, img_path, img_size=None):
        ori_image = self.get_ori_image(
            img_path=img_path, device=device, img_size=img_size,
        )
        latent, recon_image = self.reconstruct(ori_image)

        _, _, h, w = ori_image.shape
        new_image = Image.new(
            mode="RGB", size=(w * 6, h),
        )
        new_image.paste(self.vis_image(ori_image), (0, 0))
        new_image.paste(self.vis_image(recon_image), (w, 0))
        new_image.paste(
            self.vis_latent(latent, h=h, w=w), (w * 2, 0),
        )
        new_image.show()

    @torch.inference_mode()
    def get_mse(self, ori_image, recon_image):
        return F.mse_loss(ori_image, recon_image, reduction="mean").item()

    @staticmethod
    def shift_scale(x):
        return (x - 0.5) / 0.5

    @torch.inference_mode()
    def get_lpips(self, ori_image, recon_image):
        return self.lpips_fn.to(ori_image.device).forward(
            self.shift_scale(ori_image), self.shift_scale(recon_image),
        ).item()

    @torch.inference_mode()
    def eval(self, img_dir):
        sum_mse = 0
        sum_lpips = 0
        cnt = 0
        for img_path in tqdm(list(Path(img_dir).glob("**/*.png"))):
            ori_image = self.get_ori_image(
                img_path, device=device, img_size=self.sample_size,
            )
            _, recon_image = self.reconstruct(ori_image)

            mse = self.get_mse(ori_image=ori_image, recon_image=recon_image)
            lpips = self.get_lpips(ori_image=ori_image, recon_image=recon_image)
            sum_mse += mse
            sum_lpips += lpips
            cnt += 1
        return sum_mse / cnt, sum_lpips / cnt


if __name__ == "__main__":
    device = torch.device("mps")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        subfolder="vae",
    ).to(device)
    # model = "CompVis/stable-diffusion-v1-4"
    # pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)
    # sd = StableDiffusionPipeline.from_pretrained(model, use_safttensors=True)
    # sd.vae.config

    model = SDVAE(vae=vae)
    # img_path="/Users/kimjongbeom/Downloads/train/train_sharp_bicubic/X4/233/00000093.png"
    img_path = "/Users/kimjongbeom/Documents/workspace/Peekaboo/resources/harry_potter.webp"
    model.vis(img_path, img_size=model.sample_size)
    model.vis(img_path, img_size=model.sample_size * 2)

    img_dir = "/Users/kimjongbeom/Downloads/train/train_sharp_bicubic/X4/239"
    mse, lpips = model.eval(img_dir)
    print(f"[ MSE: {mse:.6f} ][ LPIPS: {lpips:.4f} ]")
