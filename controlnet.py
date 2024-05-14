# References:
    # https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/distributions/distributions.py
    # https://github.com/huggingface/diffusers/blob/v0.27.2/src/diffusers/models/autoencoders/autoencoder_kl.py
    # https://huggingface.co/docs/diffusers/main/en/quicktour
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_condition.py
    # https://wandb.ai/johnowhitaker/midu-guidance/reports/Mid-U-Guidance-Fast-Classifier-Guidance-for-Latent-Diffusion-Models--VmlldzozMjg0NzA1
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_blocks.py

import torch
from diffusers import DiffusionPipeline # 0.27.2
# import diffusers
# diffusers.__version__
from tqdm import tqdm
import PIL.Image
import numpy as np
import gc
import torch
from diffusers.utils import deprecate
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import(
    StableDiffusionPipelineOutput,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
    retrieve_timesteps,
)


def unload_from_gpu(var):
    var = var.to(torch.device("cpu"))
    gc.collect()
    torch.cuda.empty_cache()


def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed`[0]`)
    print(f"Image at step {i}")
    image_pil.show()


# device = torch.device("cpu")
device = torch.device("mps")
# device = torch.device("cuda")
sd = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to(device)


prompt = "Many Pokemons"
height =None
width = None
num_inference_steps = 50
timesteps = None
guidance_scale = 7.5
negative_prompt = None
num_images_per_prompt= 1
eta = 0.0
generator = None
latents = None
prompt_embeds = None
negative_prompt_embeds = None
ip_adapter_image = None
ip_adapter_image_embeds = None
output_type = "pil"
return_dict = True
cross_attention_kwargs = None
guidance_rescale = 0.0
clip_skip = None
callback_on_step_end = None
callback_on_step_end_tensor_inputs = ["latents"]
kwargs = dict()

with torch.no_grad():
    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    if callback is not None:
        deprecate(
            "callback",
            "1.0.0",
            "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )
    if callback_steps is not None:
        deprecate(
            "callback_steps",
            "1.0.0",
            "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        )


    # 0. Default height and width to unet
    height = height or sd.unet.config.sample_size * sd.vae_scale_factor
    width = width or sd.unet.config.sample_size * sd.vae_scale_factor
    # to deal with lora scaling and other possible forward hooks

    # 1. Check inputs. Raise error if not correct
    sd.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        ip_adapter_image,
        ip_adapter_image_embeds,
        callback_on_step_end_tensor_inputs,
    )

    sd._guidance_scale = guidance_scale
    sd._guidance_rescale = guidance_rescale
    sd._clip_skip = clip_skip
    sd._cross_attention_kwargs = cross_attention_kwargs
    sd._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape`[0]`

    device = sd._execution_device

    # 3. Encode input prompt
    if sd.cross_attention_kwargs is not None:
        lora_scale = sd.cross_attention_kwargs.get("scale", None)
    else:
        lora_scale = None

    prompt_embeds, negative_prompt_embeds = sd.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        sd.do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=sd.clip_skip,
    )

    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if sd.do_classifier_free_guidance:
        prompt_embeds = torch.cat(
            [negative_prompt_embeds, prompt_embeds], dim=0,
        )

    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = sd.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            sd.do_classifier_free_guidance,
        )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        sd.scheduler, num_inference_steps, device, timesteps,
    )

    # 5. Prepare latent variables
    num_channels_latents = sd.unet.config.in_channels
    latents = sd.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = sd.prepare_extra_step_kwargs(generator, eta)

    # 6.1 Add image embeds for IP-Adapter
    added_cond_kwargs = (
        {"image_embeds": image_embeds}
        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
        else None
    )

    # 6.2 Optionally get Guidance Scale Embedding
    timestep_cond = None
    if sd.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(sd.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = sd.get_guidance_scale_embedding(
            guidance_scale_tensor,
            embedding_dim=sd.unet.config.time_cond_proj_dim,
        ).to(device=device, dtype=latents.dtype)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * sd.scheduler.order
    sd._num_timesteps = len(timesteps)
    with sd.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if sd.interrupt:
                continue

            # Expand the latents if we are doing classifier free guidance
            if sd.do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            latent_model_input = sd.scheduler.scale_model_input(
                latent_model_input, t,
            )

            # predict the noise residual
            noise_pred = sd.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=sd.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )`[0]`

            # perform guidance
            if sd.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + sd.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if sd.do_classifier_free_guidance and sd.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=sd.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = sd.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)`[0]`

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(sd, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % sd.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(sd.scheduler, "order", 1)
                    callback(step_idx, t, latents)

    if not output_type == "latent":
        image = sd.vae.decode(
            latents / sd.vae.config.scaling_factor,
            return_dict=False,
            generator=generator,
        )`[0]`
        image, has_nsfw_concept = sd.run_safety_checker(
            image, device, prompt_embeds.dtype,
        )
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape`[0]`
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = sd.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload all models
    sd.maybe_free_model_hooks()

    image`[0]`.show()


def lock_net(net):
    for param in net.parameters():
        param.requires_grad = False


import torch.nn as nn
import copy


class ControlNet(nn.Module):
    """
    "We lock (freeze) the parameters Θ of the original block and simultaneously
    clone the block to a trainable copy with parameters Θc. The trainable copy
    takes an external conditioning vector c as input."
    "The trainable copy is connected to the locked model with zero convolution
    layers, denoted $\mathcal{Z}(\cdot; \cdot)$. Specifically,
    $\mathcal{Z}(\cdot; \cdot)$ is a 1 × 1 convolution layer with both weight
    and bias initialized to zeros."
    
    "we use two instances of zero convolutions with parameters Θz1 and Θz2 respectively. The complete ControlNet then computes yc = F(x; Θ) + Z(F(x + Z(c; Θz1 ); Θc ); Θz2 ), (2) where yc is the output of the ControlNet block. In the first training step, since both the weight and bias parameters of a zero convolution layer are initialized to zero, both of the Z(\cdot; \cdot) terms in Equation (2) evaluate to zero, and $y_{c} = y$."
    """
    @staticmethod
    def zero_init_conv(conv):
        conv.weight.data.zero_()
        conv.bias.data.zero_()

    def __init__(self, net, channels):
        super().__init__()

        self.trainable_copy = copy.deepcopy(net)
        lock_net(net)
        self.zero_conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.zero_init_conv(self.zero_conv1)
        self.zero_conv2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.zero_init_conv(self.zero_conv2)

    def forward(self, x, cond):
        return net(x) + self.zero_conv2(
            self.trainable_copy(x + self.zero_conv1(cond)),
        )
net = nn.Identity()
channels = 3
controlnet = ControlNet(net=net, channels=channels)
x = torch.randn((1, 3, 512, 512))
cond = torch.randn((1, 3, 512, 512))
out = controlnet(x, cond=cond)
out.shape
"""
"Text prompts are encoded using the CLIP text encoder [66], and diffusion timesteps are encoded with a time encoder using positional encoding."

"particular, we use ControlNet to create a trainable copy of the 12 encoding blocks and 1 middle block of Stable Diffusion. The 12 encoding blocks are in 4 resolutions (64 × 64, 32 × 32, 16 × 16, 8 × 8) with each one replicated 3 times. The outputs are added to the 12 skip-connections and 1 middle block of the U-net."
"""

# ├── `down_blocks`
# │   ├── `[0]`: 'SD Encoder Block A' (`CrossAttnDownBlock2d`)
# │   │   ├── `resnets`
# │   │   │   └── `[0]`: 'Resnet layer' (`ResnetBlock2D`)
# │   │   ├── `attentions`
# │   │   │   └── `[0]`: 'ViT' (`Transformer2DModel`)
# │   │   ├── `resnets`
# │   │   │   └── `[1]` 'Resnet layer' (`ResnetBlock2D`)
# │   │   ├── `attentions`
# │   │   │   └── `[1]`: 'ViT' (`Transformer2DModel`)
# │   │   └── `downsamplers`:
# │   │       └── `[0]`: 'Down-sampling convolution layer' (`Downsample2D`)
# │   ├── `[1]`: 'SD Encoder Block B' (`CrossAttnDownBlock2d`)
# │   │   ...
# │   ├── `[2]`: 'SD Encoder Block C' (`CrossAttnDownBlock2d`)
# │   │   ...
# │   └── `[3]`: 'SD Encoder' (`DownBlock2D`)
# │       └── `resnets` # 11
# ├── `mid_block`: 'SD Middle' (`UNetMidBlock2DCrossAttn`)
#     ...
# └── `up_blocks`
#     ...


unet = sd.unet
len(unet.down_blocks[0].attentions)
unet.config.layers_per_block
unet.config.transformer_layers_per_block

unet.conv_in
unet.time_proj
unet.time_embedding
# [i`[0]` for i in list(unet.down_blocks`[0]`.named_children())]

unet.mid_block
unet.up_blocks