import torch
from diffusers import DiffusionPipeline

# device = torch.device("cuda")
device = torch.device("mps")
sd = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32,
)
sd = sd.to(device)

text_tokenizer = sd.tokenizer
text_enc = sd.text_encoder
vae = sd.vae
unet = sd.unet
sd.scheduler.timesteps

prompt = "Stable Diffusion’s U-net architecture connected with a ControlNet on the encoder blocks and middle block."
tokenized = text_tokenizer(prompt)
input_ids = torch.tensor(tokenized["input_ids"], device=device)[None, ...]
attn_mask = torch.tensor(tokenized["attention_mask"], device=device)[None, ...]
text_enc_out = text_enc(input_ids, attention_mask=attn_mask)
img_size = 512
noisy_image = torch.randn((1, 3, img_size, img_size)).to(device)

posterior = vae.encode(noisy_image).latent_dist
z = posterior.mode()

unet_out = unet(
    sample=z,
    timestep=50,
    encoder_hidden_states=text_enc_out.last_hidden_state,
)
unet_out

out = sd(prompt)
out.images[0].show()
out.images[0].size