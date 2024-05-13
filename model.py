import torch
from diffusers import DiffusionPipeline

# device = torch.device("cuda")
device = torch.device("cpu")
sd = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
sd = sd.to(device)

text_tokenizer = sd.tokenizer
text_enc = sd.text_encoder
vae = sd.vae
unet = sd.unet

prompt = "Stable Diffusion’s U-net architecture connected with a ControlNet on the encoder blocks and middle block."
text_tokenizer(prompt)
tokenized = text_tokenizer(prompt)
input_ids = torch.tensor(tokenized["input_ids"], device=device)[None, ...]
attn_mask = torch.tensor(tokenized["attention_mask"], device=device)[None, ...]
text_enc_out = text_enc(input_ids, attention_mask=attn_mask)
img_size = 512
noisy_image = torch.randn((1, 3, img_size, img_size)).to(device)
# vae_out = vae(noisy_image)
vae_enc_out = vae.encoder(noisy_image)

unet_out = unet(
    sample=vae_enc_out,
    timestep=50,
    encoder_hidden_states=text_enc_out.last_hidden_state,
)
unet

out = sd(prompt)
out.images[0].show()
out.images[0].size