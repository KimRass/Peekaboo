# text_tokenizer = sd.tokenizer
# text_enc = sd.text_encoder
# vae = sd.vae
# unet = sd.unet
# scheduler = sd.scheduler
# scheduler.set_timesteps(num_inference_steps=50)

# tokenized = text_tokenizer(prompt)
# input_ids = torch.tensor(tokenized["input_ids"], device=device)[None, diffusers.]
# attn_mask = torch.tensor(tokenized["attention_mask"], device=device)[None, diffusers.]
# text_enc_out = text_enc(input_ids, attention_mask=attn_mask)

# out = sd.encode_prompt(
#     prompt,
#     num_images_per_prompt=1,
#     do_classifier_free_guidance=True,
#     device=device,
# )
# out[0].shape
# out[1].shape

# # vae_enc_out = vae.encode(rand_noise)
# # posterior = vae_enc_out.latent_dist
# # z = posterior.mode()

# rand_noise = torch.randn(
#     (
#         1,
#         unet.config.in_channels,
#         unet.config.sample_size,
#         unet.config.sample_size,
#     ),
#     dtype=torch.float16,
# ).to(device)
# z = rand_noise
# for i, timestep, in enumerate(tqdm(scheduler.timesteps)):
#     with torch.no_grad():
#         unet_out = unet(
#             sample=z,
#             timestep=timestep,
#             encoder_hidden_states=text_enc_out.last_hidden_state,
#         )
#         # noisy_resid = unet_out.sample
#         # scheduler_step_out = scheduler.step(
#         #     model_output=noisy_resid, timestep=timestep, sample=z,
#         # )
#         # z = scheduler_step_out.prev_sample
#         z = unet_out.sample
# dec_out = vae.decode(z)
# display_sample(dec_out.sample.detach(), i)