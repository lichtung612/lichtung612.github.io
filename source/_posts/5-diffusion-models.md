---
title: 扩散模型（五）| Stable Diffusion
date: 2024-01-14
mathjax: true
cover: false
category:
 - Diffusion model
tag:
 - Stable Diffusion
---

>  High-Resolution Image Synthesis with Latent Diffusion Models
>
> https://arxiv.org/pdf/2112.10752.pdf
>
> https://github.com/CompVis/latent-diffusion

## 动机

1. Diffusion models(DMs)通常在像素空间操作，因此训练DMs消耗巨大的GPU资源，基于序列化生成的推理过程代价也比较高。为了使DM能够在有限的计算资源下训练，并且保持优越的生成质量和灵活性，本文将DM**应用在预训练好的自编码器的潜在空间中**，相比pixel-based DM**极大地减小计算需求**。并且由于是在潜在空间训练模型，它在空间维度方面展现更好的**缩放特性**。
2. 引入**cross-attention**层，使得模型可以更灵活的根据不同类型条件输入进行图像生成。

## 方法

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/4-diffusion-models/0.jpg" alt="img" style="zoom:80%;" />

## 代码

**diffusers库实现**

>   https://github.com/xiaohu2015/nngen/blob/main/models/stable_diffusion/stable_diffusion.ipynb

- stable diffusion默认生成512*512图像；如果想生成其他size图像，最好height和width能被8整除，一个维度还是使用512，另一个维度比512更大（如果两个维度都比512小，导致低质量；如果均比512大，可能会生成有重复图像区域的图像）。

- 推理流程
  - Tokenize and encode the text
    - tokenizer：根据一个vocab.json表，把文本编码成token数字和attention_mask矩阵。比如输入prompt为“A dragon fruit wearing karate belt in the snow”，则经过tokenizer编码：
      
      ![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/4-diffusion-models/1.jpg)
      
      ![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/4-diffusion-models/2.jpg)
      
      ![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/4-diffusion-models/3.jpg)
      
      其中tokenizer过程中一个词不一定代表一个token，当这个词在vocab中没有出现的时候它会对其进行分词，所以有可能一个词对应两个单词。空格字符也是token的一部分。stable diffusion限制最大token数是77。
      
    - text_encoder
    
       特征嵌入维度为768，得到的text_embeddings：[B,77,768]
    
       ```python
       with torch.no_grad():
           text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
       ```

    - Classifier-free guidance
    
       无条件的embeddings和有条件的embeddings拼接得到训练样本（batch维度拼接），相当于本来是batch个样本，现在为batch*2个样本。
    
       ```python
       # Get the text_embeddings for the prompt
       text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
       with torch.no_grad():
           text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
       # Get the unconditional text embeddings for classifier-free guidance
       max_length = text_input.input_ids.shape[-1]
       uncond_input = tokenizer(
           [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
       )
       with torch.no_grad():
             uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
       # Concatenate both into a single batch
       text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
       ```
    
    - 生成latent空间下噪声（比原图缩小8倍）
    
       Stable diffusion模型的latent space为 $4 \times 64\times 64$，比图像像素空间小48倍。
    
       ```python
       # Generate the intial random noise
       latents = torch.randn(
           (batch_size, unet.in_channels, height // 8, width // 8), #unet.in_channels=4
           generator=generator, device=device
       )
       ```
    
    - 推理代码
    
       ```python
       # Set inference steps for the noise scheduler
       noise_scheduler.set_timesteps(num_inference_steps)
       
       # scale the initial noise by the standard deviation required by the scheduler
       latents = latents * noise_scheduler.init_noise_sigma # for DDIM, init_noise_sigma = 1.0
       
       # It's more optimized to move all timesteps to correct device beforehand
       timesteps_tensor = noise_scheduler.timesteps.to(device)
       
       # Do denoise steps
       for t in tqdm(timesteps_tensor):
           # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
           latent_model_input = torch.cat([latents] * 2)
           latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t) # for DDIM, do nothing
       
           # predict the noise
           with torch.no_grad():
               noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
       
           # perform guidance
           noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
           noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
       
           # compute the previous noisy sample x_t -> x_t-1
           latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
           
       # scale and decode the image latents with vae
       latents = 1 / 0.18215 * latents
       
       with torch.no_grad():
           image = vae.decode(latents).sample
       ```
    
       
