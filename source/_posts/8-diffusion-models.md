---
title: 扩散模型（八）| Image-Conditioned T2I：ControlNet
date: 2024-01-19
mathjax: true
cover: false
category:
 - Diffusion model
tag:
 - Image-Conditioned T2I
---

> Adding Conditional Control to Text-to-Image Diffusion Models-2302
>
> https://arxiv.org/pdf/2302.05543.pdf
>
> https://github.com/lllyasviel/ControlNet

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/6-diffusion-models/0.jpg)

## 概要

1. 一种高效微调text-to-image diffusion models的方法，可以让diffusion model学习空间条件注入信息。
2. ControlNet冻结stable diffusion的参数，复用它的encoding layers来训练，其中复用的encoding layers的参数为零初始化（"zero convolutions", zero-initialized convolution layers)，确保没有有害的噪声影响微调。
3. 在输入粗糙的边缘、pose、深度、草图、语义图等条件下都可以得到满意的效果。

## 方法

### ControlNet

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/6-diffusion-models/1.jpg" alt="img" style="zoom:50%;" />

初始输入feature map: $x$，输出feature map: $y$

初始网络： $y = F(x;\theta)$

冻结初始网络 $\theta$，复制这个block成一个新的block，新block的参数为 $\theta_c$。新的block增加一个额外的条件输入 $c$。

trainable copy被zero convolution层连接，记为 $Z(,;,)$，一般来说$Z(,;,)$是一个1x1卷积层，weight和bias初始化为0。

新的网络： $y_c = F(x;\theta)+Z(F(x+Z(c;\theta_{z1});\theta_c);\theta_{z2})$

初始情况下 $y_c = y$，这样保证初始状态没有引入有害噪声。

### ControlNet with Stable Diffusion

#### Stable Diffusion

Stable Diffusion采用U-Net架构。U-Net由一个encoder，一个middle block，一个包含跳跃连接的decoder组成。encoder和decoder都包含12个block，整个模型有25个block。其中，8个block是降采样或者升采样层，其余17个block中每个包含4个resnet层和2个ViT层，其中每个ViT层包含cross-attention和self-attention机制。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/6-diffusion-models/2.jpg" alt="img" style="zoom:50%;" />

如图所示，“SD Encoder Block A"包含4个resnet和2个ViT，“x3”表示这个Block重复了3次。

Text prompts使用CLIP text encoder编码；时间步使用正余弦位置编码来编码。

ControlNet结构被应用在U-net的每一个encoder上。具体来说，创造一个stable diffusion的12个encoding blocks和1个middle block的trainable copy。输出被添加上12个skip-connetction和1个middle block。

条件 $c_f$的编码设计：因为stable diffusion是在Latent空间进行操作的（512x512 pixel-space->64x64 latent images)，所以首先将输入的条件从512x512变成64x64。使用一个具有4层卷积的网络，每一层为4x4 kernels,2x2 strides（ReLU激活函数，16, 32, 64, 128 channels，高斯权重初始化，和整个模型一起训练），将在图像空间中的条件 $c_i$编码成特征空间中的向量 $c_f$。

在NVIDIA A100 40G测试表明，优化带有ControlNet的Stable Diffusion仅仅需要比原始stable diffusion增加23%的GPU内存和34%的时间消耗。

#### Training

在训练过程中随机使用空字符串替换50%的text prompts $c_t$。这样可以增加ControlNet直接识别空间条件语义信息的能力。

训练过程中存在“sudden convergence phenomenon"（模型不是逐渐地学习控制条件，而是在某些步骤忽然get到控制条件，通常小于10K个steps）：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/6-diffusion-models/3.jpg" alt="img" style="zoom:67%;" />

#### Inference

**Classifier-free guidance resolution weighting**

CFG(Classifier-Free Guidance)公式表示为 $\epsilon_{prd} = \epsilon_{uc}+\beta_{cfg}(\epsilon_c-\epsilon_{uc})$，其中 $\epsilon_{prd},\epsilon_{uc},\epsilon_c,\beta_{cfg}$分别表示模型最终的输出、无条件输出、有条件输出、用户定义的权重。

当一个空间条件图片被添加到ControlNet，它可以被添加到 $\epsilon_{uc}$和 $\epsilon_c$上，或者仅仅被添加到 $\epsilon_c$上。在一些有挑战性的场景，比如当没有prompts的时候，把空间条件图片在 $\epsilon_{uc}$和 $\epsilon_c$上将完全移除CFG guidance（图5b)；仅仅在$\epsilon_c$上添加将使该指导特别强烈（图5c）；解决方式是首先添加条件图片到 $\epsilon_c$上，之后在stable diffusion和ControlNet之间的每个连接上乘以权重 $w_i=64/h_i$，其中 $h_i$是第i个block的size，如 $h_1=8,h_2=16,...,h_{13}=64$。通过减小CFG指导强度，可以完成图5d的效果。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/6-diffusion-models/4.jpg" alt="img" style="zoom:80%;" />

**Composing multiple ControlNets**

为了同时应用多个条件图片注入，可以直接把ControlNets的输出添加到stable diffusion里。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/6-diffusion-models/5.jpg" alt="img" style="zoom:67%;" />

## 代码

>   https://github.com/huggingface/diffusers/blob/v0.27.2/src/diffusers/pipelines/controlnet/pipeline_controlnet.py

原本sd模型使用`UNet2DConditionModel`类；添加的trainable copy部分使用`ControlNetModel`类，它提取condition_images的特征插入原本的UNet模型中。

### CFG推理过程

#### 输入有prompt

- 右侧的controlnet模型输入噪声图control_model_input = latent_model_input(2倍batch_size，2份latents)，文本嵌入controlnet_prompt_embeds = prompt_embeds（无条件prompt+有条件prompt)，**条件图controlnet_cond=image(2倍batch_size，相当于输入条件图concat起来：`image = torch.cat([image]*2)`)**，时间步t。

  输出要和unet网络残差连接的down_block_res_samples和mid_block_res_sample。

    ```Python
    down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )
    ```

- 左侧的unet模型输入类似，噪声图latent_model_input（2*batch)，文本嵌入prompt_embeds(2*batch)，时间步t，**controlnet的输出**（down_block_additional_residuals，mid_block_additional_residual）。

  输出预测的噪声noise_pred(2*batch)。

    ```Python
    # predict the noise residual
    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
    ```

#### 输入没有prompt

没有prompt时，根据Inference章节，仅仅把空间条件图像添加到controlnet的有条件的输入部分，无条件部分没有空间条件图像。

- 右侧的controlnet模型输入为单个batch：输入噪声图control_model_input = latents（1份latents），controlnet_prompt_embeds=prompt_embeds.chunk(2)[1]（有条件部分的prompt），空间条件图image（1份batchsize），时间步t
  -  输出为down_block_res_samples，mid_block_res_sample。因为输出为一个batchsize，而unet对应的为2*batchsize，所以输出的前面部分concat上全0特征。

   ```Python
     #guess_mode=True表示输入空prompt的情况
    if guess_mode and self.do_classifier_free_guidance:
        # Infer ControlNet only for the conditional batch.
        control_model_input = latents
        control_model_input = self.scheduler.scale_model_input(control_model_input, t)
        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
    
    if guess_mode and self.do_classifier_free_guidance:
        # Infered ControlNet only for the conditional batch.
        # To apply the output of ControlNet to both the unconditional and conditional batches,
        # add 0 to the unconditional batch to keep it unchanged.
        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
   ```
- perform guidance

    ```Python
    if self.do_classifier_free_guidance:
        noise_pred_uncond,noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
    ```
