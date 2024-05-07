---
title: 扩散模型（四）| Imagen
date: 2024-01-11
mathjax: true
cover: false
category:
 - Diffusion model
tag:
 - Imagen
---

> Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding 
>
> Google-https://arxiv.org/pdf/2205.11487
>
> 参考：https://www.zhangzhenhu.com/aigc/imagen.html#

## 创新

（1）在text-only语料数据上训练的大语言模型（如T5）对text-to-image生成有非常重要的影响，增大大语言模型的size比增大图像diffusion模型更能提高生成样本质量。

（2）引入动态阈值，可以让diffusion模型利用更高的guidance权重生成更现实主义和细节的图像。

（3）提出Efficient U-Net架构，使模型更简洁、收敛更快、显存占用更小。

## 概览

Imagen，一个text-to-image diffusion模型，同时结合大语言模型和扩散模型的优势。

Imagen架构如下图所示：使用一个frozen text encoder（T5-XXL）来将输入文本编码成text embeddings。一个条件扩散模型将text embedding生成64x64图片，之后利用超分模型上采样图像，从64x64->256x256，之后256x256->1024x1024。全部的diffusion模型都以文本嵌入序列为条件，使用classifier-free guidance。Imagen依靠新的采样技术（dynamic thresholding），在不降低样本质量的情况下使用大的指导权重，生成图像质量更好。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/11-diffusion-models/0.jpg" alt="img" style="zoom:80%;" />

## 方法

### Pretrained text encoders

当前的text-to-image模型中的文本编码器经常在成对的image-text数据上训练（如CLIP）。大语言模型是Text encoder的另一种选择（如BERT、GPT、T5）。大语言模型在大量的纯Text语料上训练，相比成对的image-text数据，其能接触到更丰富和广泛分布的文本（生成文字的效果更好，对文字感知能力更强）。

Imagen探索了几种预训练的text encoders：BERT、T5、CLIP。冻结这些text encoders的权重进行训练。实验发现：

（1）缩放text encoder size可以提升text-to-image生成质量（图5(a)中T5模型越大，FID越低，CLIP分数越高）。

（2）当T5-XXL和CLIP在简单的benchmark如MS-COCO上的FID和CLIP评价指标相似时，人类的评估通常更偏向T5-XXL（图5(a)中CLIP和T5-XXL的分数类似，但是图5（b）中可以看出T5-XXL在文本图像对齐和图像保真度方面分数都比CLIP要高）。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/11-diffusion-models/1.jpg)

（3）缩放text encoder size比缩放U-Net size更重要。如下图所示，缩放text encoder比缩放U-Net的FID分数变化更明显。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/11-diffusion-models/2.jpg)

### Large guidance weight samplers

增加classifier-free guidance权重提升了image-text对齐效果，但是破坏了图像保真度，容易产生高度饱和和不自然的图像。这是由于高CFG权重导致的train-text mismatch。

在每个采样步t，预测出的图像 $\hat x_0^t$必须和训练数据 $x$在相同范围内，如[-1,1]，然而经验性的高的guidance weights造成$\hat x_0^t$超出了这个界限，导致train-test不匹配。由于扩散模型在整个采样过程中反复应用其自身的输出，采样过程会产生不自然的图像。为了解决这个问题，引入static thresholding和dynamic thresholding。

- Static thresholding

  通过裁剪直接将$\hat x_0^t$控制到[-1,1]。这种方法在之前的工作（DDPM）中就有被使用，但是未被重点强调。静态阈值对于具有大引导权重的采样至关重要，并且可以防止生成空白图像。尽管如此，随着引导权重的进一步增加，静态阈值仍然会导致图像过度饱和和细节不足。

- Dynamic thresholding

  引入了一种新的动态阈值化方法：在每个采样步骤，将s设置为 $\hat x_0^t$的某个百分位数对应的值，如果s>1，则将所有数阈值化到范围[−s，s]，然后除以s。动态阈值处理将饱和像素（接近-1和1的像素）向内推，从而防止像素在每一步饱和。（比如数据为[1,2,3,4,5]，百分位数为80%，则s为4，数据首先根据阈值4进行裁剪，变为[1,2,3,4,4]，之后除以4，变成[0.24,0.5,0.75,1,1]；如果采用static thresholding，则数据变为[1,1,1,1,1]）

  ![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/11-diffusion-models/3.jpg)

  实验发现，当使用非常大的指导权重时，动态阈值可以显著改善照片真实性以及实现更好的图像-文本对齐。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/11-diffusion-models/4.jpg)

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/11-diffusion-models/5.jpg)

### Robust cascaded diffusion models

使用带有**噪声条件增强**的级联扩散模型对于生成高质量图像是有效的。

给定一个低分辨率图像和增强水平（aug_level），使用此增强来破坏低分辨率图像，在破坏后的图像上进行diffusion。训练过程中,aug_level被随机选择；推理过程中，使用不同aug_level值进行生成，找到最佳的样本。Imagen使用高斯噪声作为增强，$aug\\_level \in [0,1]$。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/11-diffusion-models/6.jpg)

实验表明，具有noise conditioning augmentation的超分模型产生更好的CLIP和FID得分。在推理阶段向低分辨率图像中添加噪声+使用大的guidance权重允许超分模型生成更多样的上采样结果，同时移除了低分辨率图像中的量化伪影。

### Neural network architecture

- base model

  应用U-Net架构作为base 64x64 text-to-image diffusion model。

  Text condition选择cross attention。如下图所示，Mean pooling, attention pooling, cross attention三种条件注入方式中cross attention效果最好。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/11-diffusion-models/7.jpg)

- Super-resolution model

  对U-Net模型做了一些改动，使其节省内存、训练收敛更快、推理时间更短，改进后模型为Efficient U-Net。

  Efficient U-Net保持text cross-attention层不动，移除了self-attention层。

## Evaluating Text-to-Image Models

- FID：image fidelity
- CLIP score：image-text alignment

因为guidance weight对于控制图像质量和text alignment来说是重要的，大部分消融实验的结果使用不同guidance weights下的CLIP和FID曲线来呈现。

- Human evaluation
  - Image quality：which image is more photorealistic? We report the percentage of times raters choose model generations over reference images (the reference rate)
  - Alignment: Does the caption accurately describe the above image? Respond with "yes","somewhat","no"

## Experiments

- Results on COCO

  Imagen outperforming the concurrent work of DALL-E2 and even models trained on COCO.

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/11-diffusion-models/8.jpg)

- Results on DrawBench

  Imagen和DALL-E2、GLIDE、Latent Diffusion、CLIP-guided VQ-GAN比较结果：

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/11-diffusion-models/9.jpg)