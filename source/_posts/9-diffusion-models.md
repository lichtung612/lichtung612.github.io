---
title: 扩散模型（九）| Transformer-based Diffusion：U-ViT
date: 2024-01-30
mathjax: true
cover: false
category:
 - Diffusion model
tag:
 - Transformer-based diffusion
---

>  《All are Worth Words: A ViT Backbone for Diffusion Models》-23CVPR- https://arxiv.org/pdf/2209.12152.pdf
>
> 代码：https://github.com/baofff/U-ViT

## 总体介绍

***背景：***ViT在各种视觉任务中取得优异效果，而基于CNN的U-Net模型依然在扩散模型中占据主体地位。一个很自然的问题：在扩散模型中，ViT能否代替基于CNN的U-Net？ 

***模型架构：***在本论文中，我们设计了一个简单通用的**基于ViT架构的**图像生成扩散模型U-ViT：

- 把包括时间、条件、噪声所有的输入视为tokens
- 在浅层和深层之间应用长距离跳跃连接（因为图像生成是一个pixel-level预测任务，对低层级特征敏感。长跳跃连接提供低层次的特征，使模型更容易训练）
- 在输出前添加3x3卷积块，以获得更好的视觉质量

***实验：***我们在无条件图像生成、类别条件图像生成和text-to-image生成三种类型任务上评估U-ViT，实验结果表明：

- U-ViT即使不优于类似大小的基于CNN的U-Net,也具有可比性。特别地，在不访问大型外部数据集的方法中，U-ViT在ImageNet 256x256 class-conditional生成任务中取得破纪录的FID 2.29分，MS-COCO text-to-image生成任务中取得5.48分。
- 对于扩散模型图像建模，长距离跳跃连接是至关重要的，而基于CNN的U-Net中上采样和下采样操作不是必要的。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/8-diffusion-models/0.jpg" alt="img" style="zoom:50%;" />

## 模型架构设计

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/8-diffusion-models/1.jpg)

如上图所示为不同的设计方案，在进行实验后U-ViT选择了各种方案中FID分数最好的方案，带*号表示U-ViT的选择。

- **Long skip connection:** 对于网络主分支的特征和来自跳跃连接的特征 $h_m,h_s \in R^{L\times D}$，U-ViT选择将它们concat起来，之后执行线性投影，即 $Linear(Concat(h_m,h_s))$。
- **Feed time into the network：**有2种方案，一种是直接将时间t视为token，一种是类似于adaptive group normalization，在LayerNorm后插入时间t，即 $AdaLN(h,t)=t_sLayerNorm(h)+t_b$，h是transformer block的特征， $t_s,t_b$是 $t$经过线性投影后的特征。实验发现直接将时间t视为token效果更好。
- **在Transformer后添加额外卷积块：**有3种方案，一种是在线性投影后添加3x3卷积块，一种是线性投影前添加3x3卷积块，一种是不添加卷积块，实验发现在线性投影后添加卷积块效果更好。
- **Patch embedding：**有2种方案，方案一是采用线性投影来做token embedding，方案二是用一个3x3卷积块+1x1卷积块来做token embedding。实验结果表明方案一更好。
- **Positional embedding：**有2种方案，方案一是和原始ViT一样，采用一维可学习的位置嵌入，方案二是采用2维正弦位置嵌入（concat像素（i,j）的sinusoidal embeddings）。实验结果表明方案一更好。本文也尝试不使用任何的位置嵌入，发现模型不能生成有意义的图像，可见位置编码在图像生成中的重要性。

## 缩放能力

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/8-diffusion-models/2.jpg)

上图通过缩放深度（层数）、宽度（隐藏层的维度）、patch size探究了U-ViT的缩放能力。

- Depth(#layers)：当深度增加到17在第500k迭代时模型性能不再提升
- Width(hidden size)：宽度从256提高到512，模型性能提升；但是从512提高到768时性能不再提升
- Patch size: 减小patch size提升了模型性能。一个小的patch size=2拥有很好的表现。作者推测因为噪声预测任务是低层次的，因此需要更小的patches（不同于需要高层次语义特征的分类任务）。因为对高分辨率图像来说采用较小的patch size很消耗资源，所以我们首先将图像转换到低维度潜在空间中，之后用U-ViT建模潜在特征表示。

## 同等参数量和计算量的效果对比

同等参数量和计算量下（U-ViT：501M parameters,133 GFLOPs；U-Net：646M parameters,135 GFLOPs），在classifier-free guidance的情况下U-ViT在不同的训练迭代中始终优于U-Net。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/8-diffusion-models/3.jpg)

## 实验

### Unconditional and Class-Conditional Image Generation

#### FID得分

从下表可以看出，U-ViT在图像无条件生成以及类别条件生成上取得了和其他模型可比或者更优的FID。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/8-diffusion-models/4.jpg)

#### 潜在空间建模性能

U-ViT在ImageNet256数据集上取得了SOTA的FID，可以发现其在潜在空间性能更好。和用U-Net建模特征空间的[Latent Diffusion](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2112.10752) 相比，在使用相同采样器（dpm_solver）和相同采样步数的情况下，U-ViT均能取得更优的表现。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/8-diffusion-models/6.jpg)

### Text-to-Image Generation on MS-COCO

#### FID得分

U-ViT展现了杰出的多模态融合能力，在没有额外数据的情况下，U-ViT取得了MS-COCO数据集上text-to-image generation任务的SOTA FID。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/8-diffusion-models/7.jpg)

#### 图像与文本匹配质量更高

如下图显示了U-Net和U-ViT使用相同随机种子生成的样本，发现U-ViT生成了更多高质量的样本，同时语义与文本匹配得更好。例如，给定文本“棒球运动员向球挥动球棒”，U-Net既不生成球棒也不生成球。相比之下，U-ViT-S在更少的训练参数下可以生成球，而我们的U-ViT-S（Deep）更近一步把球和球棒都生成出来。我们假设这是因为文本和图像在U-ViT的每一层都有交互，这比只在cross attention层交互的U-Net更频繁。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/8-diffusion-models/8.jpg)
