---
title: 扩散模型（八）| 3D Diffusion：PointE
date: 2024-01-24
mathjax: true
cover: false
category:
 - Diffusion model
---

> PointE: A system for Generating 3D Point Clouds from Complex Prompts
>
> https://arxiv.org/pdf/2212.08751.pdf
>
> https://github.com/openai/point-e

## Abstract

最近的text-conditional 3D生成模型经常需要多个GPU-hours来生成一个单一样本，这和图像生成模型（几秒或几分钟就可以生成样本）相比有巨大的差距。本文提出一个3D生成模型PointE（generate Point clouds Efficiently)，可以用单个GPU在1-2分钟内生成3D模型。

该方法首先用一个text-to-image diffusion模型生成一个单一的合成视角的图像，之后用另一个输入条件为图像的diffusion模型生成3D点云。

该方法虽然质量上和sota模型有差距，但是速度比其他的快1-2个数量级。

## Introduction

最近的text-to-3D方法大致分成2类：

- 使用成对的数据（text，3D）或者没有标签的3D数据训练生成模型。这些方法利用现有的生成模型方法来高效生成样本，但是它们面对生成多样性和复杂的文本Prompt是困难的，因为目前缺乏大规模3D数据集。
- 利用预训练的text-image模型来优化3D表示。这些方法可以处理复杂多样的text prompts，但是需要昂贵的优化过程来生成样本。此外，由于缺乏3D先验知识，这些方法容易陷入局部最优，不能生成有意义或连贯的三维对象。

本文试图结合两种方法的优势，通过将text-to-image模型和image-to-3D模型结合起来。其中，text-to-image模型利用大量的(text,image)语料对，允许模型根据多样复杂的prompt进行生成；image-to-3D模型在一个更小的(image,3D)数据集上训练。为了根据text prompt生成一个3D物体，首先使用text-to-image模型采样一个图像，之后根据这个采样的图像使用image-to-3D模型采样一个3D object。

## Method

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/0.jpg)

1. 通过一个text caption生成一个合成视角的图像 <- 3-billion参数的在渲染的3D模型数据集上微调的GLIDE模型
2. 通过合成视角的图像生成一个粗粒度的点云（1024points) <-具有置换不变性的diffusion model
3. 在低分辨率点云和合成视角图像条件下进行进一步生成，生成一个细粒度的点云（4096points）<-和第二个diffusion模型相似但是更小的扩散模型

### Synthesis GLIDE Model

为了确保text-to-image模型可以生成正确的合成视角，训练模型让模型可以生成满足点云训练数据集分布的3D渲染。

为此，使用GLIDE初始训练数据集和3D渲染数据集微调GLIDE模型。因为3D渲染数据集相比于GLIDE训练数据集小很多，仅仅只在5%的时间内对3D数据集的图像进行采样，其余95%使用原始数据集。微调100K个迭代，意味着模型已经在整个3D数据集上训练了几个epochs。

在测试时，为了确保我们总是在3D分布渲染中采样，我们在每个3D渲染的文本提示中添加一个特殊的标记，指示它是3D渲染；然后在测试时使用此token进行采样。

### Point Cloud Diffusion

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/1.jpg" alt="img" style="zoom:50%;" />

将点云表示成一个tensor，shape为K x 6，K是点云中点的数量，特征维度为(x,y,z,R,G,B)。全部坐标和颜色被标准化为[-1,1]。输入随机噪声K x 6，通过diffusion模型逐步denoising，生成点云。

利用简单的Transformer-based模型基于输入图像、时间步t、噪声点云 $x_t$来预测噪声 $\epsilon$和方差 Σ。 

- 条件编码
  - t：通过一个小的MLP，获得一个D维向量
  - 噪声点云：每个点通过一个linear层，得到K x D
  - 图像：使用一个预训练的ViT-L/14 CLIP模型提取特征，选择最后一层的特征嵌入（256 x $D'$），线性投影成256 x D

所有条件编码在batch维度concat，构成transformer的输入：(K+257)xD。为了获得长度为K的最终输出序列，我们取输出的最后K个标记并将其投影，以获得K个输入点的ε和∑预测。

注意，因为在整个过程中都没有使用位置编码，所以模型满足置换不变性。

### Point Cloud Upsampler

upsampler模型采用和base模型相同的架构。对于低分辨率的point cloud采用另外的condition tokens，将条件点云通过一个分离的线性嵌入层，使模型可以区分高分辨率点云和低分辨率点云的信息。

## Results

### Qualitative Results

PointE可以基于复杂的Prompts生成高质量的3D shapes。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/2.jpg" alt="img" style="zoom:50%;" />

失败的例子：

1. 错误地理解不同部分的相对比例，导致生成了一个高的狗而非短的长的狗。
2. 不能推理出被遮挡的部分

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/3.jpg" alt="img" style="zoom:50%;" />

### Model Scaling and Ablations

1. 仅仅使用text conditioning，而没有text-to-image步骤导致模型产生更差的结果
2. 使用一个单一的token编码图像CLIP embedding比使用多个token编码CLIP embedding产生更差的结果
3. 扩大模型可以加速收敛，增大CLIP R-Presicion结果。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/4.jpg" alt="img" style="zoom:67%;" />