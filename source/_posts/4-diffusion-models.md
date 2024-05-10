---
title: 扩散模型（四）| DALL-E2(unCLIP)
date: 2024-01-09
mathjax: true
cover: false
category:
 - Diffusion model
tag:
 - DALLE
---

>  OpenAI - Hierarchical Text-Conditional Image Generation with CLIP Latents
>
> 论文：https://arxiv.org/pdf/2204.06125
>
> 参考：
>
> 1. https://blog.csdn.net/xiqiao_ce/article/details/132769442
> 2. https://www.bilibili.com/video/BV17r4y1u77B/?spm_id_from=333.999.0.0&vd_source=f16d47823aea3a6e77eec228119ddc27
> 3. https://zhuanlan.zhihu.com/p/394467135
> 4. https://zhuanlan.zhihu.com/p/648389187

## 文生图模型时间线

上面是闭源相关，下面是开源相关模型。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/13-diffusion-models/0.jpg)

## 历史生成式方法

### GAN

GAN包含生成器(Generator)和判别器(Discriminator)。

生成器根据随机噪音z生成图片x'；生成图片x'和真实图片x被送入判别器中，判别是真图片还是假图片。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/13-diffusion-models/1.jpg)

训练的时候先固定生成器，只训练判别器，此过程只更新判别器参数，不更新生成器参数；之后训练生成器，此过程只更新生成器参数，不更新判别器参数。训练过程二者相互对抗，交替训练。

GAN模型优势：

（1）生成图像的保真度高；

（2）推理时间快

GAN模型缺点：

（1）因为同时要训练2个网络，训练不平衡，容易训练坍塌；

（2）因为对图像真实度要求高，反而生成图像的多样性、创造性差；

（3）数学理论上不够好，黑盒模型。

### VAE

- Autoencoder

  输入x，经过encoder得到特征z，z的特征维度一般很小，z经过decoder得到图像x'。训练目标希望x'尽可能和x接近。因为是自己重建自己的过程，所以是自编码器。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/13-diffusion-models/2.jpg)

- Denoising autoencoder

  输入x，首先对x加一定噪声变成 $x_c$（corrupted x），之后把 $x_c$传入encoder得到z，z传入decoder得到x'。训练目标同AE，希望x'尽可能接近x。

  DAE比AE效果好，依据原理是因为图像的像素的冗余性太高，尽管对之前图像做一些污染，仍能重建出原图，并且生成图像的多样性有所提高。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/13-diffusion-models/3.jpg)

- VAE(Variational Auto-encoder)

  无论是AE还是DAE，它们的主要目的都是为了去学习中间的bottleneck特征，把中间的特征拿去做分类、检测、分割等任务，不是用来做生成任务的。因为它学习到的中间的特征不是一个概率分布，没办法对它进行采样。它不像GAN里面的z是一个符合某种分布的随机噪声。

  VAE中间不再是学习一个bottleneck特征了，而是去学习一个分布。假设这个分布是一个高斯分布。当我们得到从编码器出来的特征，在后面加一些FC层，去预测一个均值和方差。得到对应的均值和方差之后，用公式 $z=u+\sigma\epsilon$采样出一个噪声z出来，z经过decoder得到x'。训练好模型后，直接把编码器部分扔掉，随机采样噪声z，就可以生成图像。

  因为VAE学习的是一个概率分布，生成图片的多样性比GAN要好很多，且训练更稳定。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/13-diffusion-models/4.jpg)

### VQVAE

VQ-VAE中VQ指vector quantised，即把VAE做离散化。现实中大部分信号，比如图像、声音等都是连续的，任务都是回归任务。但是当把它们表示出来，图像变成像素，语音抽样化，效果更好的模型是分类模型，计算机更擅长处理离散的东西。VAE中连续的分布不太好学习，所以VQVAE取而代之，不去做分布的推测，而是使用一个离散的codebook来代替。

codebook可以理解为聚类的中心，大小为KxD。K一般为8192，D为512或者768，即有8192个长度为D的向量，8192个聚类中心。如果有一个图片经过encoder得到一个特征图，把特征图里的向量和codebook里的向量做对比，看它和哪个聚类中心最接近，把最接近的聚类中心的index存到z里，根据z中的index和codebook中的特征生成新的特征图 $f_q$，$f_q$此时不再是随机的东西，它永远来自codebook里的特征。

生成目标还是希望生成图像x'和输入图像x一致。

因为中间特征不是一个分布，而是一个代表前面图像信息的从codebook里面取出的压缩特征，所以VQVAE更适合用于分类、检测、分割任务，而非生成，VQVAE更类似AE，而不是VAE。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/13-diffusion-models/5.jpg)

### DALL-E

DALL-E的输入是text和image。

text首先经过BPE编码，得到256个token（token数不满256的话padding到256）。

image使用VQVAE,把256x256图像压缩成32x32图片token。(dVAE：discrete VAE，和VQVAE类似一个东西）

256个text token和1024个图像token进行拼接，得到长度为1280的token，将此token送入GPT进行生成训练。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/13-diffusion-models/6.jpg)

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/13-diffusion-models/7.jpg)

推理阶段，进行文生图任务时，只输入text，对text编码之后进入GPT解码出image tokens，之后将image tokens通过VQVAE的codebook得到Latent code，再送入VQVAE的decoder解码出原图片。可以生成很多张图片，根据CLIP分数进行筛选。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/13-diffusion-models/8.jpg)

## 概述

CLIP模型可以学到很稳健的图像特征，能同时捕获图像的语义和风格信息。DALL-E2又名为unCLIP，因为CLIP是给定文本和图像，生成文本和图像特征；DALL-E2为给定文本特征，得到图像特征，进一步得到图像的过程，类似于CLIP的逆过程。

DALL-E2是一个二阶段模型：prior阶段利用给定的文本描述生成CLIP图像嵌入特征，decoder阶段根据图像嵌入特征+文本特征生成真实图像。（文本->frozen的CLIP文本编码器生成文本特征->prior网络根据文本特征生成图像特征->decoder网络将图像特征解码成图像）。

其中，decoder网络采用扩散模型，prior网络实验了扩散模型和自回归模型（如DALLE），最终还是选择了扩散模型，因为扩散模型计算更高效，生成样本质量更高。

下图为DALL-E2流程。文本首先经过CLIP的text encoder得到text embedding，这一步骤无需训练，直接使用frozen的权重。之后text embedding经过prior网络得到image embedding，这一阶段使用CLIP模型中文本图像对的text embedding和img embedding进行监督学习。之后image embedding经过decoder得到图像。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/13-diffusion-models/9.jpg)

实验表明，DALL-E2生成的图像逼真多样（GAN生成的图像保真度好，但不多样）。

## 方法细节

### Prior

Prior也采用扩散模型。因为输入输出是embedding，所以不再使用U-Net预测，而是用decoder-only Transformer预测。扩散模型的输入：the encoded text, the CLIP text embedding, an embedding for the diffusion timestep, the noised CLIP image embedding, a final embedding whose output from the Transformer(类似CLS token）；输出：the unnoised CLIPimage embedding。

与一般工作使用噪声预测不同，DALL-E2训练模型来直接预测embedding $z_i$，使用平方误差损失：

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/13-diffusion-models/10.jpg)

### Decoder

- 基于GLIDE模型
- 根据image embedding和text embedding两个条件进行图像重建

>  图片出处：https://zhuanlan.zhihu.com/p/648389187

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/13-diffusion-models/11.jpg)

- 采用级联模型，训练2个diffusion upsampler models。第一个模型从64x64->256x256，第二个模型从256->1024。为了提升上采样的鲁棒性，在训练过程中轻度破坏图片，如使用gaussian blur等。
- 采用classifier-free guidance，训练过程随机以10%的概率将CLIP embeddings设置为0，随机以50%的概率丢弃text capltion。
