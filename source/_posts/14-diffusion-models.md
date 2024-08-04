---
title: 扩散模型（十四）｜Image-Conditioned T2I：IP-Adapter
date: 2024-03-07
mathjax: true
cover: false
category:
 - Diffusion model
tag:
 - Image-Conditioned T2I

---

> IP-Adapter: https://arxiv.org/pdf/2308.06721

“an image is worth a thousand words”，图像中蕴含丰富的信息，image prompt比text prompt更直观易理解。本文介绍一个将图像条件注入扩散模型的方法：IP-Adapter。除了IP-Adapter外，ControlNet也是经典的将图像条件注入模型的方法，其通过在原始sd模型中添加额外的网络（并行的unet的encoder层）进行训练来完成空间条件注入，在此前的博客https://lichtung612.github.io/posts/8-diffusion-models/ 中有专门介绍。IP-Adapter、ControlNet都在未改变原扩散模型权重参数的情况下，完成了空间条件信息的注入；IP-Adapter和ControlNet可以结合起来，完成对参考图的结构控制生成（见本文应用部分）。

## 动机

以往的基于adapter的图像条件注入方法，如T2I-adapter、Uni-ControlNet，它们的做法通常如下：

1. 使用CLIP image encoder提取参考图像特征
2. 将提取到的图像特征送入可训练的image adapter网络中，进一步将CLIP提取到的image embedding和扩散模型内部特征对齐。
3. 将对齐后的image embedding和text embedding进行concat，得到图文融合特征
4. 图文融合特征送入U-Net中，通过cross attention来完成图文特征注入

这种做法生成的图像仅仅部分和参考图一致，问题出在cross attention机制上。concat起来的图像和文本特征被一视同仁地送入相同的cross attention模块上进行生成，而这个**cross attention模块最初被训练时仅仅是用来注入文本信息**，这很容易导致细粒度的图像信息不能有效地被融合进unet模型中，

为此，IP-Adapter设计了针对文本和图像解耦的cross-attention机制，添加可训练的图像条件注入cross-attention模块，在sd1.5上仅仅添加22M参数便可达到和全量微调image prompt模型相同的效果。

## 方法

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/14-diffusion-models/0.jpg" alt="img" style="zoom: 67%;" />

### Image Encoder

1. 使用预训练好的CLIP image encoder来从参考图中提取图像特征。
2. 使用一个小的可训练的投影网络将提取到的图像特征投影成一序列长度为N的特征（N=4），image features和text features的特征维度相同。投影网络由一个线性层和一个层归一化组成。

### Decoupled Cross-Attention

在unet每一个cross attention block位置添加一个针对图像特征的cross attention block。其 $W_q $矩阵用来投影生成图像的特征，采用和文本cross attention block同样的$W_q $投影矩阵，不需要训练； $W_k$和 $W_v$矩阵用来投影参考图像特征，需要训练，初始化方式为和文本cross attention同样的$W_k$和 $W_v$，这样初始化可以更快收敛。简单地将image cross-attention和text cross-attention输出相加，得到最终输出。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/14-diffusion-models/1.jpg)

### Training and Inference

**训练阶段：**

训练过程，仅仅训练IP-Adapter中添加的image cross-attention中的KV投影矩阵以及图像编码模块的linear和LN层，原始扩散模型保持frozen。训练使用和SD模型相同的训练目标，即预测噪声的MSE损失。

训练过程随机将image条件和prompt条件dropout(image和text同时dropout，概率为0.05)，来使模型同时学习到有条件生成和无条件生成能力，以便进行classifier-free guidance：

$$\hat \epsilon_\theta(x_t,c_t,c_i,t)=w\epsilon_\theta(x_t,c_t,c_i,t)+(1-w)\epsilon_\theta(x_t,t)$$

训练数据集为从LAION-2B和COYO-700M数据集中收集的1000万张text-image数据对，使用8V100训练。

**推理阶段：**

推理阶段可以通过权重因子 $\lambda$控制图像条件的权重：

$$Z^{new}=Attention(Q,K,V)+\lambda \cdot Attention(Q,K',V')$$

## 应用

### IP-Adapter+自定义模型/其它adapter

IP-Adapter可以直接使用到其它基于相同基线模型的微调模型和adapter上。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/14-diffusion-models/2.jpg" alt="img" style="zoom: 67%;" />

### IP-Adapter+结构控制

IP-Adapter可以和其它基于结构控制的方法兼容，如ControlNet。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/14-diffusion-models/3.jpg" alt="img" style="zoom:80%;" />

### Image-to-Image/Inpainting

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/14-diffusion-models/4.jpg" alt="img" style="zoom:80%;" />

### 多模态prompts

IP-Adapter可以较好地完成image prompt和text prompt一起进行图像生成引导。可以通过调整$\lambda$来在image prompt和text prompt之间取得平衡。如下图所示，可以使用text prompt来对图像进行编辑、改变主题的场景等。

但是，本文也强调，尽管IP-Adapter可以集成参考图的内容和风格来生图，它**在维持主题一致性方面表现比较差**，不如Textual Inversion和DreamBooth效果更好。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/14-diffusion-models/5.jpg)