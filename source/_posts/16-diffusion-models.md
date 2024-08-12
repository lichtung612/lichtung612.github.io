---
title: 扩散模型（十五）| 主题生成：Textual Inversion
date: 2024-03-12
mathjax: true
cover: false
category:
 - Diffusion model
tag:
 - 主题生成
---
> An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion
>
> 论文：NVIDIA-https://arxiv.org/pdf/2208.01618
>
> 代码：https://github.com/rinongal/textual_inversion

## 概览

***任务：***Textual Inversion属于主题生成任务的一种方法。用户给定3-5张指定概念的参考图片，如一个主题或者某种风格，训练模型，使模型可以个性化生成和此概念有关的图像。

***以往的方法：***冻结住模型，训练transformation模块以在面对新概念时适应其输出。这些方法往往会导致对先验知识的遗忘，即有可能学会生成这个概念，但是忘记了其它概念如何生成；并且这些方法难以持续学习新的概念（学习概念A后，无法在学会概念A的情况下再学会概念B）。

***Textual Inversion:*** "An Image is Worth One Word"，将参考图片看作一个伪单词，学习其在文本嵌入空间中的text embedding表示。后续文生图过程中，伪单词和其它单词被一样对待，它可以和其它单词组成各种prompt，生成各种图像。这个过程没有对生成模型进行任何改动，因此保留了生成模型理解其它概念的能力，避免了微调模型导致的模型理解能力和泛化能力下降问题。

***Textual Inversion vs Dreambooth:*** Textual Inversion训练更加简单，只需要学习图像对应的一个word embedding，参数量非常之小。相比下来，dreambooth需要微调u-net模型，参数量显然更大。也因此，Textual Inversioin效果不如Dreambooth更加精细，主题生成一致性能力不如Dreambooth。从论文中也可以看出，Textual Inversion的实验很多是图像风格的这种比较粗糙的转换。有研究者尝试将Textual Inversion和Dreambooth两种方法结合起来训练，先训练Textual Inversion2000步，之后训练Dreambooth500步，效果比单纯训练Dreambooth效果更好（https://huggingface.co/blog/zh/dreambooth）。

## 方法

文生图过程中，首先输入文本被Tokenizer转换成一系列tokens，每一个token被编码为对应的嵌入向量，这些文本特征嵌入向量被当作text condition送入下游模型进行生图。

Textual Inversion使用一个伪单词$S\*$来表示概念图像，目的是学习到伪单词$S\*$ 的word embedding。

为了学习到伪单词S* 的text embedding，使用3-5张概念图像对此word embedding进行训练。训练目标即扩散模型的MSE损失。训练prompt为随机从CLIP ImageNet模版中采样的中性上下文文本，如“A photo of $S\*$”或者“A rendition of $S\*$”。word embedding被初始化为概念图像的单个粗略描述词的嵌入（如当想要生成特定的猫时，可以用“cat”的word embedding来初始化）

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/16-diffusion-models/0.jpg" alt="img" style="zoom:67%;" />

## 实验

### 主题生成

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/16-diffusion-models/1.jpg" alt="img" style="zoom:80%;" />

### 风格迁移

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/16-diffusion-models/2.jpg" alt="img" style="zoom:80%;" />

### 多概念生成

模型可以同时对多个新的伪词进行推理。然而，它在处理这些概念之间的关系方面存在困难（它无法将两个概念并排放置，生成的都是以风格A生成概念B，不能生成概念A和概念B）。作者认为这是因为他们的训练仅考虑单个概念场景，在多概念场景上训练可能会缓解此缺陷。（感觉多主题生成本身也是文生图模型不擅长的地方。不考虑特定概念，baseline模型本身可能效果就比较差）

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/16-diffusion-models/3.jpg" alt="img" style="zoom:80%;" />