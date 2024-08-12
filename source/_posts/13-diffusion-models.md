---
title: 扩散模型（十三）| 主题生成：DreamBooth
date: 2024-03-04
mathjax: true
cover: false
category:
 - Diffusion model
tag:
 - 主题生成
---

> DreamBooth：Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation
>
> 论文：https://arxiv.org/pdf/2208.12242.pdf
>
> 代码：https://dreambooth.github.io/
>
> 参考：https://www.bilibili.com/video/BV18m4y1H7fs/?spm_id_from=333.788&vd_source=f16d47823aea3a6e77eec228119ddc27

## 任务

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/12-diffusion-models/0.jpg)

给定针对特定主题的输入图像（通常3-5张）和文本提示，通过diffusion model可以生成该主题在其他语义下的图像。生成结果可以自然和环境交互，具有多样性和高保真度。

## 方法

### Text-to-Image Diffusion Models

预训练的text-to-image diffusion model：$\hat x_\theta$

初始噪声图: $\epsilon \sim N(0,I)$

条件向量 $c=T(P)$，T是text encoder，P是text prompt

生成图像 $x_{gen} = \hat x_\theta(\epsilon,c)$

训练损失函数：

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/12-diffusion-models/1.jpg)

### Personalization of Text-to-Image Models

#### **Prompt设计**

训练prompt设计为："a [identifier] [class noun]"

其中，[identifier]是一个独一无二的和要生成主题相关联的标识符，[class noun]是一个粗粒度的主题类别描述符（如cat,dog,watch,etc）。

对于identifier，其需要在语言模型和diffusion模型中有一个弱的先验。一个不好的方式是在英语字母中随机选择字母，之后拼接起来，产生一个稀有的identifier，如"xxy5syt00"。因为在实现中，tokenizer可能逐个字母进行tokenizer，diffusion模型对这些字母有强的先验知识，使用这些词和使用普通的英语单词有相同的弱点。

DreamBooth的方法是去找词汇表中出现的稀有tokens，之后将这些tokens转换到text空间，得到对应的单词。

对于Imagen，均匀随机采样符合解码后长度小于等于3的单词的tokens；对于T5-XXL：选择{5000-10000}之间的tokens。

#### **Class-specific Prior Preservation Loss**

DreamBooth微调模型的所有层，包括text embeddings。这会导致两个问题：

- Language drift：在大规模文本语料中预训练的模型，之后在很小的数据中微调，会导致模型丢掉一些语义理解，只会生成该主题图片，忘记如何生成同个类别的其他主题图片。（只会生成某只狗，不会生成其他狗）

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/12-diffusion-models/2.jpg)

- Reduced output diversity：模型因为反复使用几张图像微调，可能会过拟合，导致输出多样性降低。如下图第二行所示，输出的狗全部都是和输入一样的趴着的姿态。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/12-diffusion-models/3.jpg)

提出**class-specific prior preservation loss**来解决上述问题。

为了使DreamBooth在微调过程中仍然保持先验，其**使用模型在微调前自己生成的样本来监督微调过程**。

使用prompt:"a [class noun]"通过预训练好的diffusion model生成数据集 $x_{pr}$。损失表示为：

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/12-diffusion-models/4.jpg)

其中第一项为模型原本的损失，添加第二项prior-preservation term来使用模型自身生成的图像监督模型训练，其中 $\lambda$控制第二项的权重。如下图所示表示模型训练过程：

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/12-diffusion-models/5.jpg)
