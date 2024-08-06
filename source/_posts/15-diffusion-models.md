---
title: 扩散模型（十五）｜T2I Diffusion Models：Kolors
date: 2024-08-06
mathjax: true
cover: false
category:
 - Diffusion model
tag:
 - T2I Diffusion Models

---

> Kolors: Effective Training of Diffusion Model for Photorealistic Text-to-Image Synthesis
>
> https://github.com/Kwai-Kolors/Kolors/blob/master/imgs/Kolors_paper.pdf
>
> https://github.com/Kwai-Kolors/Kolors

近期，快手提出Kolors模型，具备强大的中英文理解能力，主打生成写实风格图片。Kolors在backbone架构方面和SDXL相同，直接继承SDXL的U-Net架构，没有对模型架构做改进（这也意味着基于sdxl进行设计的方法可以较容易迁移到Kolors上）。即使没有对U-Net进行改进，Kolors也表现出卓越的性能，超越了目前的开源方法，达到Midjourney-v6等级水平。相比SDXL和其它文生图模型，Kolors的改进：

1. 在增强文本语义理解方面：

   - 使用General Language Model(GLM)作为text encoder，而非CLIP或T5等。因为GLM具备强的中英文双语理解能力。

   - 使用多模态大语言模型来重新标注训练数据集，显著提升Kolors理解综合复杂语义信息的能力，特别是涉及多个实体的语义。

   - 通过在训练数据中添加关于中文文字的合成数据和真实数据，提升了模型的中文文本渲染的能力。

2. 在提升视觉效果方面：

   - 将Kolors的训练分为两个阶段：具有广泛知识的概念学习阶段+使用高美感数据的质量提升阶段。

   - 引入一个新的schedule策略，来优化高分辨率图像的生成。

## 方法

### **增强文本语义理解**

#### 大语言模型作为Text Encoder

不同文生图模型使用的text encoder总结如下表所示：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/15-diffusion-models/0.jpg" alt="img" style="zoom: 50%;" />

- CLIP：被经典方法如sd1.5和DALLE2使用。因为CLIP的训练目标是对齐文本和整个图像的特征，难以理解细粒度的图像描述，如包含多个实体、位置、颜色等描述。
- T5：被Imagen和PixArt-α等模型使用。T5可以捕获更多细粒度的局部信息。
- CLIP&&T5：被eDiff-I和SD3模型使用，可以同时捕获全局语义特征和局部细粒度语义特征表示。

上述文本编码器只能基于英文。

- HunyuanDiT：利用双语CLIP和多语言T5模型来完成中文文生图的生成。然而，在多语言T5数据集中中文训练预料少于2%，双语CLIP产生的文本编码对复杂prompt的理解能力不充分。

为了解决以上限制，Kolors选择General Language Model(GLM)作为text encoder，具体选择的是ChatGLM3-6B-Base版本。GLM是一个双语（中英）预训练大语言模型，具有**强的中文理解能力**。且相比于最多77个token的CLIP模型，ChatGLM3具备优越**长文本理解能力**，Kolors将ChatGLM3设置为可接受**256个token**。

Kolors with CLIP和Kolors with GLM生图对比如下：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/15-diffusion-models/1.jpg" alt="img" style="zoom: 50%;" />

#### 多模态大语言模型重新细粒度标注训练数据集

训练T2I使用的text-image对通常来自于网络，对应的标注充满噪声和不准确性。效仿DALL-E3，Kolors使用多模态大语言模型重新标注text-image pairs。

对不同多模态大语言模型进行评估，评估结果如下表所示：

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/15-diffusion-models/2.jpg)

其中，LLaVA1.5-13B、CogAgent-VQA、CogVLM-1.1-chat的结果是从英文翻译成中文得到的（因为这几个模型生成英文能力更强）。GPT4-V的结果最好，但是它是昂贵耗时的；为此Kolors选择CogVLM-1.1-chat来重新进行标注。

考虑到MLLM可能无法识别图像中不存在于其知识语料库中的特定概念，Kolors采用了一种使用50%原始描述与50%合成描述比率的策略进行训练（和sd3的训练配置一致）。

#### 增强中文文本渲染能力

在文生图中渲染文字一直是一个有挑战性的问题。stable diffusion3和dalle-3展现出很强的渲染英文文本能力。然而，渲染中文文本依然存在挑战，主要原因：

1. 与英语相比，大量的汉字和这些汉字复杂的纹理使中文文本的翻译更具挑战性。
2. 缺乏足够的训练数据，包括中文文本和相关图像，导致模型训练和拟合能力不足。

Kolors通过整合合成数据和真实数据，提升了模型对中文文本渲染的能力。

- synthetic data：选择了50000个常用词，使用合成的方法构建数千万图文对的训练数据集。这些数据在概念学习阶段被学习。
- real-world data：为了增强生成图像的真实感，利用OCR和多模态大语言模型来生成真实世界描述，如海报和场景文字，得到百万个样本。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/15-diffusion-models/3.jpg" alt="img" style="zoom: 50%;" />

### **提升视觉效果**

#### 高质量数据

Kolors的训练被分为两个阶段：概念学习阶段和质量提升阶段。

在概念学习阶段，模型从大规模数据集中学习综合知识和概念。该阶段数据来自public datasets(LAION,DataComp,JourneyDB)。

在质量提升阶段，模型关注提升图像细节和美感。为了获得高美感数据，首先对数据集应用了传统的过滤方法(如分辨率、OCR精度、人脸数量、清晰度和美学得分)，从而将数据集缩减到约数千万张图像。这些图像随后经过人工标注，标注被分为五个不同的级别。为了减轻主观偏差,每张图像都进行了三次标注，最终级别通过投票决定。不同级别图像的特征如下:

- 第1级:包含色情、暴力、血腥或恐怖内容的不安全图像。
- 第2级:显示人工合成痕迹的图像,如包含LOGO、水印、黑白边框、拼接等。
- 第3级:存在参数错误的图像,如模糊、过曝、欠曝或主题不清晰。
- 第4级:普通快照级别的平凡照片。
- 第5级:具有高美学价值的优质照片,不仅曝光、对比度、色调和饱和度适当,还能传达一定的叙事性。

这种方法最终产生了数百万张第5级的高美学图像,这些图像被用于质量提升阶段。

#### 针对高分辨率的训练

由于在正向扩散过程中对图像的破坏不足，扩散模型在高分辨率下的表现往往不佳。如下图所示，当按照SDXL中提供的scheduler添加噪声时，低分辨率图像会退化为近乎纯噪声，而高分辨率图像往往在最终阶段保留低频分量。由于模型在推理过程中必须从纯高斯噪声开始，这种差异可能会导致高分辨率下训练和推理之间的不一致（训练时T=1000不是纯噪声，而推理时T=1000是纯噪声）。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/15-diffusion-models/4.jpg" alt="img" style="zoom:67%;" />

Kolors选择基于DDPM的训练方法。在概念学习的低分辨率训练阶段，采用和SDXL相同的noise scheduler。在高分辨率训练阶段，引入一个新的schedule，它将步数从1000扩展到1100，使模型达到一个更低的最终信噪比。在此过程，调整 $\beta$来维持$\bar \alpha$曲线不变。如下图所示，Kolors的$\bar \alpha$曲线包含了sdxl的$\bar \alpha$曲线，而其它方法都存在偏差。这表明，从低分辨率的base schedule迁移到新的schedule时，与其它schedule相比，此schedule在适应和学习难度方面都有所降低。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/15-diffusion-models/5.jpg" alt="img" style="zoom:67%;" />

如下图所示，通过集成高质量训练数据和优化的高分辨率训练技巧，生成图片的质量大幅提升。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/15-diffusion-models/6.jpg" alt="img" style="zoom: 50%;" />

此外，Kolors在高分辨率训练阶段应用NovelAI的bucketed采样方法，训练生成不同分辨率大小的图像，如下图所示：

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/15-diffusion-models/7.jpg)

## 评估

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/15-diffusion-models/8.jpg)

## 展望

1. 快手将发布基于Kolors的多个应用，如ControlNet、IP-Adapter等。
2. 计划发布基于Transformer架构的扩散模型。
