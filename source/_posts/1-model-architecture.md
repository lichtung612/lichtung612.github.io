---
title: 模型架构｜移动端基础模型架构设计
date: 2023-08-22
mathjax: true
cover: false
category:
 - model architecture
tag:
 - lightweight_CNN
---

## 移动端基础模型

### 目标

针对**资源有限**的移动设备，在控制**模型大小**和**模型推理时间**等前提下实现卓越的模型性能。

### 评价指标

- **Parameter**
  -  模型训练中需要训练的参数总数，用于衡量模型的大小（空间复杂度）。

  -  考虑到移动设备越来越便宜和更大的存储空间，参数量的限制已经大大放松。
  
- **FLOPs（floating-point operations)**
  -  浮点运算次数，乘法和加法的数量，可以理解为计算量，用来衡量模型的时间复杂度。
  
- **MACs（Multiply-Accumulate Operations)**

  乘加累积操作。1MACs包含一个乘法操作和一个加法操作，大约相当于2个FLOPs。MACs和MAdds同义。

  仅仅使用FLOPs来估测模型速度是**不准确**的，内存访问成本、模型并行度、平台架构等都会影响模型速度。

- **MAC(memory access cost)**
  -  内存访问成本。在强计算资源如GPUs下，算力不是问题，而内存访问成本可能成为模型瓶颈。
- **Inference Speed**

  单位Batches/sec，模型推理时每秒成功处理的批次数。

  下图展示GPU上，batchsize=8，相同FLOPs的模型具有不同的推理速度(ShuffleNet v2)：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/0.jpg" alt="img" style="zoom:70%;" />

- **Latency**
  -  指从输入数据传入模型到模型输出结果可用之间的时间间隔。它衡量了模型处理单个样本所需的时间。

  - MobileOne在iphone12的实验：

    > pytorch模型->onnx形式->用core ML工具转换成coreml packages->在iPhone12上测量latency

    （1）很多模型的参数量很大，但是latency很低
  
    （2）CNN模型相对transformer模型在相同FLOPs和参数量下具有更低的latency

| ![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/1.jpg) | ![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/2.jpg) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                       FLOPS vs Latency                       |                  Parameter Count vs Latency                  |

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/3.jpg" alt="img" style="zoom:67%;" />

## 经典模块

### Depthwise separable convolution

> MobileNet-google-17: https://arxiv.org/pdf/1704.04861.pdf

深度可分离卷积（depthwise separable convolution) = 深度卷积 (depthwise convolution)+ 逐点卷积(pointwise convolution)

- Depthwise convolution：卷积核拆分成单通道，不改变图像深度情况下，对每一通道进行卷积操作。节省参数量，关注长宽方向空间信息
- Pointwise convolution：将不同通道特征融合，关心跨通道信息；对特征图进行升维和降维

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/conv.png" alt="img" style="zoom: 50%;" />

### Inverted residual

> MobileNetV2-google-18:https://arxiv.org/pdf/1801.04381.pdf

相比MobileNet v1的depthwise separable convolution, MobileNet v2：

- 在高维空间做DW卷积。即先用PW升维，之后DW，之后PW降维。在高维空间可以保证信息丢失更少。
- 添加残差连接

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/6.jpg" alt="img" style="zoom:50%;" />

### Squeeze-and-excite

> Squeeze-and-Excitation Networks：https://arxiv.org/pdf/1709.01507.pdf

 $F_{tr}$:传统卷积结构

 $F_{sq}$：Squeeze过程，对输出每个通道对应特征图做一个全局平均池化操作

 $F_{ex}$: Excitation过程，全连接层

$F_{scale}$:用sigmoid将输出限制到[0，1]的范围，把这个值作为scale乘到U的C个通道上， 作为下一级的输入数据

SE层原理：通过控制scale的大小，把重要的通道特征增强，不重要的通道特征减弱，从而让提取的特征指向性更强

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/7.jpg)

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/8.jpg" alt="img" style="zoom:50%;" />

### Structural reparameterization

> RepVGG: https://arxiv.org/pdf/2101.03697.pdf

训练和推理的结构不同。训练时采用多分支结构，可以增加模型表征能力；推理时通过结构重参数化将模型转变为单分支结构，因为单路结构不需要保存其中间结果，会占有更少的内存，同时单路架构并行度高，速度更快。

RepVGG架构如下：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/9.jpg" alt="img" style="zoom:67%;" />

## Lightweight CNNs

### RepViT

> RepViT-2307: Revisiting Mobile CNN From ViT Perspective
>
> https://arxiv.org/pdf/2307.09283.pdf

#### 动机

最近，轻量级ViT相比较于轻量级CNN在资源受限型设备上证明了卓越的性能和低的延迟。这个提升经常被归功于多头自注意力机制，多头自注意力机制可以让模型学习全局特征表示。然而，**轻量级ViT和轻量级CNN在模型架构上的区别没有被充分审视**。

问题：**轻量级ViT的架构用到轻量级CNN上，能不能提升轻量级CNN的性能？**

本文：重新研究高效轻量级CNN和ViT，通过将轻量级ViT的架构融合进标准轻量级CNN架构MobileNet v3中，来增强轻量级CNN的移动设备友好性。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/10.jpg" alt="img" style="zoom:50%;" />

#### 背景

##### MobileNet v3

Mobilenet V3 block: Inverted residual + Squeeze-and-Excite

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/11.jpg" alt="img" style="zoom:67%;" />

##### MetaFormer

> MetaFormer: MetaFormer Is Actually What You Need for Vision
>
> https://arxiv.org/pdf/2111.11418.pdf

观点：ViTs的有效性主要来自它们通用的token mixer和channel mixer结构，即MetaFormer架构。其性能与特定的token mixer如attention mixer关系不大。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/12.jpg)

#### 架构

##### 概览

主要根据latency和top-1 accuracy来设计选择模型架构，逐步modernize MobileNetV3-L。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/13.jpg" alt="img" style="zoom:80%;" />

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/14.jpg)

##### 对齐训练策略

- MobileNetV3-L:采用CNN的训练策略（RMSPropOptimizer with 0.9 momentum for 600 epochs等设置）
- RepViT: 为了公平比较，采用ViT的训练策略（和DeiT类似，使用Adam optimizer,cosine annealing learning rate schedule for 300 epochs等设置）。

##### **Block design**

###### 分离token mixer和channel mixer

根据MetaFormer, ViT的有效性存在于它们的token mixer和channel mixer架构，而不是特定的token mixer。为此，作者**将MobileNetV3 block改成MetaFormer结构**。

- Depthwise convolution：空间信息的交互融合，相当于token mixer
- 1x1 pointwise convolution：通道之间特征交互，相当于channel mixer

在MobileNet V3 block中token和channel mixer混合在一起了，为此RepViT将其分离。此外，RepViT采用了结构重参数化技术，减小推理延迟。

下图：(a)MobileNetV3块（b)RepViT块训练结构（c)RepViT块推理结构：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/15.jpg" alt="img" style="zoom:67%;" />

###### 减小expansion ratio，增加width

- Expansion ratio: 1x1升维卷积将特征矩阵升维的倍数；在MobileNetV3-L中，不同stage升维倍数从2.3到6，**通道数量越大模块越冗余**。因此RepViT将expansion ratio减少为2。
- Network width：1x1降维卷积后每个block输出的维度。RepViT增加了每个block的输出维度。

##### **Macro design**

###### Early convolutions for stem

Early convolutions：在轻量级ViT中被广泛采用的stem，用堆叠的stride=2的3x3卷积处理。相比MobileNetV3-L中复杂的stem,更简洁，降低延迟并增加了准确率。

如图所示，(a)MobileNetV3-L的stem (b)RepViT的stem：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/2.jpg" alt="img" style="zoom:67%;" />

###### Deeper downsampling layers

研究表明**单独的下采样层有助于增加网络深度，并减轻分辨率降低带来的信息损失**，这通常被轻量级ViT采用。

MobileNetV3-L：在拟残差块中使用stride=2的depthwise卷积完成降采样。

RepViT：用stride=2的depthwise卷积完成降采样，之后2个1x1卷积增加通道数。

更进一步，RepViT首先在深度卷积后增加单独的1*1卷积变更通道数，以使降采样层与块结构解耦。在此基础上，降采样层由前面的RepViT块及后面的FFN进一步加深。

(a)MobileNetV3-L降采样层(b)采用RepViT块设计后的降采样层(c)RepViT最终降采样层结构

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/17.jpg" alt="img" style="zoom:67%;" />

###### Simple classifier

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/18.jpg" alt="img" style="zoom:67%;" />

###### Overall stage ratio

Stage ratio表示不同阶段中块数的比率，暗示各阶段的计算分布。

研究表明**在第三阶段使用更多的块可以在精度和速度之间取得良好的平衡**，这通常被轻量级ViT所采用。

MobileNetV3-L：1:2:5:2

RepViT：1:1:7:1（network depth: 2:2:14:2)

##### **Micro design**

###### Kernel size selection

- 大尺寸的卷积核的计算复杂度和MAC更高，对移动设备不友好；
- 与3x3卷积相比，编译器和计算库通常不会对较大的卷积核进行高度优化

RepViT将卷积核尺寸全部设定为3x3（MobileNetV3中有一部分卷积核是5x5）

###### Squeeze-and-excitation layer placement

与卷积相比，自注意模块的一个优点是能够根据输入调整权重，即具有数据驱动属性。作为一种通道式的attention模块，SE层可以在缺乏数据驱动属性的情况下补偿卷积的限制，带来更好的性能。

MobileNetV3-L在某些块中包含SE层，主要关注后两个阶段。然而研究表明**与具有更高分辨率的阶段相比，具有低分辨率的阶段从SE提供的全局平均池操作中获得的精度益处较小**。同时，随着性能的提高，SE层也引入了不可忽略的计算成本。因此，RepVit设计了一种策略，在所有阶段的cross-block中利用SE层，以最小的延迟增量最大限度地提高准确性。

#### 实验

**Image Classification**

- *Compared with widely used lightweight CNNs, RepViT generally achieves a better trade-off between accuracy and latency*
- *pure lightweight CNNs can outperform existing state-of-the-art lightweight ViTs on mobile devices by incorporating the efficient architectural designs.* 

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/19.jpg" alt="img" style="zoom:67%;" />

### MobileOne

> MobileOne-2206: An Improved One millisecond Mobile Backbone:https://arxiv.org/pdf/2206.04040.pdf

#### 动机

移动设备backbones通常针对FLOP或Parameter等指标进行优化。然而，当部署在移动设备上时，这些指标可能与网络的延迟不太相关。为此，本文通过在移动设备上部署模型进行不断调试分析，**针对iphone12架构**设计了一种高效的主干MobileOne，其变体在iPhone 12上的推理时间低于1毫秒，在ImageNet上的最高准确率为75.9%。

#### 分析

##### 激活函数

不同激活函数对延迟的影响很大，MobileOne选择ReLU激活函数。

比较在30层CNN中不同激活函数的latency：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/20.jpg" alt="img" style="zoom:67%;" />

##### Architectural Blocks

影响运行时性能的两个关键因素是**内存访问成本(MAC）**和**并行度**。

- MAC：在multi-branch architectures（如skip connection)中，因为必须存储每个分支的特征图来计算下一个特征张量，MAC成本显著增加。为此，MobileOne采用**推理时没有分支**的架构。

- Degree of parallelism：强制同步操作的架构块（如SE块中的global pooling操作）也会由于同步成本而影响整体运行时间。为此，MobileOne限制SE块的使用。

  比较在30层CNN中不同架构blocks：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/21.jpg" alt="img" style="zoom:80%;" />

#### 架构

##### MobileOne Block

和MobileNet-V1类似，由depthwise卷积和pointwise卷积构成，只不过增加了trivial over-parameterization branches。

如下图所示，MobileOne block在训练和推理时的架构不同：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/22.jpg" alt="img" style="zoom:67%;" />

**重参数化分支的影响**：

下表为在ImageNet上的top-1准确率：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/23.jpg" alt="img" style="zoom:80%;" />

**重参数化因子k**：超参数，范围从1到5。对于MobileOne的较大变体，k增加带来的增益开始减少。对于较小变体MobileOne-S0，通过使用多个参数化分支，性能提升0.5%

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/24.jpg" alt="img" style="zoom:67%;" />

##### Model Scaling

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/25.jpg)

MobileOne-S1比和它大3倍的RepVGG-B0性能更好：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/26.jpg" alt="img" style="zoom:67%;" />

##### Training

小模型需要较少的正则化来防止过拟合，在训练过程中逐渐减小权重衰减正则化更为有效。

使用余弦调度（cosine schedule)来进行学习率的调整。此外，我们也使用相同的调度方法来逐渐减小权重衰减系数。引入progressive learning。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/27.jpg" alt="img" style="zoom:80%;" />

#### 实验

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-model-architecture/28.jpg" alt="img" style="zoom:80%;" />
