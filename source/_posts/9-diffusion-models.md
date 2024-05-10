---
title: 扩散模型（九）| Diffusion for Point Cloud
date: 2024-01-24
mathjax: true
cover: false
category:
 - Diffusion model
---

## PointE

> PointE: A system for Generating 3D Point Clouds from Complex Prompts
>
> https://arxiv.org/pdf/2212.08751.pdf
>
> https://github.com/openai/point-e

### Abstract

最近的text-conditional 3D生成模型经常需要多个GPU-hours来生成一个单一样本，这和图像生成模型（几秒或几分钟就可以生成样本）相比有巨大的差距。本文提出一个3D生成模型PointE（generate Point clouds Efficiently)，可以用单个GPU在1-2分钟内生成3D模型。

该方法首先用一个text-to-image diffusion模型生成一个单一的合成视角的图像，之后用另一个输入条件为图像的diffusion模型生成3D点云。

该方法虽然质量上和sota模型有差距，但是速度比其他的快1-2个数量级。

### Introduction

最近的text-to-3D方法大致分成2类：

- 使用成对的数据（text，3D）或者没有标签的3D数据训练生成模型。这些方法利用现有的生成模型方法来高效生成样本，但是它们面对生成多样性和复杂的文本Prompt是困难的，因为目前缺乏大规模3D数据集。
- 利用预训练的text-image模型来优化3D表示。这些方法可以处理复杂多样的text prompts，但是需要昂贵的优化过程来生成样本。此外，由于缺乏3D先验知识，这些方法容易陷入局部最优，不能生成有意义或连贯的三维对象。

本文试图结合两种方法的优势，通过将text-to-image模型和image-to-3D模型结合起来。其中，text-to-image模型利用大量的(text,image)语料对，允许模型根据多样复杂的prompt进行生成；image-to-3D模型在一个更小的(image,3D)数据集上训练。为了根据text prompt生成一个3D物体，首先使用text-to-image模型采样一个图像，之后根据这个采样的图像使用image-to-3D模型采样一个3D object。

### Method

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/0.jpg)

1. 通过一个text caption生成一个合成视角的图像 <- 3-billion参数的在渲染的3D模型数据集上微调的GLIDE模型
2. 通过合成视角的图像生成一个粗粒度的点云（1024points) <-具有置换不变性的diffusion model
3. 在低分辨率点云和合成视角图像条件下进行进一步生成，生成一个细粒度的点云（4096points）<-和第二个diffusion模型相似但是更小的扩散模型

#### Synthesis GLIDE Model

为了确保text-to-image模型可以生成正确的合成视角，训练模型让模型可以生成满足点云训练数据集分布的3D渲染。

为此，使用GLIDE初始训练数据集和3D渲染数据集微调GLIDE模型。因为3D渲染数据集相比于GLIDE训练数据集小很多，仅仅只在5%的时间内对3D数据集的图像进行采样，其余95%使用原始数据集。微调100K个迭代，意味着模型已经在整个3D数据集上训练了几个epochs。

在测试时，为了确保我们总是在3D分布渲染中采样，我们在每个3D渲染的文本提示中添加一个特殊的标记，指示它是3D渲染；然后在测试时使用此token进行采样。

#### Point Cloud Diffusion

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/1.jpg" alt="img" style="zoom:50%;" />

将点云表示成一个tensor，shape为K x 6，K是点云中点的数量，特征维度为(x,y,z,R,G,B)。全部坐标和颜色被标准化为[-1,1]。输入随机噪声K x 6，通过diffusion模型逐步denoising，生成点云。

利用简单的Transformer-based模型基于输入图像、时间步t、噪声点云 $x_t$来预测噪声 $\epsilon$和方差 Σ。 

- 条件编码
  - t：通过一个小的MLP，获得一个D维向量
  - 噪声点云：每个点通过一个linear层，得到K x D
  - 图像：使用一个预训练的ViT-L/14 CLIP模型提取特征，选择最后一层的特征嵌入（256 x $D'$），线性投影成256 x D

所有条件编码在batch维度concat，构成transformer的输入：(K+257)xD。为了获得长度为K的最终输出序列，我们取输出的最后K个标记并将其投影，以获得K个输入点的ε和∑预测。

注意，因为在整个过程中都没有使用位置编码，所以模型满足置换不变性。

#### Point Cloud Upsampler

upsampler模型采用和base模型相同的架构。对于低分辨率的point cloud采用另外的condition tokens，将条件点云通过一个分离的线性嵌入层，使模型可以区分高分辨率点云和低分辨率点云的信息。

### Results

#### Qualitative Results

PointE可以基于复杂的Prompts生成高质量的3D shapes。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/2.jpg" alt="img" style="zoom:50%;" />

失败的例子：

1. 错误地理解不同部分的相对比例，导致生成了一个高的狗而非短的长的狗。
2. 不能推理出被遮挡的部分

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/3.jpg" alt="img" style="zoom:50%;" />

#### Model Scaling and Ablations

1. 仅仅使用text conditioning，而没有text-to-image步骤导致模型产生更差的结果
2. 使用一个单一的token编码图像CLIP embedding比使用多个token编码CLIP embedding产生更差的结果
3. 扩大模型可以加速收敛，增大CLIP R-Presicion结果。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/4.jpg" alt="img" style="zoom:67%;" />

## PC2

> $PC^2$：Projection-Conditioned Point Cloud Diffusion for Single-Image 3D Reconstruction（牛津-CVPR23）
>
> 项目主页：https://lukemelas.github.io/projection-conditioned-point-cloud-diffusion/
>
> 补充材料：https://openaccess.thecvf.com/content/CVPR2023/supplemental/Melas-Kyriazi_PC2_Projection-Conditioned_Point_CVPR_2023_supplemental.pdf
>
> 代码：https://github.com/lukemelas/projection-conditioned-point-cloud-diffusion

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/5.jpg)

### 概览

***任务：***Single-View 3D reconstruction，给定一张RGB图片和对应视角，输出该图片中物体的3D点云。

***创新：***（1）以往的single-view 3D reconstruction生成的可能是体素或者NeRF表示，本文是第一个生成点云形式表示的工作。（2）首次将扩散模型应用到该任务中（3）条件注入方式采用投影注入(projection-conditioned)，在预测点的位置信息之后，再次使用投影条件注入方式来预测点的颜色（4）引入过滤步骤，利用diffusion生成过程的概率性，解决单视角3D重构问题中的不适应性(ill-posed nature）

***实验：***（1）对比之前的工作仅仅在合成数据集setting上进行实验，该方法不仅在合成数据集ShapeNet上达到SOTA；在复杂的真实世界数据集Co3D上也进行实验，性能达到SOTA。（2）可视化结果来看，该方法从任何视角都能产生真实的物体形态，可以生成具有更精细细节级别的形状。（3）消融实验证明了投影注入和过滤步骤的有效性

### 相关工作

#### Single-View 3D reconstruction

- 3D-R2N2：标准2D卷积网络编码图像，3D-LSTM处理图像特征，3D卷积网络解码成一个**体素形式**3D物体。
- 另一种范式是使用NeRF来学习隐式表示，如Nerf-WCE和PixelNeRF。这两种方法在few-view setting中表现不错，但是在single-view setting中表现欠佳。

本论文使用了一个完全不同的方法，利用扩散模型来进行3D重构。由于扩散模型的概率特性（probabilistic nature），使我们捕捉到不可见区域的模糊性，生成高分辨率点云形状。

#### Diffusion Models for Point Clouds

在过去一年中，扩散模型应用到**无条件****点云****生成任务**上的方法被提出。

diffusion-point-cloud和 Point-Voxel Diffusion (PVD)提出了相似的生成范式，不同的是它们的backbone一个是PointNet，一个是Point-Voxel-CNN。Point Diffusion-Refinement (PDR) 使用扩散模型来解决点云补全任务。

然而，这三个工作仅仅解决无条件的形状生成问题或者补全问题，没有解决从2D图片中进行3D重构的问题。此外，它们仅仅在合成数据集上训练，本文还展示了在真实世界数据上的结果。

### 方法

如下图所示，输入一张RGB图片和它的camera pose，从一个三维高斯分布随机采样一组点，这组点根据RGB图片和camera pose条件在diffusion过程中逐渐降噪，最终得到图片中物体的3D点云。具体来说，在扩散过程的每一步，图片特征和camera pose被投影到点云中，使生成的点云能够很好地和输入图像对齐，具备几何一致性。随后，使用相同的投影方法预测点的颜色。

因为diffusion的概率特性，可以针对一张图片生成多个可信的点云，引入一个过滤步骤来帮助解决单视角3D重构中的不适应性。具体来说，根据给定的图像生成多个点云，通过它们和输入mask的匹配程度来过滤掉某些点云，使生成样本的质量更高。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/6.jpg)

#### Point Cloud Diffusion Models

把具有N个点的3D点云视为一个3N维度的物体，扩散模型 $s_\theta:R^{3N}\rightarrow R^{3N}$，这个网络将一组从一个高斯分布中采样的点逐渐降噪成一个可识别的物体。在每一步，预测当前点的位置和它的ground truth位置的offset，迭代这个过程直到它和目标分布 $q(X_0)$类似。网络 $s_\theta$使用Point-Voxel CNN(PVCNN)。

具体来说，网络被训练来预测噪声 $\epsilon \in R^{3N}$，使用L2 loss来监督：

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/7.jpg)

在推理阶段，从一个三维高斯分布中随机采样点云（`x_t = torch.randn(B,N,D）`），进行逆向扩散过程生成样本 $X_0$。

#### 条件注入

目标分布是一个条件分布 $q(X_0|I,V)$,I是输入图像，V是对应的相机视角。

常见的输入图像的条件注入方式是将图像编码成一个global feature，然而这种方法在输入图像和重构shape上呈现弱的几何一致性，实验展示它经常生成可信的shapes，然而这些shapes和输入图像并不匹配。

本文创造PC2（projection-conditional diffusion models）。下图展示global condition和pc2的结果对比，可以看出PC2比global condition在F-scores指标上提高不少。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/8.jpg)

PC2首先使用一个标准2D图像模型（CNN或者ViT，本文用的是MAE-ViT）提取图像特征，2D图像经过图像模型提取后的特征为 $I \in R^{H\times W \times C}$。

之后，在每一个diffusion步骤之前，将这些特征投影到点云上，投影特征 $X_t^{proj}=P_{V_I}(I,X_t)$，其中 $P_{V_I}$是从相机视角 $V_I$的投影函数，I是输入图像，$X_t$是部分降噪的点云。

经过条件增强的点云为 $X_t^+=[X_t,X_t^{proj}]$，其特征维度为 $R^{(3+C)N}$。点云中每个点得到的投影特征各不相同。

$s_\theta$是预测噪声函数 $\epsilon$: $R^{(3+C)N}\rightarrow R^{3N}$。

**投影函数**:

对于投影函数$P_{V_I}(I,X_t)$的设计，一个简单的方式是直接把3D点投影到图片上，拿对应的图片特征。然而，这忽视了点云的立体属性，点云中真面和背面的点得到的特征相同，不符合常识。选择一个**适当考虑自遮挡**的投影函数是较好的。为此作者通过**光栅化**来实现这一过程，由于Pytorch3d中有高度优化的光栅化函数（use the PyTorch3D library for rasterization during the projection conditioning phase），这一过程可以非常高效实现，它只占训练时间和计算量的一小部分。

#### Coloring model

作者发现可以使用相同的projection-based条件注入方式来重建物体颜色。特别地，本方法学习一个**分离的**coloring模型 $c_\theta:R^{(3+C)N}\rightarrow R^{CN}$，输入经过第一阶段生成的点云，输出每个点的颜色。

PC2的color生成过程利用single-step模型，因为作者发现颜色模型扩散过程中**只需要一步**就可以生成不错的结果，这降低了计算复杂度。

#### 过滤

由于单视角3D重构问题的ill-posed性质，每张图片有很多可能的ground truth。不同于先前的工作，PC2有概率性的特性，因此可以从一个单一输入图像中采样出多个不同的重构。

过滤过程对某张图片生成多个样本，之后根据某种标准过滤，选择最可信的输出作为最终结果。

提出了2种标准，它们都包含物体剪影（提取3D点云的2D剪影）。一个使用额外的图像监督，一个没有。

- With mask supervision(PC2-FM)：比较每个点云剪影 $\hat M$和输入图像的主体mask$M$(通过Mask-RCNN提取)。计算两个剪影的交并比，选择具有最高交并比的生成样本。
- Without mask supervision(PC2-FA)：根据预测样本之间的相互一致性进行筛选。计算每个预测和其他预测的IoU，选择具有最高平均IoU的预测样本。这种方式可以在不添加任何其他额外监督的情况下选择出较高质量的样本。

### 实验

#### 评价指标

F-score来评估。对于两个点云 $X$和 $\hat X$，和一个固定的阈值距离d，公式为：

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/9.jpg)

其中 $P_d$和 $R_d$表示precision和recall。 $d$是一个固定的阈值距离，本文follow之前的工作，取d=0.01。precision和recall计算公式如下（P(d)：生成点云中的所有点逐个来看，如果真实点云中点和此点的最小距离小于d，将其求和取平均；R(d):真实点云中的所有点逐个来看，如果生成点云中点和此点的最小距离小于d，将其求和取平均）：

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/10.jpg)

#### Quantitative Results

ShapeNet-R2N2数据集上，没有过滤的情况下，PC2和之前工作差不多性能。通过检查不同类别的性能，我们可以看到PC2在具有更精细的细节相关的对象的类别上表现更好，如“步枪”和“飞机”。在过滤的情况下，PC2-FM达到SOTA结果。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/11.jpg)

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/12.jpg)

#### Qualitative Results

图3和图4展示了PC2在真实世界数据集Co3D上的定性结果。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/13.jpg)

因为NeRF-WCE的生成是固定的,它很难对高程度不确定的区域进行建模，为远离参考视图的新视图生成模糊的图像。与之对比，PC2从任何视角都能产生真实的物体形态：

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/14.jpg)

#### Diversity of Generations

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/15.jpg)

#### Filtering Analysis

Oracle根据F-score选择最佳样本，提供了性能上界。PC2-FM相比PC2-FA提升更多。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/16.jpg)

当过滤不同数量的图片时，性能如下：仅仅使用2张图片进行过滤就可以极大地提升结果，添加额外的图像得到更高的性能提升。增大过滤可选择样本数可以进一步提高性能，但回报递减。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/7-diffusion-models/17.jpg)