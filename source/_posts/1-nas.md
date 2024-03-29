---
title: NAS｜AutoFormer & AutoFormer++
date: 2023-10-18
mathjax: true
cover: false
category:
 - NAS
tag:
 - NAS
---

## AutoFormer

> AutoFormer: Searching Transformers for Visual Recognition（21 ICCV-microsoft）
>
> https://github.com/microsoft/Cream/tree/main/AutoFormer

### 动机

设计Transformer是有挑战性的。depth、embedding dimension、number of heads能够影响视觉transformer的性能。先前的工作这些特征依赖人工设定。然而，如下图所示，简单地增加depth(d), head number(h)和MLP ratio(r)在初始可以帮助模型达到更高的性能，然而在达到峰值后模型变得过拟合。增加embedding dimension(e)可以提高模型的性能，但对于较大的模型，精度会停滞不前。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/0.jpg)

本工作提出一个**one-shot**架构搜索框架AutoFormer，专用于发现好的视觉Transformer模型。该方法主要解决Transformer搜索中的2项挑战：

- 如何较好地结合transformer模型中的关键因素，如网络深度、嵌入维度、头数？
- 如何高效搜索满足资源限制的不同transformer模型？

### 创新点

受到BigNAS启发，提出一个超网训练策略：**weight entanglement**。

**核心思想**：使不同的transformer块共享公共部分的权重。更新一个block中的权重将影响其他的块，因此在训练中这些块的权重被纠缠复用。

- 经典的one-shot NAS网络中的权重共享策略
  
  在同一层中不同的block是分离的，即for any two blocks $b_j^{i}$ and $b_k^{(i)}$, we have $w_j^{(i)} \cap w_k^{(i)}=\emptyset$
- Weight Entanglement

​		同一层中不同block存在包含关系，即for any two blocks $b_j^{i}$ and  $b_k^{(i)}$ in the same layer, we have $w_j^{(i)} \subseteq w_k^{(i)}$or  $w_k^{(i)} \subseteq w_j^{(i)}$

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/1.jpg" alt="img" style="zoom:80%;" />

值得注意的是，weight entanglement策略专用于同构的blocks，因为同构的blocks是结构兼容的，因此权重可以彼此共享。在实施过程中，对于每一层我们需要仅仅存储这n个同构的候选者中最大的block对应的权重。其余的较小的block可以直接地从最大的blocks中提取到对应的权重。

**两种方式对比**

和经典的权重共享策略相比，weight entanglement有如下优势：

- 更快的收敛。相比独立的更新每个block的权重，weight entanglement允许每个块的权重更新很多次。
- 较低的内存消耗。仅仅需要存储每一层中最大的building blocks的权重，而不需要存储每一层中每一个block的权重。
- 更好的子网性能。通过weight entanglement训练的子网能够达到和train from scratch不相上下的性能。这意味着该方法可以直接获得上千个满足资源限制的权重，同时维持和train from scratch同等的准确率。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/2.jpg" alt="img" style="zoom:67%;" />

### 整体架构设计

设计一个**基于模型大小约束**的演化搜索算法。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/3.jpg)

#### 搜索空间

设计一个大的transformer搜索空间，包含5个因素考虑：embedding dimension, Q-K-V dimension, number of heads, MLP ratio, network depth。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/4.jpg" alt="img" style="zoom:67%;" />

将搜索空间编码成一个超网，在该搜索空间中的每一个模型是超网的一个子集，全部子集共享公共部分的权重。在训练期间，全部可能的子网被均匀采样，对应的权重被更新。整体来说，超网包含超过 $1.7 \times 10^{16}$个候选网络。

#### 搜索流程

> One-shot NAS
>
> 一个两阶段的优化问题。
>
> - 第一阶段：训练超网，优化超网权重W。其中 $\mathcal{A}$代表搜索空间。
>
> ![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/5.jpg)
>
> - 第二阶段：通过基于学习到的权重排序这些子网的性能，来搜索架构。
>
> ![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/6.jpg)

- 第一阶段（Supernet Training with Weight Entanglement)：

​		在每个训练迭代中， 从搜索空间中均匀采样一个子网，更新相应权重，冻结住其余权重。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/7.jpg" alt="img" style="zoom:67%;" />

- 第二阶段（Evolution Search under Resource Constraints)

​		在训练完超网后，我们对此超网进行演化搜索算法，来获得最优子网。目标是最小化模型大小的情况下最大化分类准确率。

​		初始，随机选择N个架构作为种子；其中的top k架构被选择作为父母，来通过交叉和变异产生下一代。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/8.jpg" alt="img" style="zoom:67%;" />

### 实验结果

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/9.jpg" alt="img" style="zoom:50%;" />

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/10.jpg)

#### Ablation Study and Analysis

- The Efficacy of Weight Entanglement

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/11.jpg)

- Subnet Performance without Retraining

如果我们进一步在ImageNet上对搜索到的子网络进行微调或重新训练，性能提升非常小，甚至可以忽略不计。这种现象说明weight entanglement使得子集在超网络中得到了很好的训练，从而导致搜索到的Transformer不需要任何重新训练或微调，而超网络本身就是子网络排名的良好指标。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/12.jpg" alt="img" style="zoom:67%;" />

## AutoFormer v2(S3)

> Searching the Search Space of Vision Transformer(21NIPS-microsoft)

### 动机

对于NAS模型来说，搜索空间是至关重要的，因为它决定了搜索出来的架构的性能界限。

本文提出Search the Search Space(S3)，回答2个关键问题：

- 如何有效且高效地度量特定的搜索空间的好坏
- 怎样在没有人类先验知识干预下将一个有缺陷的搜索空间转变成一个好的搜索空间

### 创新点

1. 对于如何度量搜索空间质量的问题，提出一个新的度量指标：E-T Error。
2. 对于如何将有缺陷的搜索空间转变成一个好的搜索空间的问题，如下图所示，通过多个维度分解搜索空间，包括depth,embedding dimension,MLP ratio,window size,number of heads和Q-K-V dimension，逐步演化每个维度值，来组成一个更好的空间。特别地，使用线性参数来建模不同维度的趋势，来指导模型维度演化。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/13.jpg)

### 方法

#### 问题分析

大部分存在的NAS方法被定义为一个约束优化问题：

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/14.jpg)

理想中， $\mathcal{A}$是一个无限的空间 $\Omega$,其包含所有可能的架构；在实践过程中，由于计算资源限制， $\mathcal{A}$仅仅是$\Omega$的一个小的子集。在本文中，我们打破在固定空间中搜索的传统，除了搜索架构，还搜索搜索空间。具体来说，将此优化问题分解为以下三步：

（1）在特定约束下，首先**搜索一个最优的搜索空间**

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/15.jpg" alt="img" style="zoom:67%;" />

其中， $\mathcal{Q}$是搜索空间度量函数。M代表最大搜索空间尺寸。

（2）编码搜索空间成一个超网，优化超网权重

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/16.jpg)

（3）在获得训练好的超网后，通过使用该权重排序子网的性能，演化搜索最优的子网结构

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/17.jpg)

#### 搜索空间定义

整体视觉Transformer架构设计如下。给定一张输入图片，首先均匀地将其分解成不重叠的patches。这些patches被线性投影成向量，这个过程为patch embedding。这些embedding被送入transformer encoder中。最后，一个全连接层被采用作为分类任务头。

Transformer encoder由4个连续的stage组成，逐步地降采样输入分辨率。每个stage中包含相同嵌入维度的blocks。因此，stage i 有2个搜索维度：**the number of blocks** $d_i$**, embedding dimension** $h_i$**。**

每一个block由一个window-based multi-head自注意力（WSA）和一个前馈网络（FFN)组成。在此不要求每个stage中的block完全相同。在第i个stage中的block j有一些搜索维度：**window size** $w_j^{i}$**,number of heads**  $n_j^{i}$**,MLP ratio** $m_i$**,Q-K-V embedding dimension** $q_j^{i}$**.**

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/18.jpg" alt="img" style="zoom:50%;" />

#### Searching the Search Space

**搜索空间质量评估**

给定一个空间 $\mathcal{A}$,提出E-T error来度量，表示为 $\mathcal{Q}(\mathcal{A})$。它是以下2部分的平均值：expected error rate $\mathcal{Q}_e(\mathcal{A})$和top-tier error $\mathcal{Q}_t(\mathcal{A})$。

- expected error rate $\mathcal{Q}_e(\mathcal{A})$定义如下：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/19.jpg" alt="img" style="zoom:67%;" />

其中， $e_\alpha$是在ImageNet验证集上的top-1错误率。g()是计算代价函数，c是计算资源限制。整个函数度量搜索空间的整体质量。在实践中，使用N个随机采样的架构来近似这一项。（**求N个随机采样的架构在ImageNet验证集上的错误率的平均值**）

- top-tier error $\mathcal{Q}_t(\mathcal{A})$：**在资源限制下top50候选架构的平均错误率**，表示搜索空间的性能上界。

**搜索搜索空间**

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/20.jpg)

- 首先初始化一个搜索空间 $\mathcal{A}^{(0)}$,比如（number of blocks: {2,3,4},embedding dimension:{224,256,288},window size:{7,14},number of heads:{3,3.5,4},MLP ratio:{7,8,9},Q-K-V dimensions:{224,256,288})
- 优化该初始搜索空间的权重至收敛
- 随机从该搜索空间采样N个架构
- 将超网分解成子空间。根据**搜索维度**和**阶段**来分解搜索空间 $\mathcal{A}^{(t)}$。相应的子空间表示为 $S_1^{t},S_2^{t},...,S_D^{t}$,其中D表示stage和搜索维度的点积。搜索空间为此可以表示为子空间的笛卡尔乘积：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/21.jpg" alt="img" style="zoom:50%;" />

之后，通过更新每一个 $S_i^{(t)}$,将搜索空间 $\mathcal{A}^{(t)}$演化成更好的 $\mathcal{A}^{(t+1)}$。

- 对于子空间 $S_i^{(t)}$,对该子空间中的每一个选择从N个架构集合中找到相应的子空间架构，计算相应的E-T Error。（比如，对于blocks指标， $S_{blocks} ^{(t)} = \{2,3,4\}$,计算分别 $v_l^{(t)}=2$、3、4时相应模型架构的E-T Error，E-T Error和 $v_l^{(t)}$的关系可以用一条线性直线来拟合。）
- 优化此子空间

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/22.jpg)

### 分析讨论

- 第三阶段是最重要的，增加第三阶段的Blocks导致更好的性能(下图第一行）。
- 较深的层应该使用更大的MLP ratio（下图第四行）。传统来说，所有层使用相同的MLP ratio,然而下图第四行展示较浅的层使用较小的MLP ratio、较深的层使用较大的MLP ratio导致更好的性能。
- 较浅的层应该用一个小的窗口，较深的层应该用一个更大的窗口（下图最后一行）。随着搜索步骤的增加，窗口大小空间在第1阶段减小，而在第3阶段增大。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/23.jpg)

### 实验结果

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/24.jpg" alt="img" style="zoom:80%;" />

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/25.jpg" alt="img" style="zoom:80%;" />

#### 消融实验

为了验证搜索空间演化的有效性，做了2个实验。

第一个实验随机采样继承超网权重的1000个架构，之后计算和绘制误差经验分布图。将初始搜索空间、one-step演化后的空间、two-step演化后的空间分别标记为Space A, Space B和Space C。如图6所示，空间C明显比空间A和B质量更好。

第二个实验，我们使用演化算法来搜索每个空间中的顶级结构。表5展示space C有更好的准确率。结果证实了搜索空间演化的有效性。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/1-NAS/26.jpg" alt="img" style="zoom:67%;" />
