---
title: 扩散模型（二）| IDDPM
date: 2023-12-26
mathjax: true
cover: false
category:
 - Diffusion model
tag:
 - IDDPM

---

> 学习笔记
>
> 论文：https://arxiv.org/pdf/2102.09672.pdf 
>
> 代码：https://github.com/openai/improved-diffusion
>
> 参考：
>
> 1. https://blog.csdn.net/weixin_42486623/article/details/130505935
> 2. https://zhuanlan.zhihu.com/p/641013157
> 3. https://zhuanlan.zhihu.com/p/557971459
> 4. https://www.youtube.com/watch?v=JbfcAaBT66U
> 5. https://zhuanlan.zhihu.com/p/109342043、https://zhuanlan.zhihu.com/p/655398520

## 生成模型的评价指标

### FID（Fréchet Inception Distance）

把真实的图像和生成的图像分别送入分类器中，可以得到两个不同的特征分布。假设这两种分布都是高斯分布，利用Fréchet距离可以求出两个分布的距离。FID距离越小代表生成的分布和真实分布越接近，生成效果越好。

设真实图片和生成图片的特征均值: $u_r,u_g$;真实图片和生成图片的协方差矩阵： $ Σ_r, Σ_g$；公式为：

 $$FID(R, G) = ||μ_r - μ_g||^2 + Tr(Σ_r + Σ_g - 2(Σ_r*Σ_g)^{1/2})$$

FID分数和人类的视觉判断是比较一致的，并且计算复杂度不高；缺点是高斯分布的假设过于简化。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/0.jpg" alt="img" style="zoom:30%;" />

### IS(Inception Score)

从清晰度和多样性两方面考虑。

- 清晰度：从单一图片考虑，此图片可以被分类器精准的划分到某个具体的类别中，即该图片数据某一类的概率大，属于其他类别的概率都很小，因此希望 $p(y|x)$的熵很小（熵代表混乱度和不确定性；类别概率分布越集中，熵越小；越分散，熵越大），即最小化 $H(y|x)$。
- 多样性：从所有图片考虑，希望生成的图片具有多样性，即每个类别的数目差不多。即希望 $p(y)$的熵很大，即最大化 $H(y)$。

综上，希望最大化 $H(y)-H(y|x)=E_{x\sim p_g}D_{KL}(p(y|x)||p(y))$。

为了便于计算，添加指数项， $IS = exp(E_{x\sim p_g}D_{KL}(p(y|x)||p(y)))$。

其中， $p(y)\approx\frac{1}{N}\sum_ip(y^{(i)})$。

缺点：当生成模型过拟合时，生成器只记住了训练集的样本，泛化能力差，但IS无法检测到这个问题。IS只考虑模型生成图像的分布而忽略了训练数据集的分布。

### NLL（Negative Log-Likelihood)

评估概率模型的拟合程度，它等于训练集样本在 $p_g$分布下的对数似然函数的负数。

 $NLL = -\sum_{i=1}^{N} \log P(x_i;\theta)$

NLL指标越小，表示模型对数据集的拟合程度越好。

### CLIP-Score

CLIP Score用于评价text-to-image任务中生成的图像和文本对之间的关联程度。其将自然语言和图像分别转换为相应特征向量，之后计算它们的余弦相似度。CLIP分数越高（接近于1），图像文本对之间关联度更高，更匹配。

## 动机

DDPM的FID指标和IS指标结果优越，能生成高质量样本；然而，DDPM不能完成和其他模型相比有竞争力的NLL指标。

NLL指标是一个在生成模型中广泛使用的指标，优化对数似然使生成模型更好地拟合数据分布。最近的工作展示在对数似然指标上得到小的提升可以对样本质量和学习到的特征分布有极大的影响。

因此，本文探索为什么DDPM在对数似然指标上表现较差，提出对DDPM的三点改进，使得DDPM完成更优越的对数似然结果，同时维持高的样本生成质量。

## 概览

1. DDPM中方差项固定，不可学习；IDDPM引入了方差项的学习，损失项加上了惩罚项 $L_{vlb}$。学习方差的好处：

- 可以提高NLL指标
- 允许逆向过程的采样步数减少一个数量级，样本质量几乎不变。DDPM需要上百步来产生好的样本，IDDPM仅仅需要50步，极大的加速了采样过程。（后面和DDIM进行了对比）

1. DDPM的方差是linear schedule；IDDPM改进schedule，采用cosine schedule。
2. DDPM中是均匀采样；IDDPM提出重要性采样。当使用重要性采样时，单纯优化 $L_{vlb}$的NLL指标比优化 $L_{hybrid}$的效果更好。

## DDPM回顾

前向传播过程初始定义：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/1.jpg" alt="img" style="zoom:67%;" />

如果T足够大，最终 $x_T$满足高斯分布；如果我们知道逆向分布 $q(x_{t-1}|x_t)$，从高斯分布中采样一个噪声 $x_T$，通过很多逆向过程就可以推导出 $x_0$。然而， $q(x_{t-1}|x_t)$依赖整个数据分布，无法直接求出。假设其是一个高斯分布：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/2.jpg" alt="img" style="zoom:67%;" />

可以写出它的变分下届（VLB，variational lower bound)：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/3.jpg" alt="img" style="zoom:67%;" />

【注意：这里的 $L_{vlb}$相当于真实的VLB的负数。即本来应该使VLB越大越好，现在变成使 $L_{vlb}$越小越好】

式子（4）中 $L_T$不依赖于网络参数 $\theta$， $L_0$可以计算出，重点看中间的 $L_{t-1}$项。我们希望 $L_{t-1}$尽可能小，即希望这两个分布接近，即可以使用 $q(x_{t-1}|x_t,x_0)$的分布拟合我们想计算的 $p_\theta(x_{t-1}|x_t)$。结合重参数化技术和贝叶斯定理，可以推出 $q(x_{t-1}|x_t,x_0)$的分布：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/4.jpg" alt="img" style="zoom:67%;" />

式子（11）还可以被写成：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/5.jpg" alt="img" style="zoom:67%;" />

DDPM中，网络可以通过预测 $x_0$来拟合均值，也可以通过预测噪声来拟合均值。实验发现预测噪声效果更好。预测噪声使用MSE平方误差损失：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/6.jpg" alt="img" style="zoom:67%;" />

可以发现 $L_{simple}$**并没有学习方差**，仅仅学习均值。DDPM实验发现，当方差设置为 $\sigma_t^2=\beta_t$**或者** $\sigma_t^2=\widetilde\beta_t$**时，样本生成质量差不多一致（** $\beta_t$**和** $\widetilde\beta_t$**分别是方差的上界和下界）**。

## IDDPM

### 学习方差

首先探究为什么方差设置为 $\sigma_t^2=\beta_t$或者 $\sigma_t^2=\widetilde\beta_t$时，样本生成质量差不多一致。作者进行实验，发现只有在趋近t=0时，$\beta_t$和 $\widetilde\beta_t$不太一致，其余情况下它们均几乎相等。此外，当增加扩散步数，$\beta_t$和 $\widetilde\beta_t$在更多的扩散步骤中保持一致（更加紧密）。这表示在扩散步数趋于无限时，方差的选择对样本生成质量来说不是很重要。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/7.jpg" alt="img" style="zoom:67%;" />

然而，对于极大似然来说，如下图所示，diffusion过程中步数较小时对变分下界的优化最重要。因此，我们可以通过设计一个更好的方差学习方案来优化对数似然。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/8.jpg" alt="img" style="zoom:67%;" />

因为方差 $Σ_\theta(x_t,t)$的变化范围很小，所以直接训练网络预测方差是比较困难的。所以，IDDPM提出来优化$\beta_t$和 $\widetilde\beta_t$的差值。模型输出一个向量 $v$，方差如下计算：

 $Σ_\theta(x_t,t)=exp(vlog\beta_t+(1-v)log \widetilde\beta_t)$

设计新的损失函数：

 $L_{hybrid}=L_{simple}+\lambda L_{vlb}$

为防止 $L_{vlb}$过度影响 $L_{simple}$，实验中将 $\lambda$设置为0.001，同时在优化方差项时，对均值项输出采用一个stop-gradient策略。

实际实现时，直接使用一个网络同时预测均值和方差系数，输出一半用来预测噪音，一半用来预测方差系数v。

代码：https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py#L642

### Cosine schedule

Linear schedule更适用于高分辨率图像，不太适合低分辨率图像，它的前向过程添加噪声太快了：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/9.jpg" alt="img" style="zoom:80%;" />

如下图所示，可以看出一个使用linear schedule的模型当删掉20%的逆向过程时，还可以取得高的FID得分（证明很多步骤比较冗余）：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/10.jpg" alt="img" style="zoom:67%;" />

为了解决这个问题，让扩散模型学习更多的细节，IDDPM提出cosine schedule：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/11.jpg" alt="img" style="zoom:67%;" />

cosine schedule的alpha值随着时间步缓慢改变，防止突变：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/12.jpg" alt="img" style="zoom:67%;" />

 $\beta_t = 1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}$，为了防止当t很小的时候 $\beta_t$太小，添加offset s，因为实验发现当一开始噪声添加太小会影响噪声预测。s = 0.008。

### 重要性采样

IDDPM期望直接通过优化 $L_{vlb}$来完成最好的log-likelihoods，而不是直接优化 $L_{hybrid}$。然而作者发现直接优化 $L_{vlb}$是困难的。

作者假设 $L_{vlb}$的梯度中噪声比$L_{hybrid}$更多。图2可以看出，不同的扩散步骤中噪声是不同的。假设均匀采样时间步t造成 $L_{vlb}$优化过程中不必要的噪声。于是提出importance sampling：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/13.jpg" alt="img" style="zoom:67%;" />

通过importance sampling，直接优化 $L_{vlb}$可以达到最好的log-likelihoods结果。然而，这项技术对优化较少噪声的 $L_{hybrid}$帮助不大。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/14.jpg" alt="img" style="zoom:67%;" />

## 实验

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/15.jpg" alt="img" style="zoom:67%;" />

## 加快采样速度

$L_{hybrid}$模型可以减少很多扩散步数，同时产生高质量样本。

可以使用任意的具有t个值的序列S来采样。给定 $\bar\alpha_t$，对于给定的序列S可以得到 $\bar\alpha_{S_t}$，进而可以得到：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/16.jpg" alt="img" style="zoom:67%;" />

实践中，DDPM和IDDPM使用1和T之间的K个等距实数得到具有K个时间步值的序列。评估结果如下图所示，可以看出对于IDDPM来说，t=100就可以得到不错的FID分数。对于DDIM，发现DDIM可以在采样步数小于50时得到更好的结果，但是当使用更多的步数，DDIM效果不如IDDPM。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-diffusion-models/17.jpg" alt="img" style="zoom:67%;" />
