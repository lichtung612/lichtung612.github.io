---
title: 扩散模型（三）| Classifier Guidance & Classifier-Free Guidance
date: 2024-01-09
mathjax: true
cover: false
category:
 - Diffusion model
tag:
 - Classifier-Free Guidance
---

> 学习笔记，学习资料：
>
> 1. https://zhuanlan.zhihu.com/p/660518657
> 2. https://zhuanlan.zhihu.com/p/642519063
> 3. https://www.zhihu.com/question/582965404/answer/3380307490

## Classifier Guidance

思想：给定一个分类模型中存在的类别，让模型生成这个类别的东西。

比如指定模型生成图像的类别是“狗”，模型就生成一张狗的图。所以这种方式是条件生成，条件是 $y$，扩散过程中的生成图像是 $x_t$。

公式上，用贝叶斯定理将条件生成概率进行对数分解：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/3-diffusion-models/0.jpg" alt="img" style="zoom:80%;" />

第二个等号后面最后一项消失了，因为当我们要求模型生成“狗”的图像时，扩散过程始终 $y$不变，对应的梯度也是0，可以抹掉。

第三个等号后面两项中，第一项是扩散模型本身的梯度引导，新增的只能是第二项，即classifier guidance只需要额外添加一个classifier的梯度来引导。

```Python
 #加载一个训练好的图像分类模型
 classifier_model = ...
 #生成类别为1的图像，假设类别1对应“狗”这个类
 y = 1
 #控制类别引导的强弱，越大越强
 guidance_scale = 7.5
 #从高斯分布随机抽取一个跟输出图像一样shape的噪声图
 input = get_noise(...)
 
 for t in tqdm(scheduler.timesteps):
     #使用unet推理，预测噪声
     with torch.no_grad():
         noise_pred = unet(input,t).sample
     
     #用input和预测出的noise_pred和x_t计算得到x_{t-1}
     input = scheduler.step(noise_pred,t,input).prev_sample
     
     #classifier guidance步骤
     classifier_guidance = classifier_model.get_class_guidance(input,y)
     input += classifier_guidance * guidance_scale #把梯度加上去
```

在推理过程，从 $x_t$得到 $x_{t-1}$后，将 $x_{t-1}$作为输入图片、 $y$作为标签送入分类模型，计算分类loss得到 $x_{t-1}$的梯度（正常的分类模型的参数是网络连接层的权重参数，这里面输入图像本身也是可学习的参数，并且对我们有用的就是输入图像本身的梯度），把梯度乘以guidance_scale系数，添加到 $x_{t-1}$上，得到更新后的图像。

## Classifier-Free Guidance

Classifier guidance只能用分类模型控制生成的类别，生成的类别数有限。classifier-free guidance虽然需要重新训练diffusion模型，但是训好后没有类别数量的限制。

以文生图为例：

```Python
 clip_model = ... #加载一个官方clip模型
 
 text = "一只小狗"
 text_embeddings = clip_model.text_encoder(text) #编码条件文本
 empty_embeddings = clip_model.text_encoder("") #编码空文本
 text_embeddings = torch.cat(empty_embeddings,text_embeddings)#把空文本和条件文本concat到一起作为条件
 
 #噪声图
 input = get_noise(...)
 
 for t in tqdm(scheduler.timesteps):
     with torch.no_grad():
         #同时预测有空文本和有文本的图像噪声
         noise_pred = unet(input,t,encoder_hidden_states=text_embeddings).sample
         
     #classifier-free guidance引导
     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)#拆成无条件和有条件的噪声
     #把[无条件噪声指向有条件噪声]看作一个向量，根据guidance_scale的值放大这个向量
     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text-noide_pred_uncond)
     
     #使用预测出的noise_pred和x_t计算得到x_{t-1}
     input = scheduler.step(noise_pred,t,input).prev_sample
```

如下图所示，红色的箭头表示从“无条件”到“一只狗”条件的向量，通过调节guidance_scale的大小，可以控制文本条件噪声贴近文本语义的程度。

如果想让生成的图更遵循“一只狗”的文本语义，就把guidance_scale设大一点，生成的图像会更贴近“一只狗”的文本语义，但是多样性会降低。反之如果想让生成的图像更多样丰富，就把guidance_scale设小一点。通常guidance_scale取值设为7.5比较合适。

总体而言，classifier-free guidance需要在训练过程中同时训练模型的两个能力，一个是有条件生成，一个是无条件生成。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/3-diffusion-models/1.jpg" alt="img" style="zoom:50%;" />

不同guidance_scale下的图像效果：

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/3-diffusion-models/2.jpg)

### 推导

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/3-diffusion-models/3.jpg)

### U-Net模型如何融入语义信息

#### CrossAttention

首先通过`embedding layer`或者是clip等模型将文本转换为文本特征嵌入，即text embedding过程。

之后text embedding和原本模型中的image进行融合。最常见的方式是利用CrossAttention（stable diffusion采用的就是这个方法）。

具体来说是把text embedding作为注意力机制中的key和value，把原始图片表征作为query。相当于计算每张图片和对应句子中单词的一个相似度得分，把得分转换成单词的权重，[权重乘以单词的embedding]加和作为最终的特征。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/3-diffusion-models/4.jpg" alt="img" style="zoom:80%;" />

```Python
import torch 
import torch.nn as nn
from einops import rearrange

class SpatialCrossAttention(nn.Module):

    def __init__(self,dim,context_dim,heads=4,dim_head=32):
        super(SpatialCrossAttention,self).__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head*heads

        self.proj_in = nn.Conv2d(dim,context_dim,kernel_size=1,stride=1,padding=0)
        self.to_q = nn.Linear(context_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(context_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(context_dim, hidden_dim, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self,x,context=None):
        x_q = self.proj_in(x)
        b,c,h,w = x_q.shape
        x_q = rearrange(x_q,"b c h w -> b (h w) c")
        
        if context is None:
            context = x_q
        if context.ndim == 2:
            context = rearrange(context,"b c -> b () c")

        q = self.to_q(x_q)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j',q,k)*self.scale

        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d',attn,v)
        out = rearrange(out, '(b h) n d -> b n (h d)',h=self.heads)
        out = rearrange(out,'b (h w) c -> b c h w', h=h, w=w)
        out = self.to_out(out)
        return out

CrossAttn = SpatialCrossAttention(dim=32,context_dim=1024)
x = torch.randn(8,32,256,256)
context = torch.randn(8,1024)
out = CrossAttn(x,context)
```

#### Channel-wise attention

融入方式与`time-embedding`的融入方式相同。基于channel-wise的融入粒度没有CrossAttention细，一般使用类别数量有限的特征融入，如时间embedding、类别embedding。语义信息的融入更推荐使用CrossAttention。

>  《Diffusion Models Beats Gans on Image Synthesis》https://arxiv.org/pdf/2105.05233.pdf
>
> **Adaptive Group Normalization**
>
> 组归一化即对一个图片样本中的所有像素，按通道分组进行归一化。
>
> 自适应归一化可以表示为： $AdaGN(h,y)=y_sGroupNorm(h)+y_b$
>
> 其中h是残差卷积块中第一个卷积层的输出，y_s和y_b分别是步数和图片分类的embedding向量经过线性层后的投影。实验发现，使用自适应组归一化能够进一步优化FID。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/3-diffusion-models/5.jpg" alt="img" style="zoom:80%;" />

>  code：https://github.com/huggingface/diffusers/blob/v0.27.2/src/diffusers/models/resnet.py#L353-L364

```Python
 #如果参数为“default”，则时间融合方式为相加
 #如果参数为“scale_shift”，则时间融合方式为scale_shift方法
 if temb_channels is not None:
    if self.time_embedding_norm == "default":
        self.time_emb_proj = linear_cls(temb_channels, out_channels)
    elif self.time_embedding_norm == "scale_shift":
        self.time_emb_proj = linear_cls(temb_channels, 2 * out_channels)
    else:
        raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
         
if self.time_embedding_norm == "default":
    if temb is not None:
        hidden_states = hidden_states + temb
    hidden_states = self.norm2(hidden_states)
elif self.time_embedding_norm == "scale_shift":
    if temb is None:
        raise ValueError(
            f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
        )
    time_scale, time_shift = torch.chunk(temb, 2, dim=1)
    hidden_states = self.norm2(hidden_states)
    hidden_states = hidden_states * (1 + time_scale) + time_shift
else:
    hidden_states = self.norm2(hidden_states)
```

## 二者对比

成本上看，classifier guidance需要训练噪声数据版本的classifier网络，推理时每一步都需要额外计算classifier的梯度。classifier free guidance需要重新训练diffusion模型，使其具备条件生成和无条件生成的能力，推理时需要同时预测条件生成和无条件生成的图像，通过guidance_scale来控制最终组合效果。

显然classifier-free guidance效果更好一些，既能生成无穷多的图像类别，又不需要重新训练一个基于噪声的分类模型。当前最常见的是classifier-free guidance。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/3-diffusion-models/6.jpg)
