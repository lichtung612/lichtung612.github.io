---
title: 扩散模型（十一）| Transformer-based Diffusion：DiT
date: 2024-02-20
mathjax: true
cover: false
category:
 - Diffusion model
tag:
 - Transformer-based diffusion
---

>  Scalable Diffusion Models with Transformers-23ICCV oral
>
> https://arxiv.org/pdf/2212.09748.pdf
>
> https://github.com/facebookresearch/DiT
>
> 参考：https://zhuanlan.zhihu.com/p/641013157

## 概要

- 使用transformer架构代替常用的U-Net架构
- 更高Gflops的DiT（增加transformer depth/width或者输入token的数量）得到较低的FID，有更好的生成能力。

## 方法

DiT采用**IDDPM**方法（IDDPM解读见https://lichtung612.github.io/posts/2-diffusion-models/ ），同时预测模型的**噪声和方差**。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/9-diffusion-models/0.jpg)

### Patchify

DiT的输入是一个空间潜在表示 z（对于256x256x3的图像来说，z的shape为32x32x4）。

输入首先要经过patchify变成包含T个tokens的序列，每个token的维度是d。

之后使用sine-cosine位置编码来编码输入tokens。

其中，token的数量T由patch size超参数p来决定。token数量增大2倍，至少使总的Gflops增大4倍，但是不会影响下游参数数量。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/9-diffusion-models/1.jpg" alt="img" style="zoom:50%;" />

### DiT block design

以类别条件和时间条件为例，探究4种向transformer blocks中注入条件的方式：

- In-context conditioning：简单的将时间t和class label c当作2个额外的tokens，对待它们与图像token没有区别。
- Cross-attention block：将t和c的嵌入concat成一个长度为2的序列，通过一个额外的多头cross-attention层注入，latent为query，条件embeddings作为key和value。这种方式会额外引入较大的计算量。
- Adaptive layer norm (adaLN) block：将transformer blocks里的标准Layer norm层替换成adaptive layer norm，回归shift和scale参数。这种方式基本不增加计算量。
- adaLN-Zero block：将adaLN的linear层参数初始化为0（在ResNets上的研究表明对每个残差块零初始化，相当于一个identity函数，对于训练网络是有利的）。同时，除了回归scale和shift，还在每个残差块结束前添加一个gate参数。

实验发现，adaLN-Zero方法效果最好。【“虽然DiT发现adaLN-Zero效果最好，但这种方式可能只适合只有类别信息的简单条件嵌入，因为只需要引入一个class embedding；对于文生图来说，其条件往往是序列的text embeddings，采用cross-attention方案可能更合适”】

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/9-diffusion-models/2.jpg" alt="img" style="zoom:67%;" />

>  https://github.com/facebookresearch/DiT/blob/main/models.py#L101

```python
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # zero init
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
        
        
 def modulate(x, shift, scale):
     return x * (1+scale.unsqueeze(1)) + shift.unsqueeze(1)
```



### Transformer decoder

由于对输入进行token化，需要在网络的最后添加一个decoder来恢复原始输入维度。需要预测噪声和方差系数两项，它们都和原始空间输入维度相同。使用一个包含linear层和adaLN-Zero层的decoder来实现(linear层也采用zero初始化），输出为 $p \times p \times 2C$，输出特征维度是之前的2倍，分别对应噪声和方差系数。

> https://github.com/facebookresearch/DiT/blob/main/models.py#L125

```python
 class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        #zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
```



## 实验

设计了4种大小的模型：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/9-diffusion-models/3.jpg" alt="img" style="zoom:67%;" />

探究模型的缩放能力，发现模型计算量对生成效果至关重要，计算量越大，生成质量越高：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/9-diffusion-models/4.jpg" alt="img" style="zoom:67%;" />

性能上，最大的模型在classifier free guidance下可以达到sota：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/9-diffusion-models/5.jpg" alt="img" style="zoom:80%;" />