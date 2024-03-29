---
title: 扩散模型（十）| LoRA
date: 2024-02-28
mathjax: true
cover: false
category:
 - Diffusion model
tag:
 - LoRA
---

> LoRA学习笔记
>
> 论文：https://arxiv.org/abs/2106.09685
>
> 学习主要参考：https://zhuanlan.zhihu.com/p/618894919
>
> 代码：https://github.com/microsoft/LoRA/blob/main/loralib/layers.py

## 原理

模型是过参数化的，它们有更小的内在维度，模型主要依赖这个低的内在维度去做任务适配。

假设模型在适配任务时参数的改变量是低秩的，由此引出低秩自适应方法LoRA,通过低秩分解来模拟参数的改变量，从而以极小的参数量来实现大模型的微调。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/10-diffusion-models/0.jpg" alt="img" style="zoom:67%;" />

## 做法

### 步骤

- 在原模型旁边增加一个支路，通过低秩分解（先降维再升维）来模拟参数的改变量
- 训练时，原模型固定，只训练降维矩阵A和升维矩阵B
  -  前向过程： $Wx = W_0x+\Delta Wx = W_0x + BAx$

  -  缩放 $\Delta Wx$：对低秩输出部分会乘上一个scale系数 $\frac{\alpha}{r}$，缩放帮助我们在改变r时减少重新调节超参数的需求。当用Adam优化时，微调 $\alpha$相当于微调学习率，所以论文中我们直接将 $\alpha$设置为常数，不去微调它。
- 推理时，训练好的低秩矩阵可以合并到原参数上，多分支变成单分支，不引入额外的推理延迟
- 初始化：A采用高斯分布初始化，B初始化为全0，保证训练开始时旁路为0矩阵

**保证权重矩阵的种类的数量比起增加秩的维度r更重要，增加r并不一定能覆盖更加有意义的子空间。**

对于一般的任务，rank=1,2,4,8足矣，rank=1就挺好的。秩不是越大越好，大了可能会增加一些噪声。

### 具体对模型的哪些部分做低秩分解

仅仅将LoRA应用于自注意力层中的投影矩阵（Q,K,V,O)，而MLP模块以及self-attention层以外的结构均不使用。

实验结果表明，模型更倾向于我们对更多类型的投影矩阵应用LoRA(对4个投影矩阵应用LoRA时效果是最好的，尽管秩很低，也足以让△W捕捉足够信息)

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/10-diffusion-models/1.jpg)

### LoRA代码实现

- 先用预训练权重 $W_0$对输入 $x$实施前向过程，得到 $W_0x$
- 再将输入 $x$喂给低秩分解矩阵 $\Delta W=BA$，得到输出 $\Delta Wx = BAx$
- 接着对$\Delta W = BAx$作零填充使其与$W_0x$shape一致，并进行缩放(由于不一定会对整个预训练权重矩阵做低秩分解，所以 $\Delta W = BAx$的shape不一定等于 $W_0x$，要对前者进行padding，使其与后者的shape一致，才能让两者element-wise add）
- 将这部分结果加回 $W_0x$中

前向传播代码：

```Python
def forward(self,x:torch.Tensor):
    result = F.linear(x,transpose(self.weight,self.fan_in_fan_out),bias=self.bias)
    if self.r > 0:
        after_A = self.lora_A(self.lora_dropout(x))
        after_B = self.lora_B(after_A.transpose(-2,-1)).transpose(-2,-1)
    result += self.zero_pad(after_B)*self.scaling
    
    return result
```

填充部分代码：

```Python
 def zero_pad(self,x):
     #创建一个形状与x的形状相同，但最后一个维度的大小为self.out_features的全零张量
     result = x.new_zeros((*x.shape[:-1],self.out_features))
     result = result.view(-1,self.out_features)
     result[:,self.lora_ind] = x.reshape(-1,self.out_features//len(self.enable_lora)*sum(self.enable_lora))
     
     return result.view((*x.shape[:-1],self.out_features))
```

Lora类

```Python
class MergedLinear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        # enable_lora 是一个布尔类型的列表，用于指示对权重矩阵的哪些“子部分”做低秩分解。
        # 比如 W 是 shape 为 (out_features, in_features) 的矩阵，
        # 那么 enable_lora = [True, False, True] 就表示将 W 在 out_features 这个维度上按序均分成三部分 W1, W2, W3，
        # shape 均为 (out_features // 3, in_features)，然后仅对 W1 和 W3 做低秩分解。
        # 其中 W1 的第一个维度取值范围是 [0, out_features // 3)，W3 则是 [2 * out_features // 3, out_features)。
        # 同理，若 enable_lora = [True]，就表示对整个 W 都做低秩分解。
        if out_features % len(enable_lora) != 0:
            raise ValueError("The length of enable_lora must divide out_features")
        
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            # 仅 enable_lora = True 的部分应用低秩分解，每部分的 low-rank 是 r
            self.lora_A = nn.Linear(in_features, r * sum(enable_lora), bias=False)
            # 注意下这里 B 是用一维的分组卷积实现的
            '''
            nn.conv1d，kernel_size=1相当于mlp 
            https://blog.csdn.net/qq_36323559/article/details/102937606
            nn.conv1d,kernal_size=1,group=3相当于qkv矩阵分别与3个linear层投影，之后再concat起来
            如果做lora的是Q和V，Q的特征维度是d，V的特征维度是d，分别过线性层特征维度变成f，Q和V concat
            起来后特征维度变为2f；如果是2个合起来过一个线性层，过的线性层维度为[2d,2f],总参数量是4df；如果
            分别过linear层对其投影再concat起来，过的每个线性层维度为[d,f]，一共2个，总参数量2df
            '''
        
            self.lora_B = nn.Conv1d(
                r * sum(enable_lora),
                out_features // len(enable_lora) * sum(enable_lora),
                kernel_size=1,
                groups=2, #LoRA默认只对attention weights中的2个矩阵做低秩分解
                #在这里可以把groups=2改成sum(enable_lora)
                bias=False,
            )

            # scale 系数，对低秩矩阵的输出(BAx)做缩放
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            # 固定住预训练权重
            self.weight.requires_grad = False

            # Compute the indices
            # 记录权重矩阵中，做了低秩分解的是哪些“子矩阵”
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)

        self.reset_parameters()
        if fan_in_fan_out:
            # fan_in_fan_out 是针对 GPT-2 的 Conv1D 模块的，
            # 该模块和 Linear 的区别就是维度互为转置
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
```

> 分组卷积示例图：
>
> <img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/10-diffusion-models/2.jpg" alt="img" style="zoom:50%;" />

低秩分解部分如何合并到预训练权重中（无推理延迟）：

W' = W+AB

使用一维卷积将A和B相乘，（一维卷积中A当作输入，B当作卷积权重），再与原始权重相加。

ps: 因为上面是用分组卷积实现的，所以这里也用分组卷积（否则权重维度不对应）

```Python
 def train(self, mode: bool = True):
    nn.Linear.train(self, mode)
    self.lora_A.train(mode)
    self.lora_B.train(mode)

    # 注：当调用 model.eval() 时就会调用 train(mode=False)
    # 将低秩矩阵 A, B 合并至原权重矩阵 W
    if not mode and self.merge_weights and not self.merged:
        # Merge the weights and mark it
        if self.r > 0 and any(self.enable_lora):
            # \delta_W = BA
            delta_w = (
                # 这里使用1维卷积将低秩矩阵 A, B 进行“融合”：
                # A(r * k) 作为输入，r 看作是其 channel，k 看作是空间维度上的大小；
                # B(d * r * 1) 作为卷积权重，d 是 output channel, r 是 input channel, 1 是 kernel size(注意B本身就是用1维分组卷积实现的)。
                # 由于是卷积，因此二维的 A 需要增加一维给 mini-batch：r * k -> 1 * r * k。
                # 卷积后，输入(1 * r * k) -> 输出(1 * d * k)
                F.conv1d(
                    self.lora_A.weight.data.unsqueeze(0),
                    self.lora_B.weight.data,
                    groups=sum(self.enable_lora),
                )
                .squeeze(0)  # 1 * d * k -> d * k
                .transpose(-2, -1)  # d * k -> k * d
            )
            # zero_pad() 是对低秩分解矩阵 \delta_W 进行0填充，因为原权重矩阵 W 中可能有些部分没有进行低秩分解，
            # 从而得到一个和原权重矩阵 W 的 shape 对齐的结果，以便进行加和。k * d -> k * D(假设 D 是原权重矩阵 W 的 out features)
            # 对于原权重矩阵 W 是 Linear 层的情况，fan_in_fan_out = False，于是这里会进行 transpose: k * D -> D * k；
            # 而对于原权重矩阵 W 是 GPT-2 的 Conv1D 的情况，fan_in_fan_out=True，于是不需要 transpose，它的 out features 就是放在第二维的
            # W = W + # \delta_W
            self.weight.data += transpose(self.zero_pad(delta_w * self.scaling), not self.fan_in_fan_out)
    elif xxx:
        ...
```

merge完后，在进行前向过程时就无需再像上一节展示的那样分步进行，而是一步到位(见以下第二个分支)：

```Python
def forward(self, x: torch.Tensor):
    # 此部分先省略，下一节再介绍
    if xxx:
        ...
    # 低秩分解部分已合并
    elif self.merged:
        return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    # 低秩分解部分未合并
    else:
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.r > 0:
            after_A = self.lora_A(self.lora_dropout(x))
            after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
            result += self.zero_pad(after_B) * self.scaling

        return result
```

同理，当模型在某个下游任务A微调后，可以将低秩矩阵部分的参数解耦出来，还原出预训练权重，继续在另一个下游任务B上微调

```Python
 def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)

        if xxx:
            ...
        # 前一个分支是代表 mode=False，进入该分支说明 mode=True，即调用了 model.train()，
        # 那么当低秩矩阵 A, B 已经合并至原权重矩阵 W 中时，就需要将它们分解出来，以便进行训练(预训练权重 W 无需训练)。
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                # \delta_W = BA
                delta_w = (
                    F.conv1d(
                        self.lora_A.weight.data.unsqueeze(0),
                        self.lora_B.weight.data,
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .transpose(-2, -1)
                )
                # W = W - \delta_W
                self.weight.data -= transpose(self.zero_pad(delta_w * self.scaling), not self.fan_in_fan_out)
            self.merged = False
```

## PEFT方法对比

PEFT（parameter-efficient fine-tuning)参数高效的微调方法

### Adapter Layers

Adapter层嵌入在Transformer结构里面，具体位置是在feed-forward层之后。在训练时，固定住原模型参数，只对新增的adapter结构进行微调。

Adapter层内部：

- 首先是一个down-project层将高纬度特征映射到低维特征
- 经过一个非线性层之后，再用一个up-project结构将低维度特征映射回高维特征
- 添加skip-connection结构，确保在最差情况下能够退化为原模型

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/10-diffusion-models/3.jpg" alt="img" style="zoom:50%;" />

### prefix tuning

在输入token之前构造一段任务相关的虚拟tokens作为prefix，训练时只更新prefix参数，而Transformer其他部分参数固定。

该方法和构造prompt类似，只是prompt是人为构造的显式提示，而prefix是可以学习的隐式的表示。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/10-diffusion-models/4.jpg" alt="img" style="zoom: 50%;" />

同时，为了防止直接更新prefix的参数导致训练不稳定的情况，在prefix层前面添加MLP结构（相当于将prefix分解为更小维度的input与MLP的组合后输出的结果），训练完成后，只保留prefix的参数。

```Python
embedding = torch.nn.Embedding(num_virtual_tokens, token_dim)
transform = torch.nn.Sequential(
    torch.nn.Linear(token_dim, encoder_hidden_size),
    torch.nn.Tanh(),
    torch.nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
)
```

### 对比

Lora优点：

- 一个中心模型服务于多个下游任务，节省参数存储量，增加训练效率
- 低秩矩阵可以合并到预训练权重中，多分支结构变成单分支，不引入额外的推理延迟
- 与其它参数高效微调方法（Adapter,prefix-tuning)正交，可有效组合
- 训练稳定，效果好

Adapter Layer的缺点：

- 新增的adapter层必须串行处理，增加了推理延迟

prefix tuning的缺点：

- 方法本身难以优化；方法需要在模型的输入序列中预留一部分用作可微调的prompt，从而限制了原始输入文本的句长
