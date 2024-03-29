---
title: 模型架构｜MQA & GQA 调研
date: 2023-08-30
mathjax: true
cover: false
category:
 - model architecture
tag:
 - Tranformer
---

## MQA(Multi-Query Attention)

> Fast Transformer Decoding: One Write-Head is All You Need  (Google)
>
>  https://arxiv.org/pdf/1911.02150.pdf

### 动机

Transformer在训练过程中可以并行处理所有序列，所以其训练过程是快速高效的；但是其推理过程是串行的，其需要重复地加载大的键值对，加载键值对的过程耗时耗力，对内存带宽（memory-bandwidth)需求大，造成计算瓶颈（计算过程很快，但是数据加载过程慢，处理单元在空等数据，导致计算力的浪费）。本文提出MQA，键和值在所有不同注意力头中共享，需要加载的key和value矩阵的参数量变小，从而减少了增量解码过程对内存带宽的需求，提高推理速度，同时性能相比baseline只有轻微的下降。

### 方法

- MHA(Multi Head Attention): 多头注意力中query、key、value都有h个头。
- MQA(Multi Query Attention): 只有query是h个，其余key和value都是1个。**MQA和MHA的唯一区别在于MQA中不同的query头之间共享同一份key和value矩阵。**

### 代码

**多头注意力vs多查询注意力**

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-model-architecture/0.jpg)

代码看出其区别只在于建立K和V上。多查询注意力中K和V是单头的。计算attention的时候所有的query头共享同一份K和V矩阵，即所有query头乘以同一个key。

### 实验

- 模型质量

  table1可以看出MQA比MHA略差，但是比其它任何一个减小head、d_v、d_k等替换方案结果都要好。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-model-architecture/1.jpg" alt="img" style="zoom:80%;" />

- 速度：

  可以看出MQA的训练速度变快，推理速度变快得更加明显。推理速度中decoder部分速度极大地提高。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-model-architecture/2.jpg)

## GQA(Grouped-Query Attention)

> GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Google)
>
>  https://arxiv.org/pdf/2305.13245.pdf

### 动机

MQA仅使用单个键值头，大大加快了解码器推理速度。然而，MQA可能会导致质量下降。此外，虽然一些语言模型如PaLM已经使用了多查询注意力，但许多语言模型并没有使用多查询注意力，包括公开可用的模型如LLaMA、T5等，为此提出GQA，它的贡献点有两个：

1. 提出一种使用5%的原始预训练计算将现有的多头注意力模型升级为一个多查询注意力模型的方法
2. 引入分组查询注意力(GQA)，使用多个但是少于query头数量的键值头去训练模型

实验结果表明GQA实现了接近多头注意力的质量，且速度和MQA相当。

### 方法

#### Uptraining

将一个多头注意力模型转变成多查询注意力模型分为两步：

1. 转换checkpoint
2. 添加预训练来允许模型适应这个新架构

下图显示了将多头checkpoints转换为多查询checkpoints的过程，多头注意力中所有键和值头的投影矩阵被平均池化到单个投影矩阵中，作者发现这个比选择单个键和值头或者从头开始随机初始化新的键值头效果更好。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-model-architecture/3.jpg" alt="img" style="zoom:67%;" />

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-model-architecture/4.jpg" alt="img" style="zoom:67%;" />

对转换后的checkpoint再次预训练，只需要使用原始预训练设置在一小部分预训练步骤（ $\alpha $比例）上进行训练即可。

#### Grouped-query attention

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-model-architecture/5.jpg" alt="img" style="zoom:80%;" />

GQA将query头分为G组，每组共享一个键头和值头。GQA-G指G组的分组查询。GQA-1具有1个组，因此相当于有一组键和值头，相当于MQA（多查询注意力）。GQA-H具有和query相同的键值头，相当于MHA（多头注意力）。

### 实验

- 性能&速度

  如下图所示，GQA-XXL性能上比MQA-XXL好，速度上比MHA-XXL快。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-model-architecture/6.jpg" alt="img" style="zoom:50%;" />

- 分组个数
  - 对于分组的个数，如下图所示，一开始KV缓存的内存带宽开销不那么紧张，随着分组个数增加，内存带宽逐渐紧张，成本增大。我们选择8组作为有利的中间地带。

    <img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-model-architecture/7.jpg" alt="img" style="zoom:50%;" />
  
- uptraining steps

  MQA和GQA都从5%的追加训练中获益，收益从10%递减。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/2-model-architecture/8.jpg" alt="img" style="zoom:50%;" />

### 代码

```Python
 #LLaMA 2:https://github.com/facebookresearch/llama/blob/4d92db8a1db6c7f663252bf3477d2c4b8bad2385/
 def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
```