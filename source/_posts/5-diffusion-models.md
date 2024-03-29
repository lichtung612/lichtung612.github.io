---
title: 扩散模型（五）| SDXL
date: 2024-01-16
mathjax: true
cover: false
category:
 - Diffusion model
tag:
 - SDXL
---

> SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis
>
> 论文：https://arxiv.org/abs/2307.01952
>
> 代码：https://github.com/Stability-AI/generative-models
>
> 参考：https://zhuanlan.zhihu.com/p/642496862

## 主要改进

- SDXL使用**3倍大的UNet backbone**：模型参数增长主要来源于添加更多的attention块和更大的cross-attention，因为SDXL使用了2个text encoder。
- **2个简单但是有效的额外的条件条件注入**，不需要任何形式的额外监督；在多纵横比（aspect ratio）上微调SDXL。
- 引入一个分离的diffusion-based细化模型来提升视觉效果。

## 模型架构

### VAE

以一个更大的batchsize(256 vs 9）重新训练stable diffusion的autoencoder模型，同时采用EMA。

下表可以看出，SDXL-VAE的性能最强。其中，因为SD-VAE 2.x和1.x的区别仅仅是微调了decoder部分，所以SD-VAE 1.x和SD-VAE 2.x的encoder部分权重相同，latent分布一致，两个模型权重可以互相使用。而SDXL-VAE是完全重新训练的，latent分布发生改变，因此不可以将SDXL-VAE应用到SD 1.x和SD 2.x上。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/5-diffusion-models/0.jpg)

>  EMA（Exponential Moving Average)指数移动平均——给予近期数据更高权重的平均方法
>
> https://zhuanlan.zhihu.com/p/442749399、https://zhuanlan.zhihu.com/p/68748778、https://zhuanlan.zhihu.com/p/666552214
>
> 普通的平均： $\bar v = \frac{1}{n}\sum_{i=1}^n\theta_i$
>
> EMA：$v_t = \beta v_{t-1}+(1-\beta)\theta_t$，其中v_t表示前t条的EMA平均值，$\beta$是加权权重值（一般设为0.9-0.999，为确保每次EMA更新稳定）
>
> 对模型权重进行备份（EMA_weights)，训练过程中每次更新权重时同时也对EMA_weights进行滑动平均更新。在梯度下降的过程中，会一直维护着EMA_weights，但是它**并不会参与训练**。
>
>  $weights = weights + \alpha*grad$
>
>  $EMA\_weights = EMA\_weights*decay + (1-decay)*Weights$
>
> ```Python
>  def ema(source,target,decay):
>      source_dict = source.state_dict()
>      target_dict = target.state_dict()
>      for key in source_dict.keys():
>          target_dict[key].data.copy_(
>              target_dict[key].data * decay + source_dict[key].data*(1-decay))
> ```

### U-Net

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/5-diffusion-models/1.jpg)

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/5-diffusion-models/2.jpg)

- stage上，效率考虑，只采用3个阶段，意味着只进行2次下采样。之前的SD是使用4个阶段，包含3个下采样。
- 将transformer blocks应用在更高的stage：第一个stage采用普通的DownBlock2D,而不是采用基于attention的CrossAttnDownBlock2D，主要是为了计算效率。
- 使用更多的transformer blocks（参数量增加主要原因）：第二stage和第三stage分别使用2和10块block，中间的MidBlock2DCrossAttn也使用10块block。

### Text encoder

| SD 1.x | 123M OpenAI CLIP ViT-L/14                          |
| ------ | -------------------------------------------------- |
| SD 2.x | 354M OpenCLIP ViT-H/14                             |
| SDXL   | 694M OpenCLIP ViT-bigG + 354M OPenAI CLIP ViT-L/14 |

- 使用2个text encoder，分别提取OpenCLIP ViT-bigG和CLIP ViT-L的text encoder的倒数第二层特征，concat在一起。（OpenCLIP ViT-bigG的特征维度为1280，而CLIP ViT-L/14的特征维度是768，两个特征concat在一起总的特征维度大小是2048，这也就是SDXL的context dim）
- 提取了OpenCLIP ViT-bigG的pooled text embedding（用于CLIP对比学习所使用的特征），将其映射到time embedding的维度并与之相加。（这种特征嵌入方式在强度上并不如cross attention，只是作为一种辅助）

## 条件注入

### Image size——数据利用效率问题

Latent Diffusion Model的训练通常是2阶段的（先在256x256上预训练，然后在512x512上继续训练），这导致我们需要一个最小的图像尺寸。当使用256x256尺寸训练时，要过滤掉那些分辨率小于256的图像，采用512x512尺寸训练时也同样只用512x512尺寸以上的图像。

解决的方式一种是过滤数据。由于需要过滤数据，这就导致实际可用的训练样本减少了，如果要过滤256以下的图像，就其实丢掉了39%的训练样本。另一种方式是利用超分模型增大尺寸较小的图片的尺寸。但是这通常会引入放大伪影，可能会渗入最终模型的输出中，导致生成模糊的样本。

SDXL提出**将图像的原始尺寸(width and height)作为条件** $c_{size}=(h_{original},w_{original})$**嵌入到****U-Net****模型中，让模型学习到图像分辨率参数**。每一个分辨率参数采用Fourier特征编码(Fourier特征编码即sinusoidal embedding)，这些编码被concat成一个vector，被添加到timestep embedding中送入模型。

在训练过程中，可以不过滤数据；在推理时，用户可以设置目标分辨率来实现尺寸条件控制。

下图展示了采用这种方案得到的512x512模型当送入不同的size时的生成图像对比，可以看到当输入低分辨率时，生成的图像比较模糊，但是当提升size时，图像质量逐渐提升，这表明模型已经学到了将条件 $c_{size}$与分辨率相关的图像特征关联起来，这可以用来修改与给定提示相对应的输出的质量。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/5-diffusion-models/3.jpg" alt="img" style="zoom:67%;" />

### Cropping Parameters——图像裁剪问题

目前文生图模型预训练往往采用固定图像尺寸（512x512或者1024x1024等），这就需要对原始图像进行预处理。这个处理流程一般是先将图像的最短边resize到目标尺寸，然后沿着图像的最长边进行裁剪（random crop或者center crop，确保图像长宽一致）。但是图像裁剪往往会导致图像出现缺失问题，如下图所示，SD1.5和SD2.1生成的猫出现头部缺失问题，就是训练过程中裁剪导致的。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/5-diffusion-models/4.jpg" alt="img" style="zoom:80%;" />

为了解决这个问题，SDXL也将训练过程中裁剪的坐标 $c_{top}$和 $c_{left}$(整数，分别指定从左上角沿高度和宽度裁剪的像素）作为额外的条件注入到UNet中，这个注入方式可以采用和图像原始尺寸一样的方式，即通过傅立叶编码并加在time embedding上。在推理时，我们只需要将这个坐标设置为(0, 0)就可以得到物体居中的图像（此时图像相当于没有裁剪）。

下图展示了采用不同的crop坐标的生成图像对比，可以看到(0, 0)坐标可以生成物体居中而无缺失的图像，采用其它的坐标就会出现有裁剪效应的图像。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/5-diffusion-models/5.jpg)

### 条件注入算法流程

**训练数据的处理流程和之前是一样的**，只是要额外保存图像的原始width和height以及图像crop时的左上定点坐标top和left，将其作为参数传入模型中。注意，**sdxl虽然输入了size参数和crop参数，但是实际还是按照固定尺寸去训练的（把小分辨率调大，把大分辨率图像调小，把宽高不一致的图像裁剪），多的仅仅是输入size和crop，让模型知道它数据处理之前大致是什么样的）**。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/5-diffusion-models/6.jpg" alt="img" style="zoom:80%;" />

裁剪得到original size和crop top-left coord参数源码：

> https://github.com/huggingface/diffusers/blob/3bce0f3da1c0c13c5589cd97946ddbf58b8a9031/examples/text_to_image/train_text_to_image_sdxl.py#L846-L873

```Python
# Preprocessing the datasets.
train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
train_crop = transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution)
train_flip = transforms.RandomHorizontalFlip(p=1.0)
train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    # image aug
    original_sizes = []
    all_images = []
    crop_top_lefts = []
    for image in images:
        original_sizes.append((image.height, image.width))
        image = train_resize(image)
        if args.random_flip and random.random() < 0.5:
            # flip
            image = train_flip(image)
        if args.center_crop:
            y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
            x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
            image = train_crop(image)
        else:
            y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
            image = crop(image, y1, x1, h, w)
        crop_top_left = (y1, x1)
        crop_top_lefts.append(crop_top_left)
        image = train_transforms(image)
        all_images.append(image)

    examples["original_sizes"] = original_sizes
    examples["crop_top_lefts"] = crop_top_lefts
    examples["pixel_values"] = all_images
    return examples
       
#transforms.CenterCrop函数
#https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomCrop
def get_params(img: Tensor, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Get parameters for ``crop`` for a random crop.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (tuple): Expected output size of the crop.

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    _, h, w = F.get_dimensions(img)
    th, tw = output_size

    if h < th or w < tw:
        raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

    if w == tw and h == th:
        return 0, 0, h, w

    i = torch.randint(0, h - th + 1, size=(1,)).item()
    j = torch.randint(0, w - tw + 1, size=(1,)).item()
    return i, j, th, tw
```

包括下面的多尺度微调，**SDXL一共添加4个额外的条件注入UNet：pooled text embedding、original size、crop top-left coord、target size**。对于后面三个条件，它们可以像timestep一样采用傅立叶编码得到特征，然后这些特征和pooled text embedding拼接在一起，最终得到维度为2816（1280+256*2*3）的特征。我们将这个特征采用两个线性层映射到和time embedding一样的维度，然后加在time embedding上即可。代码如下：

```Python
import math
from einops import rearrange
import torch

batch_size =16
# channel dimension of pooled output of text encoder (s)
pooled_dim = 1280
adm_in_channels = 2816
time_embed_dim = 1280

def fourier_embedding(inputs, outdim=256, max_period=10000):
    """
    Classical sinusoidal timestep embedding
    as commonly used in diffusion models
    : param inputs : batch of integer scalars shape [b ,]
    : param outdim : embedding dimension
    : param max_period : max freq added
    : return : batch of embeddings of shape [b, outdim ]
    """
    half = outdim // 2
    freqs = torch.exp(
        -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
    ).to(device=inputs.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding

def cat_along_channel_dim(x: torch.Tensor,) -> torch.Tensor:
    if x.ndim == 1:
        x = x[... , None]
    assert x . ndim == 2
    b, d_in = x.shape
    x = rearrange(x, "b din -> (b din)")
    # fourier fn adds additional dimension
    emb = fourier_embedding(x)
    d_f = emb.shape[-1]
    emb = rearrange(emb, "(b din) df -> b (din df)",
                        b=b, din=d_in, df=d_f)
    return emb

def concat_embeddings(
    # batch of size and crop conditioning cf. Sec. 3.2
    c_size: torch.Tensor,
    c_crop: torch.Tensor,
    # batch of target size conditioning cf. Sec. 3.3
    c_tgt_size: torch.Tensor ,
    # final output of text encoders after pooling cf. Sec . 3.1
    c_pooled_txt: torch.Tensor,
) -> torch.Tensor:
    # fourier feature for size conditioning
    c_size_emb = cat_along_channel_dim(c_size)
    # fourier feature for size conditioning
    c_crop_emb = cat_along_channel_dim(c_crop)
    # fourier feature for size conditioning
    c_tgt_size_emb = cat_along_channel_dim(c_tgt_size)
    return torch.cat([c_pooled_txt, c_size_emb, c_crop_emb, c_tgt_size_emd], dim=1)

# the concatenated output is mapped to the same
# channel dimension than the noise level conditioning
# and added to that conditioning before being fed to the unet
adm_proj = torch.nn.Sequential(
    torch.nn.Linear(adm_in_channels, time_embed_dim),
    torch.nn.SiLU(),
    torch.nn.Linear(time_embed_dim, time_embed_dim)
)

# simulating c_size and c_crop as in Sec. 3.2
c_size = torch.zeros((batch_size, 2)).long()
c_crop = torch.zeros((batch_size, 2)).long ()
# simulating c_tgt_size and pooled text encoder output as in Sec. 3.3
c_tgt_size = torch.zeros((batch_size, 2)).long()
c_pooled = torch.zeros((batch_size, pooled_dim)).long()
 
# get concatenated embedding
c_concat = concat_embeddings(c_size, c_crop, c_tgt_size, c_pooled)
# mapped to the same channel dimension with time_emb
adm_emb = adm_proj(c_concat)
```

## 多尺度微调

SDXL训练是一个多阶段的过程，首先采用基于上述的2种条件注入方法在256x256尺寸上训练60万步（batch size = 2048），然后在512x512尺寸上继续训练20万步，相当于采样了约16亿样本。最后在1024x1024尺寸上采用**多尺度方案**进行微调。

多尺度微调指**将训练数据集按照不同的长宽比（aspect ratio)进行分组（buckets），在训练过程中，随机选择一个bucket并从中采样一个batch数据进行训练**。多尺度微调可以避免过量的裁剪图像，从而减弱对模型的不利影响，并且让模型学习到了多尺度生成。但是分组的方案就需要提前对数据集进行处理，这对于大规模训练是相对麻烦的，所以SDXL选择了先采用固定尺寸预训练，然后最后再进行多尺度微调。

SDXL所设置的buckets如下表所示，虽然不同的bucket的aspect ratio不同，但是像素总大小都接近1024x1024，相邻的bucket其height或者width相差64个pixels。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/5-diffusion-models/7.jpg" alt="img" style="zoom:80%;" />

在训练过程中，每个step在不同的buckets之间切换，每个batch内部的数据都来自相同的bucket。另外，在多尺度训练中，**SDXL也将bucket size即target size作为条件加入UNet中，**表示为 $c_{ar} = (h_{tgt},w_{tgt})$。这个条件注入方式和之前图像原始尺寸条件注入一样。将target size作为条件，让模型能够显示地学习到多尺度（或aspect ratio）。

在多尺度微调阶段，SDXL依然采用前面所说的size and crop conditioning（虽然crop conditioning和多尺度微调是互补方案，在多尺度微调下，crop conditioning仅仅在bucket boundaries（64个像素）内进行调整，但是这里也依然保持这个条件注入）。经过多尺度微调后，SDXL就可以生成不同aspect ratio的图像，SDXL默认生成1024x1024的图像。

## 细化模型（refiner model)

SDXL级联了一个细化模型来进一步提升图像质量。

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/5-diffusion-models/8.jpg)

SDXL在相同的latent space上（由同一个VAE编码）训练一个分离的LDM，它专注于高质量高分辨率的数据学习，只在**较低的noise level上**进行训练（noising-denoising过程的前200个时间步上）。

推理时，首先从base模型上得到latents，之后利用扩散过程给此latent加一定的噪音，使用相同的text input在refiner模型上进一步去噪。经过这样一个重新加噪再去噪的过程，图像的局部细节会有一定的提升。

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/5-diffusion-models/9.jpg" alt="img" style="zoom:80%;" />

refiner model和base model在结构上有一定的不同，其UNet的结构如下图所示：

<img src="https://lichtung612.eos-beijing-1.cmecloud.cn/2024/5-diffusion-models/10.jpg" alt="img" style="zoom:80%;" />

- stage上：refiner model采用4个stage，第一个stage也是采用没有attention的DownBlock2D，网络的特征维度采用384，而base model是320
- block上：refiner model的attention模块中transformer block数量均设置为4。refiner model的参数量为2.3B，略小于base model
- 条件注入上：refiner model的text encoder只使用了OpenCLIP ViT-bigG，也是提取倒数第二层特征以及pooled text embed。与base model一样，refiner model也使用了size and crop conditioning，除此之外还增加了图像的艺术评分[aesthetic-score](https://link.zhihu.com/?target=https%3A//github.com/christophschuhmann/improved-aesthetic-predictor)作为条件，处理方式和之前一样。refiner model应该没有采用多尺度微调，所以没有引入target size作为条件（refiner model只是用来图生图，它可以直接适应各种尺度）。

## 模型评测

人工评价结果：

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/5-diffusion-models/12.jpg)

FID分数不是很好：

![img](https://lichtung612.eos-beijing-1.cmecloud.cn/2024/5-diffusion-models/13.jpg)

模型局限：

- 难以生成比较复杂的结构（如人手）
- 生成的图像包含多个实体时，出现属性混淆、属性渗透
- 灯光或纹理偏离现实
