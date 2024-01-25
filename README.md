# FontCLIP: A Semantic Typography Visual-Language Model for Multilingual Font Applications

<br>
<div align="center">
    <img src="media/teaser.png" width="100%">
    <!-- <iframe src="media/teaser.pdf" width="100%" frameborder="0" style="border:none;"></iframe> -->
</div>
<br><br>


## Abstract
Acquiring the desired font for various design tasks can be challenging and requires professional typographic knowledge. While
previous font retrieval or generation works have alleviated some of these difficulties, they often lack support for multiple
languages and semantic attributes beyond the training data domains. To solve this problem, we present FontCLIP – a model that
connects the semantic understanding of a large vision-language model with typographical knowledge. We integrate typographyspecific knowledge into the comprehensive vision-language knowledge of a pretrained CLIP model through a novel finetuning
approach. We propose to use a compound descriptive prompt that encapsulates adaptively sampled attributes from a font attribute
dataset focusing on Roman alphabet characters. FontCLIP’s semantic typographic latent space demonstrates two unprecedented
generalization abilities. First, FontCLIP generalizes to different languages including Chinese, Japanese, and Korean (CJK),
capturing the typographical features of fonts across different languages, even though it was only finetuned using fonts of Roman
characters. Second, FontCLIP can recognize the semantic attributes that are not presented in the training data. FontCLIP’s
dual-modality and generalization abilities enable multilingual and cross-lingual font retrieval and letter shape optimization,
reducing the burden of obtaining desired fonts.

## Approach
We propose FontCLIP, which is a [CLIP](https://github.com/openai/CLIP) model fine-tuned with [a font dataset](https://www.dgp.toronto.edu/~donovan/font/).

We have explored several fint-tuning approaches and integrated them into a Python class named `ExCLIP`.




## Usage

First, [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
```

Replace `cudatoolkit=11.0` above with the appropriate CUDA version on your machine or `cpuonly` when installing on a machine without a GPU.

## ExCLIP