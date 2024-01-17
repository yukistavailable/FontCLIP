# FontCLIP

FontCLIP is typographic-specialized [CLIP](https://github.com/openai/CLIP).

## Approach
We propose FontCLIP, which is a CLIP model fine-tuned with font dataset.

We have explored several fint-tuning approaches and integrated them into a Python class named `ExCLIP`.




## Usage

First, [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
```

Replace `cudatoolkit=11.0` above with the appropriate CUDA version on your machine or `cpuonly` when installing on a machine without a GPU.
