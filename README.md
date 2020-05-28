# GNN-Meta-Attack

This repository is a pytorch implementation of [Adversarial Attacks on Graph Neural Networks via Meta Learning](https://arxiv.org/abs/1902.08412). 

<img src="https://github.com/facebookresearch/FixRes/blob/master/image/image2.png" height="190">

It is modified from following githubs :
> https://github.com/danielzuegner/gnn-meta-attack (Official, Tensorflow)
> https://github.com/ChandlerBang/pytorch-gnn-meta-attack (Pytorch)
> https://github.com/Kaushalya/gnn-meta-attack-pytorch (Pytorch)

## Requirements

* python 3.6
* torch 1.2
* torchvision 0.5
* numpy
* scipy
* matpotlib
* seaborn

## Training

See help (`-h` flag) for detailed parameter list of each script before executing the code.
 
To train the model(s) in the paper, run this command:

```bash
# ResNet50
python main_resnet50_scratch.py --batch 64 --num-tasks 8 --learning-rate 2e-2

```

> 📋Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

See help (`-h` flag) for detailed parameter list of each script before executing the code.

`main_evaluate_imnet.py` evaluates the network on standard benchmarks.

`main_evaluate_softmax.py` evaluates the network on ImageNet-val with already extracted softmax output. (Much faster to execute)

```bash
# FixResNeXt-101 32x48d
python main_evaluate_imnet.py --input-size 320 --architecture 'IGAM_Resnext101_32x48d' --weight-path 'ResNext101_32x48d.pth'
```

The following code give results that corresponds to table 2 in the paper :
```
# FixResNet-50
python main_evaluate_imnet.py --input-size 384 --architecture 'ResNet50' --weight-path 'ResNet50.pth'
```

> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

### Using transforms_v2 for fine-tuning
To reproduce our best results we must use the data-augmentation of transforms_v2 and use almost the same parameters as for the classic data augmentation, the only changes are the learning rate which must be 1e-4 and the number of epochs which must be 11. For FixResNet-50 fine-tune you have to use 31 epochs and a learning rate of 1e-3 and for FixResNet-50 CutMix you have to use 11 epochs and a learning rate of 1e-3.
Here is how to use transforms_v2 :

```python
from torchvision import datasets
from .transforms_v2 import get_transforms

transform = get_transforms(input_size=Train_size,test_size=Test_size, kind='full', crop=True, need=('train', 'val'), backbone=None)
train_set = datasets.ImageFolder(train_path,transform=transform['val_train'])
test_set = datasets.ImageFolder(val_path,transform=transform['val_test'])
```


## Pre-trained Networks

We provide pre-trained networks with torch.hub :

```python
import torch
model = torch.hub.load('Harry24k/releasing-research-code', 'resnet18', pretrained=True, force_reload=True)
```

## Results

We report in the table validation resolution, Top-1 and Top-5 accuracy on ImageNet validation set:

Our model achieves the following performance on :

### ImageNet

|  Models  | Resolution | #Parameters | Top-1 / Top-5 |                                        Weights                                         |
|:---:|:-:|:------------:|:------:|:---------------------------------------------------------------------------------------:|
|  ResNet-50 Baseline| 224 |     25.6M     |  77.0 /  93.4 | [FixResNet50_no_adaptation.pth](https://dl.fbaipublicfiles.com/FixRes_data/FixRes_Pretrained_Models/ResNet_no_adaptation.pth)  |
|  FixResNet-50 | 384 |    25.6M     |  79.0 / 94.6 |  [FixResNet50.pth](https://dl.fbaipublicfiles.com/FixRes_data/FixRes_Pretrained_Models/ResNetFinetune.pth)  |
|  FixResNet-50 (*)| 384 |    25.6M     |  79.1 / 94.6 |  [FixResNet50_v2.pth](https://dl.fbaipublicfiles.com/FixRes_data/FixRes_Pretrained_Models/ResNet50_v2.pth)  |
| FixResNet-50 CutMix | 320 |     25.6M     |  79.7 /  94.9 | [FixResNet50CutMix.pth](https://dl.fbaipublicfiles.com/FixRes_data/FixRes_Pretrained_Models/ResNetCutMix.pth)  |

(+)  We use Horizontal flip, shifted Center Crop and color jittering for fine-tuning (described in [transforms_v2.py](transforms_v2.py))

(+) We report different results with our FixEfficientNet (see [FixEfficientNet](README_FixEfficientNet.md) for more details)

To load a network, use the following PyTorch code: 

```python
import torch
from .resnext_wsl import resnext101_32x48d_wsl

model=resnext101_32x48d_wsl(progress=True) # example with the ResNeXt-101 32x48d 

pretrained_dict=torch.load('ResNeXt101_32x48d.pth',map_location='cpu')['model']

model_dict = model.state_dict()
for k in model_dict.keys():
    if(('module.'+k) in pretrained_dict.keys()):
        model_dict[k]=pretrained_dict.get(('module.'+k))
model.load_state_dict(model_dict)
```
The network takes images in any resolution. 
A normalization pre-processing step is used, with mean `[0.485, 0.456, 0.406]`. 
and standard deviation `[0.229, 0.224, 0.225]` for ResNet-50 and ResNeXt-101 32x48d,
use  mean `[0.5, 0.5, 0.5]` and standard deviation `[0.5, 0.5, 0.5]` with PNASNet.
You can find the code in transforms.py.

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## References to other models

Model definition scripts are based on https://github.com/pytorch/vision/ and https://github.com/Cadene/pretrained-models.pytorch.

The Training from scratch implementation is based on https://github.com/facebookresearch/multigrain.

Our FixResNet-50 CutMix is fine-tune from the weights of the GitHub page : https://github.com/clovaai/CutMix-PyTorch.
The corresponding paper is 
```
@inproceedings{2019arXivCutMix,
       author = {Sangdoo Yun and Dongyoon Han and Seong Joon Oh and Sanghyuk Chun and Junsuk Choe and Youngjoon Yoo,
       title = "{CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features}",
       journal = {arXiv e-prints},
       year = "2019"}
```