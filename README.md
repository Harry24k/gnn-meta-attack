# GNN-Meta-Attack

This repository is a pytorch implementation of [Adversarial Attacks on Graph Neural Networks via Meta Learning](https://arxiv.org/abs/1902.08412). 

<img src="https://github.com/facebookresearch/FixRes/blob/master/image/image2.png" height="190">

It is modified from following githubs :

> https://github.com/ChandlerBang/pytorch-gnn-meta-attack (Main Reference, Pytorch)

> https://github.com/danielzuegner/gnn-meta-attack (Official, Tensorflow)

> https://github.com/Kaushalya/gnn-meta-attack-pytorch (Pytorch)

## Requirements

* python 3.6
* torch 1.2
* torchvision 0.5
* [higher](https://github.com/facebookresearch/higher)
* numpy
* scipy
* matpotlib
* seaborn

## Usage

See help (`-h` flag) for detailed parameter list of each script before executing the code.
 
### Training

To train the model(s) in the paper, run this command:

```bash
# cora_ml
python main.py --train True --hidden 16 --data-name cora_ml --save-path sample.pth

```

### Evaluation

To evaluate the model(s) saved locally, run this command:

```bash
# cora_ml
python main.py --train False --hidden 6 --data-name cora_ml --save-path sample.pth

```

### Generate Poisoned Data with Meta Attack

Here is how to generate poisoned data :

```bash
# cora_ml
python poison.py --hidden 6 --lambda_ 0.5 --train-iters 15 --perturb-rate 0.05 --save-path sample.pth --data-name cora_ml

```

(+) Approximated Meta Attack is not implemented.


## Results

The accuracy on each dataset :

(+) Unless it's mentioned, all values are set to default.
(+) For each method, lambdas are 0(Self), 0.5(Both) and 1(Train).

### cora_ml

|  Description | Perturb Rate | Accuracy(Dropped) | Reported | Data Name |
|:---:|:------------:|:------:|:------:|:------:| 
| Clean |     0.00     |  85.80%(-) |  83.40% | cora_ml.npz |
| Self  |     0.05     |  80.43%(5.37%p) |  75.50% | cora_ml_self_5.npz |
| Both  |     0.05     |  81.42%(4.38%p) |  85.80% | cora_ml_both_5.npz |
| Train |     0.05     |  82.52%(3.28%p) |  78.00% | cora_ml_train_5.npz |
| Self  |     0.20     |  58.01%(27.79%p) |  - | cora_ml_self_20.npz |
| Both  |     0.20     |  67.73%(18.07%p) |  - | cora_ml_both_20.npz |
| Train |     0.20     |  80.03%(5.77%p) |  - | cora_ml_train_20.npz |

(+) Due to GPU Limitation, I used train-iters=15 when generating poisoned data. Thus, the accuracies of meta-attack are higher than reported ones.

## Visualization

Here is a visualization of changes of edges with [Lightning](http://lightning-viz.org/lightning-python/index.html).

Please see [Visulaize.ipynb](https://github.com/Harry24k/gnn-meta-attack/Visualize.ipynb) for reproducing these images.

cora_ml | cora_ml_both_5
:---: | :---:
<img src="https://github.com/Harry24k/gnn-meta-attack/blob/master/images/cora_ml.png" width="300" height="300"> | <img src="https://github.com/Harry24k/gnn-meta-attack/blob/master/images/cora_ml_both_5.png" width="300" height="300">

(+) Statistics

* Deleted Edges: 0.0
* Created Edges: 212.0
* Number of Nodes
    * Effected Train Nodes: 34
    * Effected Test Nodes: 135