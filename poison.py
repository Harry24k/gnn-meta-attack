# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy import sparse

from metaattack import *
from loader import *

from scipy import sparse


def run(hidden, lambda_, train_iters, perturb_rate, save_path,
        data_name, random_state, use_cuda):

    device = "cpu"
    if use_cuda:
        device = "cuda"

    # Load Train data
    train_data = load_data(data_name=data_name, train=True, random_state=random_state)
    features, edges, labels = train_data

    dim_features = features.shape[1]
    nnodes = edges.shape[0]
    num_labels = max(labels).item() + 1

    perturbations = int(perturb_rate * (edges.sum()//2))

    attack = Metattack(nfeat=dim_features, hidden_sizes=[hidden], nclass=num_labels, 
                       nnodes=nnodes, lambda_=lambda_, device=device)

    modified_adj = attack(features, edges, labels, perturbations, train_iters, ll_constraint=False)

    # Save as the same format of 'cora_ml'

    
    _, _, full_labels = load_data(data_name=data_name, train=True, random_state=random_state, test_size=0)
    
    sA = sparse.csr_matrix(modified_adj.detach().cpu().numpy())
    sB = sparse.csr_matrix(features.detach().cpu().numpy())
    sC = full_labels.numpy()

    loader = {}
    np.savez('data/'+save_path,
             adj_data=sA.data,
             adj_indices=sA.indices,
             adj_indptr=sA.indptr,
             adj_shape=sA.shape,
             attr_data=sB.data,
             attr_indices=sB.indices,
             attr_indptr=sB.indptr,
             attr_shape=sB.shape,
             labels=sC)


if __name__ == "__main__":
    parser = ArgumentParser(description="Script for gnn-meta-attack",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hidden', default=6, type=int, help='Dimension of hidden layer')
    parser.add_argument('--lambda_', default=0.5, type=float, help='labmda of gnn-meta-attack')
    parser.add_argument('--train-iters', default=15, type=int, help='Dataset path')
    parser.add_argument('--perturb-rate', default=0.05, type=float, help='Random-state for train-test-split')
    parser.add_argument('--save-path', help='Poisoned data save path.')
    parser.add_argument('--data-name', default='cora_ml', type=str, help='Dataset path')
    parser.add_argument('--random-state', default=42, type=int, help='Random-state for train-test-split')
    parser.add_argument('--use-cuda', default=True, type=str, help='True for cuda')

    args = parser.parse_args()

    run(args.hidden,
        args.lambda_,
        args.train_iters,
        args.perturb_rate,
        args.save_path,
        args.data_name,
        args.random_state,
        args.use_cuda)