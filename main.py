# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from train import *
from loader import *
from models import GCN

def run(is_train, hidden, save_path, data_name, random_state, use_cuda):
    
    device = "cpu"
    if use_cuda:
        device = "cuda"
    
    # Load Train data
    train_data = load_data(data_name=data_name, train=True, random_state=random_state)
    dim_features = train_data[0].shape[1]
    num_labels = max(train_data[2]).item() + 1
    
    model = GCN(dim_features, hidden, num_labels)
    
    # Train Phase
    if is_train :
        print("---Training Start---")
        train(model, train_data, save_path, device)
        print("---Training Done--- \n")
    
    # Load model
    model.load_state_dict(torch.load(save_path))
    
    # Test Phase   
    print("---Test Start---")
    test_data = load_data(data_name=data_name, train=False, random_state=42)
    print("Test Accuracy: %2.2f %%"%get_acc(model, test_data, device))
    print("---Test Done--- \n")
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Script for gnn-meta-attack",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', default=True, type=bool, help='If train=True, train a new model and save it. Otherwise, only test the model.')
    parser.add_argument('--hidden', default=16, type=int, help='Dimension of hidden layer')
    parser.add_argument('--save-path', default='base.pth', type=str, help='GNN weights')
    parser.add_argument('--data-name', default='cora_ml', type=str, help='Dataset path')
    parser.add_argument('--random-state', default=42, type=int, help='Random-state for train-test-split')
    parser.add_argument('--use-cuda', default=True, type=str, help='True for cuda')
    
    args = parser.parse_args()

    run(args.train,
        args.hidden,
        args.save_path,
        args.data_name,
        args.random_state,
        args.use_cuda)
