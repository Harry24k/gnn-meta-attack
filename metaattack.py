import torch
from torch import nn
from torch import optim
from models import GCN
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
from tqdm import tqdm
import utils
from train import *
import higher


class BaseMeta(Module):

    def __init__(self, nfeat, hidden_sizes, nclass, nnodes, lambda_, device):
        super(BaseMeta, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.nfeat = nfeat
        self.nclass = nclass
        
        self.lambda_ = lambda_        
        self.device = device
        
        self.gcn = GCN(nfeat=nfeat,
                       nhid=hidden_sizes[0],
                       nclass=nclass).to(self.device)

        self.nnodes = nnodes
        self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes)).to(self.device)
        self.adj_changes.data.fill_(0)

    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.
        Returns
        -------
        torch.Tensor shape [N, N], float with ones everywhere except the entries of potential singleton nodes,
        where the returned tensor has value 0.
        """

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()

        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        flat_mask = 1 - logical_and_symmetric
        return flat_mask

    def train_surrogate(self, features, edges, labels, train_iters=200):
        print('=== training surrogate model to predict unlabled data for self-training')
        
        model = self.gcn
        train(model, (features, edges, labels), self.device,
              save_path=None, epochs=train_iters)
        
        pre = model(features, edges)
        _, pre = torch.max(pre.data, 1)
        
        self.labels_self_training = pre.detach()
    
        model.initialize()
    
    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        """
        Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
        log likelihood constraint to be violated.
        """

        t_d_min = torch.tensor(2.0).to(self.device)
        t_possible_edges = np.array(np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T
        allowed_mask, current_ratio = utils.likelihood_ratio_filter(t_possible_edges,
                                                                    modified_adj,
                                                                    ori_adj, t_d_min,
                                                                    ll_cutoff)

        return allowed_mask, current_ratio


class Metattack(BaseMeta):

    def __init__(self, nfeat, hidden_sizes, nclass, 
                 nnodes, lambda_, device):

        super(Metattack, self).__init__(nfeat, hidden_sizes, nclass,
                                        nnodes, lambda_, device)

    def get_meta_grad(self, features, edges, labels, train_iters):

        model = self.gcn
        loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
            
            for i in range(train_iters):
                pre = fmodel(features, edges)
                idx = select_index(labels, -1, same=False)
                pre, Y = pre[idx], labels[idx]

                cost = loss(pre, Y)

                diffopt.step(cost)

            pre = fmodel(features, edges)
            idx = select_index(labels, -1, same=False)
            sudo_idx = select_index(labels, -1, same=True)
            
            cost = 0
            
            if self.lambda_ > 0 :
                cost =+ self.lambda_ * loss(pre[idx], labels[idx])
            
            if (1-self.lambda_) > 0 :
                cost =+ (1-self.lambda_) * loss(pre[sudo_idx], self.labels_self_training[sudo_idx])
                
            return torch.autograd.grad(cost, self.adj_changes, retain_graph=False)[0]

    def forward(self, features, ori_adj, labels, perturbations, train_iters, ll_constraint=True, ll_cutoff=0.004):
        
        features, ori_adj, labels = features.to(self.device), ori_adj.to(self.device), labels.to(self.device)
        
        if (1-self.lambda_) > 0 :
            self.train_surrogate(features, ori_adj, labels)

        for i in tqdm(range(perturbations), desc="Perturbing graph"):
            adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
            ind = np.diag_indices(self.adj_changes.shape[0])
            adj_changes_symm = torch.clamp(adj_changes_square + torch.transpose(adj_changes_square, 1, 0), -1, 1)
            modified_adj = adj_changes_symm + ori_adj
            
            adj_grad = self.get_meta_grad(features, modified_adj, labels, train_iters)

            adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
            adj_meta_grad -= adj_meta_grad.min()
            adj_meta_grad -= torch.diag(torch.diag(adj_meta_grad, 0))
            singleton_mask = self.filter_potential_singletons(modified_adj)
            adj_meta_grad = adj_meta_grad *  singleton_mask

            if ll_constraint:
                allowed_mask, self.ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, ll_cutoff)
                allowed_mask = allowed_mask.to(self.device)
                adj_meta_grad = adj_meta_grad * allowed_mask

            # Get argmax of the meta gradients.
            adj_meta_argmax = torch.argmax(adj_meta_grad)
            
            row_idx = adj_meta_argmax // adj_meta_grad.shape[0]
            col_idx  = adj_meta_argmax %  adj_meta_grad.shape[0]
            
            self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)

        return self.adj_changes + ori_adj