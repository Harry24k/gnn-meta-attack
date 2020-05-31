import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

import higher

from models import GCN
from train import *

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
        allowed_mask, current_ratio = likelihood_ratio_filter(t_possible_edges,
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
    
def likelihood_ratio_filter(node_pairs, modified_adjacency, original_adjacency, d_min, threshold=0.004):
    """
    Filter the input node pairs based on the likelihood ratio test proposed by Zügner et al. 2018, see
    https://dl.acm.org/citation.cfm?id=3220078. In essence, for each node pair return 1 if adding/removing the edge
    between the two nodes does not violate the unnoticeability constraint, and return 0 otherwise. Assumes unweighted
    and undirected graphs.
    """

    N = int(modified_adjacency.shape[0])

    original_degree_sequence = original_adjacency.sum(0)
    current_degree_sequence = modified_adjacency.sum(0)

    # Concatenate the degree sequences
    concat_degree_sequence = torch.cat((current_degree_sequence, original_degree_sequence))

    # Compute the log likelihood values of the original, modified, and combined degree sequences.
    ll_orig, alpha_orig, n_orig, sum_log_degrees_original = degree_sequence_log_likelihood(original_degree_sequence, d_min)
    ll_current, alpha_current, n_current, sum_log_degrees_current = degree_sequence_log_likelihood(
        current_degree_sequence, d_min)

    ll_comb, alpha_comb, n_comb, sum_log_degrees_combined = degree_sequence_log_likelihood(concat_degree_sequence, d_min)

    # Compute the log likelihood ratio
    current_ratio = -2 * ll_comb + 2 * (ll_orig + ll_current)

    # Compute new log likelihood values that would arise if we add/remove the edges corresponding to each node pair.

    new_lls, new_alphas, new_ns, new_sum_log_degrees = updated_log_likelihood_for_edge_changes(node_pairs,
                                                                                               modified_adjacency, d_min)

    # Combination of the original degree distribution with the distributions corresponding to each node pair.
    n_combined = n_orig + new_ns
    new_sum_log_degrees_combined = sum_log_degrees_original + new_sum_log_degrees
    alpha_combined = compute_alpha(n_combined, new_sum_log_degrees_combined, d_min)

    new_ll_combined = compute_log_likelihood(n_combined, alpha_combined, new_sum_log_degrees_combined, d_min)
    new_ratios = -2 * new_ll_combined + 2 * (new_lls + ll_orig)

    # Allowed edges are only those for which the resulting likelihood ratio measure is < than the threshold
    allowed_edges = new_ratios < threshold
    try:
        filtered_edges = node_pairs[allowed_edges.cpu().numpy().astype(np.bool)]
    except:
        filtered_edges = node_pairs[allowed_edges.numpy().astype(np.bool)]

    allowed_mask = torch.zeros(modified_adjacency.shape)
    allowed_mask[filtered_edges.T] = 1
    allowed_mask += allowed_mask.t()
    return allowed_mask, current_ratio


def degree_sequence_log_likelihood(degree_sequence, d_min):
    """
    Compute the (maximum) log likelihood of the Powerlaw distribution fit on a degree distribution.

    """
    # Determine which degrees are to be considered, i.e. >= d_min.

    D_G = degree_sequence[(degree_sequence >= d_min.item())]
    try:
        sum_log_degrees = torch.log(D_G).sum()
    except:
        sum_log_degrees = np.log(D_G).sum()
    n = len(D_G)

    alpha = compute_alpha(n, sum_log_degrees, d_min)
    ll = compute_log_likelihood(n, alpha, sum_log_degrees, d_min)
    return ll, alpha, n, sum_log_degrees


def updated_log_likelihood_for_edge_changes(node_pairs, adjacency_matrix, d_min):

    # For each node pair find out whether there is an edge or not in the input adjacency matrix.

    edge_entries_before = adjacency_matrix[node_pairs.T]

    degree_sequence = adjacency_matrix.sum(1)

    D_G = degree_sequence[degree_sequence >= d_min.item()]
    sum_log_degrees = torch.log(D_G).sum()
    n = len(D_G)

    deltas = -2 * edge_entries_before + 1
    d_edges_before = degree_sequence[node_pairs]

    d_edges_after = degree_sequence[node_pairs] + deltas[:, None]

    # Sum the log of the degrees after the potential changes which are >= d_min
    sum_log_degrees_after, new_n = update_sum_log_degrees(sum_log_degrees, n, d_edges_before, d_edges_after, d_min)

    # Updated estimates of the Powerlaw exponents
    new_alpha = compute_alpha(new_n, sum_log_degrees_after, d_min)
    # Updated log likelihood values for the Powerlaw distributions
    new_ll = compute_log_likelihood(new_n, new_alpha, sum_log_degrees_after, d_min)

    return new_ll, new_alpha, new_n, sum_log_degrees_after


def update_sum_log_degrees(sum_log_degrees_before, n_old, d_old, d_new, d_min):
    # Find out whether the degrees before and after the change are above the threshold d_min.
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min
    d_old_in_range = d_old * old_in_range.float()
    d_new_in_range = d_new * new_in_range.float()

    # Update the sum by subtracting the old values and then adding the updated logs of the degrees.
    sum_log_degrees_after = sum_log_degrees_before - (torch.log(torch.clamp(d_old_in_range, min=1))).sum(1) \
                                 + (torch.log(torch.clamp(d_new_in_range, min=1))).sum(1)

    # Update the number of degrees >= d_min
    new_n = n_old - (old_in_range!=0).sum(1) + (new_in_range!=0).sum(1)
    new_n = new_n.float()
    return sum_log_degrees_after, new_n


def compute_alpha(n, sum_log_degrees, d_min):

    try:
        alpha =  1 + n / (sum_log_degrees - n * torch.log(d_min - 0.5))
    except:
        alpha =  1 + n / (sum_log_degrees - n * np.log(d_min - 0.5))
    return alpha


def compute_log_likelihood(n, alpha, sum_log_degrees, d_min):
    # Log likelihood under alpha
    try:
        ll = n * torch.log(alpha) + n * alpha * torch.log(d_min) + (alpha + 1) * sum_log_degrees
    except:
        ll = n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * sum_log_degrees

    return ll