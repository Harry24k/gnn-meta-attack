import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split

def load_npz(file_name, is_sparse=True):
    if not file_name.endswith('.npz'):
        file_name += '.npz'

    with np.load(file_name) as loader:
        loader = dict(loader)
        if is_sparse:

            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                        loader['adj_indptr']), shape=loader['adj_shape'])

            if 'attr_data' in loader:
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                             loader['attr_indptr']), shape=loader['attr_shape'])
            else:
                features = None

            labels = loader.get('labels')

        else:
            adj = loader['adj_data']

            if 'attr_data' in loader:
                features = loader['attr_data']
            else:
                features = None

            labels = loader.get('labels')

    return adj, features, labels


def get_adj(dataset, is_sparse, require_lcc=True):
    print('reading %s...' % dataset)
    _A_obs, _X_obs, _z_obs = load_npz(r'data/%s.npz' % dataset, is_sparse=is_sparse)
    _A_obs = _A_obs + _A_obs.T
    _A_obs = _A_obs.tolil()
    _A_obs[_A_obs > 1] = 1

    if _X_obs is None:
        _X_obs = np.eye(_A_obs.shape[0])

    # require_lcc= False
    if require_lcc:
        lcc = largest_connected_components(_A_obs)

        _A_obs = _A_obs[lcc][:,lcc]
        _X_obs = _X_obs[lcc]
        _z_obs = _z_obs[lcc]

        assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

    # whether to set diag=0?
    _A_obs.setdiag(0)
    _A_obs = _A_obs.astype("float32").tocsr()
    _A_obs.eliminate_zeros()

    assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
    assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"

    return _A_obs, _X_obs, _z_obs


def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.
    Parameters
    """
    _, component_indices = sp.csgraph.connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(data_name, train=True, test_size=0.9, random_state=1,
              from_sparse=True, to_sparse=False):

    print('Loading {} dataset...'.format(data_name))
    adj, features, labels = get_adj(data_name, from_sparse)
    features = sp.csr_matrix(features, dtype=np.float32)

    labels = torch.LongTensor(labels)
    if to_sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = torch.FloatTensor(np.array(features.todense()))
        adj = torch.FloatTensor(adj.todense())

    train_idx, test_idx, _, _ = train_test_split(list(range(len(labels))), labels.numpy(), 
                                                 test_size=test_size, random_state=random_state)

    if train :
        labels[test_idx] = -1
    else :
        labels[train_idx] = -1

    return features, adj, labels