import torch
import torch.nn as nn
import torch.optim as optim

def train(model, data, device, save_path=None, epochs=200, loss=None, optimizer=None):

    try:
        features, edges, labels = data
    except Exception as e:
        print(e)
        raise RuntimeError("data must be (features, edges, labels)")

    if loss is None :
        loss = nn.CrossEntropyLoss()

    if optimizer is None :
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model = model.to(device)
    model.train()

    features, edges = features.to(device), edges.to(device)
    labels = labels.to(device)

    for i in range(epochs):
        pre = model(features, edges)

        idx = select_index(labels, -1, same=False)
        pre, Y = pre[idx], labels[idx]

        # Calculate loss
        cost = loss(pre, Y)

        # Backward
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    print("Train Accuracy: %2.2f %%"%get_acc(model, data, device))
    
    if save_path is not None :
        torch.save(model.state_dict(), save_path)
        print("Model saved as", save_path)


# Test
def get_acc(model, data, device):
    # Set Cuda or Cpu
    device = torch.device(device)
    model.to(device)

    try:
        features, edges, labels = data
    except Exception as e:
        print(e)
        raise RuntimeError("data must be (features, edges, labels)")

    features, edges = features.to(device), edges.to(device)
    labels = labels.to(device)

    # Set Model to Evaluation Mode
    model.eval()

    pre = model(features, edges)

    idx = select_index(labels, -1, same=False)
    pre, Y = pre[idx], labels[idx]
    
    _, pre = torch.max(pre.data, 1)
    total = 0. + pre.size(0)
    correct = 0. + (pre == Y).sum()

    return (correct/total).item()*100


def select_index(y, value, same=True) :
    if same : 
        idx = (y == value).nonzero().view(-1)
    else :
        idx = (y != value).nonzero().view(-1)
    return idx