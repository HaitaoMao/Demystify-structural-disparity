import torch.nn as nn
import torch

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)
    
def sgc_precompute(features, adj, degree):
    for _ in range(degree):
        features = torch.spmm(adj, features)
    return features