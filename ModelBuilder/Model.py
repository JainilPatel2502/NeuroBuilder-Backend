import torch.nn as nn
class Model(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.network = nn.Sequential(*layers)
    def forward(self,x):
        out = self.network(x)
        return out