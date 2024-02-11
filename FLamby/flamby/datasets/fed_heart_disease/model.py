import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, input_dim=13, output_dim=1, MCDO=False, BN=False):
        super(Baseline, self).__init__()
        self.BN = BN
        self.MCDO = MCDO
        hidden_dim = 32
        if self.BN:
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.bn = torch.nn.BatchNorm1d(hidden_dim)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        if self.MCDO:
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.bn = torch.nn.BatchNorm1d(hidden_dim)
            self.relu = torch.nn.ReLU()
            self.dropout = nn.Dropout(p=0.2)
            self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        else:
            self.linear = torch.nn.Linear(input_dim, output_dim)
        i = 0
        flag = 0
        self.bns = []
        def dfs(layer):
            nonlocal i,flag
            if flag == 1:
                return
            name = str(layer)
            if name.startswith("BatchNorm"):
                self.bns.append(layer)
                i += 1
                if i == 49:
                    flag = 1
            elif "BatchNorm" in name:
                for l in layer.children():
                    dfs(l)
        dfs(self)

    def forward(self, x):
        if self.MCDO:
            x = self.relu(self.bn(self.fc1(x)))
            return torch.sigmoid(self.fc2(self.dropout(x)))
        if self.BN:
            x = self.fc1(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.fc2(x)
            return torch.sigmoid(x)
        return torch.sigmoid(self.linear(x))

    def getallfea(self, x):
        if self.BN or self.MCDO:
            return [self.fc1(x).clone().detach()]
        else:
            return [self.linear(x).clone().detach()]

    def get_sel_fea(self, x, plan):
        if self.BN or self.MCDO:
            return self.fc1(x)
        else:
            return self.linear(x)