"""Reference ExoNet-V2: dual-branch 1D-CNN (global 2001, local 201) + 3 metadata features."""
import torch, torch.nn as nn
def branch(n):
    L, c = [], 1
    for o in (16, 32, 64)[:n]:
        L += [nn.Conv1d(c, o, 5, padding=2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2)]; c = o
    return nn.Sequential(*L, nn.AdaptiveAvgPool1d(1), nn.Flatten())
class DualBranchExoCNN(nn.Module):
    def __init__(s):
        super().__init__(); s.g, s.l = branch(3), branch(2)
        s.head = nn.Sequential(nn.Linear(64 + 32 + 3, 64), nn.ReLU(), nn.Dropout(.4), nn.Linear(64, 1))
    def forward(s, g, l, m): return torch.sigmoid(s.head(torch.cat([s.g(g), s.l(l), m], 1))).squeeze(1)
if __name__ == "__main__":
    print(DualBranchExoCNN()(torch.randn(2, 1, 2001), torch.randn(2, 1, 201), torch.randn(2, 3)).shape)
