import torch
import torch.nn as nn
import torch.functional as F


class Expand_Sequence(nn.Module):
    def __init__(self):
        super(Expand_Sequence, self).__init__()
        self.l1 = nn.Linear(16, 64)

    def forward(self, x: torch.Tensor):
        return self.l1(x.transpose(1, 2)).transpose(1, 2)
    

a = torch.randn((1, 16, 128))


model = Expand_Sequence()
print(model)
y = model(a)
print(a.size())
print(y.size())