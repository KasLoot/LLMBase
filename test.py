import torch
from torch.nn.utils.rnn import pad_sequence
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
# b = torch.ones(22, 300)
# c = torch.ones(15, 300)
print(a)
c = pad_sequence(a)
print(c)
print(c.size())