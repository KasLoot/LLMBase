import torch



a = torch.tensor([[1, 2, 3],
                 [4, 5, 6]])
print(a.is_contiguous())
print(a)

b = a.view(3, 2)

a = a.transpose(1, 0)
b = a.transpose(1, 0)
print(a.is_contiguous())
print(a)
print(b.is_contiguous())
print(b)