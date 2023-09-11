import torch

t = torch.randn((5, 5))

print(t)
max = torch.argmax(t, 1)

print(max)