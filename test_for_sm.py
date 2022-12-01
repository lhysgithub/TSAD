import torch

tensor_0 = torch.arange(3, 12).view(3, 3)
print(tensor_0)

index = torch.tensor([[0, 2],
                      [1, 2]])
tensor_1 = tensor_0.gather(1, index)
print(tensor_1)