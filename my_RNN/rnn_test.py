# coding=utf-8

from torch import nn
import torch

if __name__ == "__main__":
    test_tensor = torch.randn(2, 3, 4)
    print(test_tensor)
    print(test_tensor.view(6, -1))
