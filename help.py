# Importing the PyTorch library
import torch
import os
import pickle
from plotting import *
from models import FastText


# A constant tensor of size n
a = torch.randn(3,2)
a = torch.sigmoid(a)
print(a)
b = torch.sum(a, dim=1)
print(b)

model = FastText(1000, 128, 2)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total trainable params: ', pytorch_total_params)

embeddings = torch.load(os.path.join('saved_models', 'w5_word_embeddings_h256_10000.pth')).embeddings.weight.data
model2 = FastText(1000, 128, 2, word_embeddings=embeddings)
pytorch_total_params = sum(p.numel() for p in model2.parameters() if p.requires_grad)
print('Total trainable params: ', pytorch_total_params)
print('Embedding dim: ', embeddings.size())

m = torch.nn.ReLU()
input = torch.randn(2)
print(input)
output = m(input)
print(output)