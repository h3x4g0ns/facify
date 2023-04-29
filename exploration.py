import torch
import torchvision

dataset = torchvision.datasets.LFWPeople()
x, y = dataset[0]
print(x.shape, y.shape)