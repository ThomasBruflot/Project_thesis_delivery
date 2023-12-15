import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import datasets


from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import DiehlAndCookNodes, Input
from bindsnet.network.topology import Connection, Conv2dConnection, MaxPool2dConnection
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.analysis.plotting import (
    plot_conv2d_weights,
    plot_input,
    plot_spikes,
    plot_voltages,
)
intensity = 128.0
time = 50
dt = 1.0


# As torch.Tensor
test_dataset = datasets.MNIST(
    root="/Users/thomasbruflot/Documents/5.klasse/1.Semester/Project_thesis/data/MNIST",
    download=True,
    transform=transforms.ToTensor()
)

x, _ = test_dataset[7777] # x is now a torch.Tensor
plt.imshow(x.numpy()[0], cmap='gray')
print(test_dataset[5003])
#ax = plt.subplot(1)
#plt.imshow(test_dataset[5003])
x, _ = test_dataset[5003] # x is now a torch.Tensor
plt.imshow(x.numpy()[0], cmap='gray')
plt.show()