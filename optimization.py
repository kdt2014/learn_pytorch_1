import torch
from torch import nn
from torch.utils.data improt Dataloader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root = "data"
    train = True
    download = True
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data"
    train = True
    download = True
    transform = ToTensor()
)

train_dataloader = Dataloader(training_data, batch_size=64)
test_dataloader = Dataloader(test_data, batch_size=64)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = MyNetwork()