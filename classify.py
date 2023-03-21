import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device using {device}")

#自定义类
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork,self).__init__()
        self.flatten = nn.Flatten()
        # nn.Sequential() 一个序列容器，用于搭建神经网络的模块被按照被传入构造器的顺序添加到nn.Sequential()容器中。
        # 利用nn.Sequential()
        # 搭建好模型架构，模型前向传播时调用forward()
        # 方法，模型接收的输入首先被传入nn.Sequential()
        # 包含的第一个网络模块中。然后，第一个网络模块的输出传入第二个网络模块作为输入，按照顺序依次计算并传播，直到nn.Sequential()
        # 里的最后一个模块输出结果。

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),  #现行层，输入28*28，输出512
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self,x):
        x = self.flatten(x) # 对x进行降维，默认从第1维开始，-1维也就是最后一维结束。注意数组是从第0维开始的，也就是第1维代表的是数组的第二个维度。
        logits = self.linear_relu_stack(x) #把降维后的数据传入上面设计好的神经网络中，将得到的结果赋值给logits
        return logits

# 实例化类，并将其转入device中
model = MyNetwork().to(device)
print(model)

#输出层，将数据经过模型迭代，打印出一个最大概率
X = torch.rand(1,28,28,device=device) #将一张28*28的图像数据输入到指定设备中
logits = model(X) #将张量数据传递给实例化的模型，这里指model
print(logits) #打印经过模型前向处理过后数据
print(logits.shape) #打印经过前向处理后的数据的形状

pred_probab = nn.Softmax(dim=1)(logits)#将类别的概率归一化
print(pred_probab)
y_pred = pred_probab.argmax(1)#返回最大值所在地索引
print(f"Predicted class: {y_pred}")#输出最大值索引