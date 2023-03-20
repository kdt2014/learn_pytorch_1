import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8)) # 创建一张图8*8英寸
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()  # torch.randint()得到len范围内的一个随机数值，size表示是一维的，并且只包含1个数字，数据是tensor格式，item()得到tensor内的数值
    img, label = training_data[sample_idx] #每一个训练和测试数据都包含两部分：图像和标签
    figure.add_subplot(rows, cols, i) #创建子图1，如:（a,b,c）该子图高为1/a，宽为1/b，区域位置为c。区域从第一行开始编号，从左到右开始编号。最上面为第一行。
    plt.title(labels_map[label]) #labels_map是一个字典，通过下标的方式对应的标签内容，标签是前面的数字，对应的名字是后面的英文部分
    plt.axis("off") #关闭坐标轴
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()