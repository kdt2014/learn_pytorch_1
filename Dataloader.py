import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

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
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[4]  #.squeeze()  # h删掉第一行数据中的一维维度
label = train_labels[0]  # 获取标签数据中的信息
plt.imshow(img.squeeze(), cmap="gray")  # 以灰度图像显示
plt.show()
print(f"Label: {label}")  # 打印处对应的标签值
# print(f"train_features: {train_features}")

# 这是因为这个image是三维的,.squeeze()函数可以把三维数组变为二维。因为它会把为1的维度给去掉。