import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import time

# batchsize = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

# 将训练和测试数据全部放入GPU显存，以后的每次训练和测试直接从GPU中加载数据，省略了大量倒腾数据的时间
train_images, train_labels = training_data.data.to('cuda'), training_data.targets.to('cuda')
test_images, test_labels = test_data.data.to('cuda'), test_data.targets.to('cuda')

# Reshape the images to 1D tensors and normalize
train_images = train_images.view(train_images.size(0), -1) / 255.
test_images = test_images.view(test_images.size(0), -1) / 255.

# Combine images and labels into TensorDatasets
train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

# Create data loaders
batch_size = len(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
batch_size = len(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = MyNetwork().to(device) #将实例化后的模型传递给设置好的设备并赋值给model
print(model)


learning_rate = 1e-3 #学习率
loss_fn = nn.CrossEntropyLoss()  # Initialize the loss function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
epochs = 10000


start_time = time.time()
# Train model
for epoch in range(epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()  # 梯度清零
        pred = model(inputs)  # 将图像数据通过训练模型
        loss = loss_fn(pred, targets)  # 计算预测值和真实值的误差大小
        # Backpropagation
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度更新
    # loss = loss.item()
    # print(f"Epoch {epoch + 1}\n-------------------------------")
    # print(f"loss: {loss:>7f}")  # > 右对齐，< 左对齐

# Test the model
correct = 0
test_loss = 0
total = 0

# 定义test_loop评估模型的性能。
for inputs, targets in test_loader:
    # size = len(test_loader)  # 获取测试数据集总数大小
    with torch.no_grad(): # 禁用梯度计算
    #     pred = model(inputs)
    #     test_loss += loss_fn(pred, targets).item()  # 计算预测值和真实值的误差大小，此时的测试误差是所有样本数据的误差
    #     correct += (pred.argmax(1) == targets).type(torch.float).sum().item()  # 累加所有测试数据中预测正确的样本个数
    #     total += targets.size(0)
    #
    # test_loss /= total  # 为了和训练数据做对比，在此处除上分组个数，以获取64张图片的测试误差
    # correct /= total  # 除以样本总数得到准确率
    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print('Test accuracy: {:.2f}%'.format(accuracy))

end_time = time.time()
total_time = end_time - start_time
print(f"Total training time: {total_time:.2f} seconds")

torch.save(model, "model1.pth")
print("Done!")