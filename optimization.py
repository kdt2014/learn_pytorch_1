import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

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

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = MyNetwork()
print(model)



# Initialize the loss function


# 定义train_loop循环优化代码
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 获取训练数据集总数大小
    for batch, (X, y) in enumerate(dataloader):  # 从数据加载器中获取批次个数，图像数据和对应标签, batch是一个从0开始增加，直到dataloader最后一个数的序号
        # Compute prediction and loss
        pred = model(X)  # 将图像数据通过训练模型
        loss = loss_fn(pred, y)  # 计算预测值和真实值的误差大小

        # Backpropagation
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度更新

        if batch % 100 == 0:  # 因为批次大小为64，所以每获取100个批次，也就是6400张图片时输出此时的误差
            loss, current = loss.item(), batch * len(X)  # 获取当前的误差，batch * len(X)获取当前训练后的图片数量,batch=100,200,len(x)=64
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") # > 右对齐，< 左对齐

# 定义test_loop评估模型的性能。
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # 获取测试数据集总数大小
    num_batches = len(dataloader)  # 获取测试样本数据集中的组数，10000/64=157
    test_loss, correct = 0, 0  # 将之前的测试误差和准确率清零

    with torch.no_grad():  # 禁用梯度计算
        for X, y in dataloader:  # 从数据加载器中获取图像数据和对应标签
            pred = model(X)
            test_loss += loss_fn(pred, y).item()  # 计算预测值和真实值的误差大小，此时的测试误差是所有样本数据的误差
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 累加所有测试数据中预测正确的样本个数
            # 在这段代码中， pred.argmax(1)返回每个输入样本对应的模型预测输出中最大值的索引，即预测类别的索引。而 y包含了每个输入样本的真实类别标签。
            # 因此，pred.argmax(1) == y 的作用是比较模型的预测输出和真实类别标签是否一致，返回一个布尔值的数组。
            # 如果模型的预测输出和真实类别标签一致，则对应位置的元素为True，否则为False。
            # 接着，.type(torch.float)将布尔值数组转换为浮点数数组，其中 True 转换为1.0，False 转换为0.0。
            # 最后，.sum().item()计算浮点数数组中所有元素的总和，即为当前批次中预测正确的样本数。
    test_loss /= num_batches  # 为了和训练数据做对比，在此处除上分组个数，以获取64张图片的测试误差
    correct /= size  # 除以样本总数得到准确率
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

learning_rate = 1e-3 #学习率
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
epochs = 10

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)  # 输入参数，进行训练
    test_loop(test_dataloader, model, loss_fn)  # 输入参数，进行测试
print("Done!")