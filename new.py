import torch
import torchvision
import time

# Load FashionMNIST dataset and convert to PyTorch tensors
train_data = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True,
    transform=torchvision.transforms.ToTensor()
)
test_data = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True,
    transform=torchvision.transforms.ToTensor()
)
train_images, train_labels = train_data.data.to('cuda'), train_data.targets.to('cuda')
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

# Define MLP model
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(784, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Initialize model and optimizer
model = MLP().to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


start_time = time.time()
# Train the model
for epoch in range(45):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

# Test the model
correct = 0
total = 0
for inputs, targets in test_loader:
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print('Test accuracy: {:.2f}%'.format(accuracy))

end_time = time.time()
total_time = end_time - start_time
print(f"Total training time: {total_time:.2f} seconds")