import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
train_dataset = datasets.ImageFolder(root='data/beeant/train',
                                           transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
