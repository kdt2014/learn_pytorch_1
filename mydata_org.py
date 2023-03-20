import torch
import torchvision

import os
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

annotations_file = r'''C:\Users\kongd\PycharmProjects\learn_pytorch\data\MyDataset\label.csv'''
img_dir = r'''C:\Users\kongd\PycharmProjects\learn_pytorch\MyDataset\traindata'''

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        image = self.img_labels.iloc[idx, 0]
        image = Image.open(img_dir+'\\'+image)
        label = self.img_labels.iloc[idx, 1]
        image = transforms.ToTensor()(image)
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label

data = CustomImageDataset(annotations_file, img_dir)

data_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0)

for pic, label in data_loader:
    print(pic, label)