import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

datapath = r'data/MyDataset/traindata'
txtpath = r'data/MyDataset/label.txt'

class MyDataset(Dataset):
    def __init__(self,txtpath):

        #创建一个list用来储存图片信息和标签信息
        imgs = []

        #打开第一步创建的TXT，按行读取，将结果以元组的形式保存在imgs里
        datainfo = open(txtpath, 'r')
        for line in datainfo:
            line = line.strip('\n') #同时去掉左右两边的空格
            words = line.split() #以空格为分割进行切片
            imgs.append((words[0], words[1]))

        self.imgs = imgs

    #返回数据集大小
    def __len__(self):
        return len(self.imgs)

    #打开index对应图片进行预处理后返回处理后的图片和标签，将数据转换成tensor格式
    def __getitem__(self, index): #按照索引读取每个元素的具体内容
        pic,label = self.imgs[index]
        pic = Image.open(datapath+'\\'+pic)
        pic = pic.resize((224, 224))  # 设置每一张图片的大小
        pic = transforms.ToTensor()(pic)
        # label = transforms.ToTensor()(label)
        return pic,label
#实例化对象
data = MyDataset(txtpath)

data_loader = DataLoader(data, batch_size=1,shuffle=True,num_workers=0)

for pics,label in data_loader:
    print(pics,label)