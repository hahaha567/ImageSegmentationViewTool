import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import torch
import torch.utils.data as Data
import cv2
import os



    
class MyDataset(Data.Dataset):
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.filenames = os.listdir(dataPath + "/images")
        self.transform = transforms.Compose([
            # transforms.Resize((288,288)),
            transforms.ToTensor()
        ])
        self.transform2 = None
        # self.transform2 = transforms.Compose([
        #     transforms.Normalize(mean, std)
        # ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        imgPath = self.dataPath + "/images/" + self.filenames[item]
        labelPath = self.dataPath + "/labels/" + self.filenames[item][:-4] + ".png"
        img = self.transform(Image.open(imgPath))
        label = self.transform(Image.open(labelPath))
        # 如果有，对图像进行归一化
        if self.transform2 is not None:
            img = self.transform2(img)
        return img, label

if __name__=="__main__":
    dataset = MyDataset("./swinyseg/train")
    trainloader = Data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)
    for i_batch, (image_batch, label_batch) in enumerate(trainloader):
        print(image_batch.shape)
        print(image_batch)
        print(label_batch.shape)
        print(label_batch)
        break



