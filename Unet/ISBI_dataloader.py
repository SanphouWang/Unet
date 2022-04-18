import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


class ISBI_dataloader(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_path = [f"{img_dir}/{x}" for x in os.listdir(img_dir)]
        self.label_path = [f"{label_dir}/{x}" for x in os.listdir(label_dir)]

    def augmentation(self, img, code):
        fliped = cv2.flip(img, code)
        return fliped

    def __getitem__(self, item):
        image = cv2.imread(self.img_path[item])
        label = cv2.imread(self.label_path[item])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
        if label.max() >1:
            label=label/255
        image = Image.fromarray(image)
        label = Image.fromarray(label)
        image_trans = transforms.Compose([
            transforms.Resize((572, 572)),
            transforms.ToTensor()
        ])
        image = image_trans(image)
        label = image_trans(label)
        return image, label

    def __len__(self):
        return len(self.img_path)


if __name__ == "__main__":
    isbi_dataset = ISBI_dataloader("data/train/image", "data/train/label")
    print("数据个数", len(isbi_dataset))
    img, lab = isbi_dataset[0]
    print(img.type(), lab.type())
    loader = torch.utils.data.DataLoader(dataset=isbi_dataset, batch_size=2, shuffle=False)
    for i, j in loader:
        print(i.shape, j.shape)
