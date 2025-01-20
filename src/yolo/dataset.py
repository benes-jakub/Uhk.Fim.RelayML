import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.imgs = os.listdir(img_dir)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        label_path = os.path.join(self.label_dir, self.imgs[idx].replace('.bmp', '.txt'))  # Upravte podle formátu

        # Načtení obrázku
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Načtení labelů
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, center_x, center_y, width, height = map(float, line.strip().split())
                    boxes.append([class_id, center_x, center_y, width, height])

        # Převod na tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        return image, boxes

