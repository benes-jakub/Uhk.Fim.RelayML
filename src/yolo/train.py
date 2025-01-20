import os
import torch
import sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from yolo.dataset import YOLODataset
from PIL import Image

def yolo_train_and_save_model():    
    # Příprava transformací
    transform = transforms.Compose([
        transforms.Resize((1388, 1038)),
        transforms.ToTensor(),
    ])

    # Načtení datasetu
    dataset = YOLODataset('../dataset/train/merged', '../dataset/train/merged/labels_yolo', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(dataloader)

    
    sys.path.append('./yolo/yolov5')  # adjust the path as needed    
    from models.experimental import attempt_load        
    model = attempt_load('./models/yolov5s.pt')

    # Trénink modelu
    for epoch in range(10):  # nastavte počet epoch dle potřeby
        for images, labels in dataloader:
            images = images.cuda()  # pokud používáte GPU
            labels = labels.cuda()  # pokud používáte GPU
            
            # Trénink
            loss = model(images, labels)  # tréninková funkce se může lišit v závislosti na verzi YOLO
            print(f'Epoch: {epoch}, Loss: {loss.item()}')