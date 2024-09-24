import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import ToTensor
from .dataset import VOCDataset
from .utils import PATH_DATASET_IMAGES, PATH_DATASET_ANNOTATIONS, PATH_MODEL

def collate_fn(batch):
    return tuple(zip(*batch))


def fasterrcnn_train_and_save_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    print("Device: " + str(device))

    # Načtení všech obrázků
    train_dataset = VOCDataset(PATH_DATASET_IMAGES, PATH_DATASET_ANNOTATIONS, transforms=ToTensor())

    # Rozdělení datasetu na trénovací a testovací sadu (80% trénink, 20% test)
    # train_size = int(0.8 * len(voc_dataset))
    # test_size = len(voc_dataset) - train_size
    # train_dataset, test_dataset = random_split(voc_dataset, [train_size, test_size])    

    # Vytvoření DataLoaderu
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Definice Faster R-CNN modelu
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)  # Použití předtrénovaného modelu        
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)

    # Definice optimalizátoru
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 10

    # Set device
    model.to(device)

    # Tréninkový cyklus
    model.train()
    for epoch in range(num_epochs):         
        img_counter = 0
        for images, targets in train_loader:                                       
            img_counter += 1              
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]      

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {losses.item():.4f}")

    torch.save(model.state_dict(), PATH_MODEL)