import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from .dataset import VOCDataset
from .utils import PATH_MODEL, PATH_DATASET_IMAGES, PATH_DATASET_ANNOTATIONS

def plot_image_with_boxes(image, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


def fasterrcnn_predict(image_path):        
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    print("Device: " + str(device))

    # Load model
    model = fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)

    model.load_state_dict(torch.load(PATH_MODEL, weights_only=True))        
    model.to(device)

    # Prediction
    model.eval()
        
    image = Image.open(image_path).convert("RGB")

    # Převedeme obrázek na tensor a přesuneme na správné zařízení (např. GPU)
    transform = ToTensor()
    input_image = transform(image).unsqueeze(0).to(device)  # Přidání dimenze pro batch a přesun na zařízení    

    # Získání predikcí z modelu
    with torch.no_grad():
        prediction = model(input_image)

    boxes = prediction[0]['boxes'].cpu().numpy()
    
    plot_image_with_boxes(image, boxes)

    