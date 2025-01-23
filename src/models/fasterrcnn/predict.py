import time
import torch
import torchvision
import glob
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from paths import PATH_FASTER_RCNN_MODEL, PATH_FASTER_RCNN_DEBUG

class DetectionResult:
    def __init__(self):
        self.detected_boxes = [] 
        self.time = 0 

def fasterrcnn_predict(image_path):
    # Load model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    model = fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(PATH_FASTER_RCNN_MODEL, weights_only=True))        
    model.to(device)
    model.eval()
    
    result = DetectionResult()
    
    if ".bmp" in image_path:      
            prediction = predict(image_path, model, device)
            result.detected_boxes.append(prediction[0])  
            result.time = prediction[1]
    else:
        count = 0
        for filename in glob.glob(image_path + "/*.bmp"):                            
            prediction = predict(filename, model, device)
            result.detected_boxes.append(prediction[0])  
            if count > 0:
                result.time += prediction[1]
            count += 1

    result.detected_boxes = np.array(result.detected_boxes)
    result.detected_boxes = np.round(result.detected_boxes).astype(int)
    result.time = result.time / (len(image_path) - 1)

    return result


def predict(image_path, model, device):      
    image = Image.open(image_path).convert("RGB")    
    image = image.resize((640, 480))
    # Převedeme obrázek na tensor a přesuneme na správné zařízení (např. GPU)
    transform = ToTensor()
    input_image = transform(image).unsqueeze(0).to(device)  # Přidání dimenze pro batch a přesun na zařízení    

    # Získání predikcí z modelu       
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        prediction = model(input_image)
        torch.cuda.synchronize()
        end = time.time()                

    boxes = prediction[0]['boxes'].cpu().numpy()
    # plot_image_with_boxes(image, image_path.split("\\")[1], boxes)    
    return [boxes, (end - start) * 1000]    


def plot_image_with_boxes(image, name, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    # plt.show()
    
    plt.savefig(PATH_FASTER_RCNN_DEBUG + "/" + name.replace(".bmp", ".png"), bbox_inches='tight')

    