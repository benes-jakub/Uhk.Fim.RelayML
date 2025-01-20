import torch
from enum import Enum
import torchvision
import glob
from fasterrcnn.predict import fasterrcnn_predict
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import numpy as np
from .utils import PATH_MODEL


class MLModel(Enum): 
    FASTER_RCNN = 1    

def find_boxes_fasterrcnn(image_path):
    boxes = []

    # Load model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    model = fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)

    model.load_state_dict(torch.load(PATH_MODEL, weights_only=True))        
    model.to(device)

    # Prediction
    model.eval()

    if ".bmp" in image_path:        
            boxes.append(fasterrcnn_predict(image_path, model, device))  
    else:
        for filename in glob.glob(image_path + "/*.bmp"):                
            boxes.append(fasterrcnn_predict(filename, model, device))    

    boxes = np.array(boxes)
    boxes = np.round(boxes).astype(int)

    return boxes

    