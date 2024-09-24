from enum import Enum
import glob
from fasterrcnn.predict import fasterrcnn_predict
import numpy as np

class MLModel(Enum):
    FASTER_RCNN = 1    

def get_boxes(model, image_path):
    boxes = []

    if ".bmp" in image_path:
        if model == MLModel.FASTER_RCNN:
                boxes.append(fasterrcnn_predict(image_path))  
    else:
        for filename in glob.glob(image_path + "/*.bmp"):    
            if model == MLModel.FASTER_RCNN:
                boxes.append(fasterrcnn_predict(filename))    

    boxes = np.array(boxes)
    boxes = np.round(boxes).astype(int)

    return boxes

    