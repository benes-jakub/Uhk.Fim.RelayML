from ultralytics import YOLO
from paths import PATH_YOLO_DATA_CONFIG
import os

def yolo_train_and_save_model(baseModelPath, customModelPath):    
    # Load of pretrained model YOLOv11
    # model = YOLO(PATH_YOLO_MODEL_V11_S)
    model = YOLO(baseModelPath)

    # Train on custom dataset
    model.train(data=PATH_YOLO_DATA_CONFIG, epochs=10, imgsz=640, batch=1, device='cuda', workers=0)

    # Save model
    # model.save(PATH_YOLO_MODEL_CUSTOM_S)
    model.save(customModelPath)