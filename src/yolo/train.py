from ultralytics import YOLO
import os

def yolo_train_and_save_model():    
    # Načtení předtrénovaného modelu YOLOv10
    model = YOLO('yolo11n.pt')

    # Trénování na vlastních datech
    model.train(data='../dataset/train/merged_yolo/data.yaml', epochs=10, imgsz=640, batch=1, device='cuda', workers=0)

    model.save('./yolo/yolo.pt')