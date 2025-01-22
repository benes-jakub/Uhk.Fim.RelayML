from ultralytics import YOLO

def yolo_predict(image_path):        
    model = YOLO('./yolo/yolo.pt') 

    results = model.predict(image_path, conf=0.8, show=True, device="cuda")