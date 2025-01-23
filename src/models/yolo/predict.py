import time
from ultralytics import YOLO

class DetectionResult:
    def __init__(self):
        self.detected_boxes = [] 
        self.time = 0 

def yolo_predict(image_path, customModelPath):     
    result = DetectionResult()

    print("Loading model " + customModelPath)
    model = YOLO(customModelPath) 

    start = time.time()
    predictions = model.predict(image_path, conf=0.5, show=False, device="cuda", verbose=True, save=True)
    end = time.time()

    result.time = end - start


    # Počítáme průměr, je nutné vynechat první obrázek, trvá nesmyslně déle
    # Pozor, ztrácí smysl, když testujeme 1 obrázek
    inference_sum = 0
    i = 0
    for p in predictions:        
        result.detected_boxes = p.boxes.xyxy.tolist()        
        if(i > 0):
            inference_sum += p.speed['inference']
        i += 1

    # avrg    
    result.time = inference_sum / (len(predictions) - 1)

    return result