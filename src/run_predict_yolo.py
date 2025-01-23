from models.yolo.predict import yolo_predict
from paths import PATH_YOLO_MODEL_CUSTOM_S, PATH_YOLO_MODEL_CUSTOM_N, PATH_YOLO_MODEL_CUSTOM_M

# result = yolo_predict("../dataset/test/PE514F03/image_638096484607584580.bmp")
result = yolo_predict("../dataset/test/merged/", PATH_YOLO_MODEL_CUSTOM_M)
print(str(result.time) + "ms")
