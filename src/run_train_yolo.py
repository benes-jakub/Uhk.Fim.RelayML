from models.yolo.train import yolo_train_and_save_model
from paths import PATH_YOLO_MODEL_V11_S, PATH_YOLO_MODEL_V11_N, PATH_YOLO_MODEL_V11_M, PATH_YOLO_MODEL_CUSTOM_S, PATH_YOLO_MODEL_CUSTOM_N, PATH_YOLO_MODEL_CUSTOM_M

yolo_train_and_save_model(PATH_YOLO_MODEL_V11_M, PATH_YOLO_MODEL_CUSTOM_M)
