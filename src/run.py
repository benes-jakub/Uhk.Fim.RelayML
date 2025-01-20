from fasterrcnn.train import fasterrcnn_train_and_save_model
from yolo.train import yolo_train_and_save_model
from fasterrcnn.get_boxes import find_boxes_fasterrcnn, MLModel
import time
import glob
from ocr.ocr import execute_ocr, OCRType

# fasterrcnn_train_and_save_model()
# fasterrcnn_predict("../dataset/test/images/PE014012/image_637873439951791312.bmp")

yolo_train_and_save_model()

#path = "../dataset_BK/test/images/PE514F03/image_638096494070302939.bmp"
# path = "../dataset/test/PE514024/"
# text_to_find = "PE514024"
# model_type = MLModel.FASTER_RCNN
# ocr_type = OCRType.KERAS_OCR

# # Find logo
# timer_start_logo = time.time()
# boxes = []
# if model_type == MLModel.FASTER_RCNN:
#     boxes = find_boxes_fasterrcnn(path)
# timer_end_logo = time.time()


# # Find PN, crop and OCR
# timer_start_ocr = time.time()
# execute_ocr(dataset_path=path, text_to_find=text_to_find, crop_regions=boxes, preprocessing="none", ocr_type=ocr_type, debug=True, disable_print=False)
# timer_end_ocr = time.time()

# # Results
# print("---")
# print("Duration AI find model: " + str((timer_end_logo - timer_start_logo) / len(boxes)) + "s")
# print("Duration OCR: " + str((timer_end_ocr - timer_start_ocr) / len(boxes)) + "s")
# print("Duration sum: " + str(((timer_end_ocr - timer_start_ocr) + (timer_end_logo - timer_start_logo)) / len(boxes)) + "s")
