from fasterrcnn.train import fasterrcnn_train_and_save_model
from yolo.train import yolo_train_and_save_model
from fasterrcnn.get_boxes import find_boxes_fasterrcnn, load_model_fasterrcnn, MLModel
import time
import glob
from ocr.ocr import execute_ocr, OCRType

# fasterrcnn_train_and_save_model()

#path = "../dataset_BK/test/images/PE514F03/image_638096494070302939.bmp"
path = "../dataset/test/PE514F03/"
text_to_find = "PE514F03"
model_type = MLModel.FASTER_RCNN
ocr_type = OCRType.TESSERACT

# Find logo
boxes = []
if model_type == MLModel.FASTER_RCNN:
    timer_start_load_model = time.time()
    modelDevice = load_model_fasterrcnn()
    timer_end_load_model = time.time()

    timer_start_logo = time.time()
    boxes = find_boxes_fasterrcnn(path, modelDevice[0], modelDevice[1])
timer_end_logo = time.time()


# Find PN, crop and OCR
timer_start_ocr = time.time()
execute_ocr(dataset_path=path, text_to_find=text_to_find, crop_regions=boxes, preprocessing="none", ocr_type=ocr_type, debug=True, disable_print=False)
timer_end_ocr = time.time()

# Results
# divider = 1
divider = len(boxes)
print("---")
print("Image count: " + str(divider))
print("Load model time: " + str((timer_end_load_model - timer_start_load_model) / divider) + "s")
print("Duration AI find model (AVRG per image): " + str((timer_end_logo - timer_start_logo) / divider) + "s")
print("Duration OCR (AVRG per image) " + str((timer_end_ocr - timer_start_ocr) / divider) + "s")
print("Duration sum (AVRG per image): " + str(((timer_end_ocr - timer_start_ocr) + (timer_end_logo - timer_start_logo)) / divider) + "s")
