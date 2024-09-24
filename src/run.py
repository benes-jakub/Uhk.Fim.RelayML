from fasterrcnn.train import fasterrcnn_train_and_save_model
from core.get_boxes import get_boxes, MLModel
import glob
from ocr.ocr import execute_ocr

# fasterrcnn_train_and_save_model()
# fasterrcnn_predict("../dataset/test/images/PE014012/image_637873439951791312.bmp")

path = "../dataset/test/images/PE014012"
# path = "../dataset/test/images/PE014012/image_637873439795232772.bmp"
text_to_find = "PE014012"
model = MLModel.FASTER_RCNN


boxes = get_boxes(model, path)
execute_ocr(dataset_path=path, text_to_find=text_to_find, crop_regions=boxes, accuracy=1, preprocessing="none", debug=True, disable_print=False)