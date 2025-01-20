
import sys
import cv2
import glob
import os
import shutil
from colorama import Fore, init
import numpy as np
from scipy import ndimage
import pytesseract
from .utils import OCRType, rotate_image, remove_unnecessary_symbols, get_string_from_results, threshold_binary, threshold_adaptive_mean, threshold_adaptive_gaussian, threshold_adaptive_otsu, erode, dilate
import easyocr
import keras_ocr

IMAGE_EXTENSION = ".bmp"
PATH_DEBUG_SAVE = "../debug/"


def execute_ocr(dataset_path, text_to_find, crop_regions, preprocessing, ocr_type, debug, disable_print):   
    # Instance of OCR engine - we need to load it only once
    ocr_instance = None
    if ocr_type == OCRType.EASY_OCR:
        ocr_instance = easyocr.Reader(['en'], gpu=True, detector=True, recognizer=True)                
    if ocr_type == OCRType.KERAS_OCR:
        ocr_instance = keras_ocr.pipeline.Pipeline()           

    if debug:
        shutil.rmtree(PATH_DEBUG_SAVE)
        os.mkdir(PATH_DEBUG_SAVE)
 
    # List of image blobs
    image_list = []
    # List of image names
    image_names = []

    # Load dataset
    if disable_print == False:
        print("Loading dataset...")
    # If there is an extension in the path, it is the path to the image not the directory
    if(IMAGE_EXTENSION in dataset_path):
        image_list.append(cv2.imread(dataset_path))        
        image_names.append(dataset_path.split('/')[-1])
    else:
        for filename in glob.glob(dataset_path + '/*' + IMAGE_EXTENSION):
            im = cv2.imread(filename)            
            image_names.append(filename.rsplit('\\', 1)[-1])
            image_list.append(im)
    if disable_print == False:
        print("Dataset loaded!")
        print("\n")

        print("Looking for text...")
    
    count_positive = 0
    for ind, img in enumerate(image_list, start = 0): 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

        # There are 2 relays in the picture. We know the approximate position of each relay and the text on it. A crop is made.    
        cropped_image_1 = img[crop_regions[ind][0][1] - 270:crop_regions[ind][0][1] - 10, crop_regions[ind][0][0] - 10:crop_regions[ind][0][2] - 10]    
        cropped_image_2 = img[crop_regions[ind][1][1] - 270:crop_regions[ind][1][1] - 10, crop_regions[ind][1][0] - 10:crop_regions[ind][1][2] - 10]  

        # Rotate the images 90 degrees 
        rotated_image_1 = ndimage.rotate(cropped_image_1, -90)
        rotated_image_2 = ndimage.rotate(cropped_image_2, -90)        

        # Appy preprocessing
        if preprocessing == "binary":
            rotated_image_1 = threshold_binary(rotated_image_1)
            rotated_image_2 = threshold_binary(rotated_image_2)
        if preprocessing == "mean":
            rotated_image_1 = threshold_adaptive_mean(rotated_image_1)
            rotated_image_2 = threshold_adaptive_mean(rotated_image_2)
        if preprocessing == "gaus":
            rotated_image_1 = threshold_adaptive_gaussian(rotated_image_1)
            rotated_image_2 = threshold_adaptive_gaussian(rotated_image_2)            
        if preprocessing == "otsu":
            rotated_image_1 = threshold_adaptive_otsu(rotated_image_1)
            rotated_image_2 = threshold_adaptive_otsu(rotated_image_2)
        if preprocessing == "erode":
            rotated_image_1 = erode(rotated_image_1)
            rotated_image_2 = erode(rotated_image_2)
        if preprocessing == "dilate":
            rotated_image_1 = dilate(rotated_image_1)
            rotated_image_2 = dilate(rotated_image_2)                        
        if preprocessing == "dilate_mean":
            rotated_image_1 = dilate(rotated_image_1)
            rotated_image_2 = dilate(rotated_image_2)                        
            rotated_image_1 = threshold_adaptive_mean(rotated_image_1)
            rotated_image_2 = threshold_adaptive_mean(rotated_image_2)

        # Show images for debug
        if debug:                   
            cv2.imwrite(PATH_DEBUG_SAVE + "cropped_image_1_" + image_names[ind], rotated_image_1)
            cv2.imwrite(PATH_DEBUG_SAVE + "cropped cropped_image_2_" + image_names[ind], rotated_image_2)
            # cv2.imshow("cropped image 1 " + image_names[ind], rotated_image_1)
            # cv2.imshow("cropped image 2 " + image_names[ind], rotated_image_2)

        # Tesseract config
        # https://muthu.co/all-tesseract-ocr-options/
        # tesseract --help-oem
        # tesseract --help-psm        
        if ocr_type == OCRType.TESSERACT:
            custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=' + text_to_find            
            # Execute Tesseract for both images
            result_1 = pytesseract.image_to_string(rotated_image_1, config=custom_config).upper()
            result_2 = pytesseract.image_to_string(rotated_image_2, config=custom_config).upper()

        # EasyOCR config
        # https://www.jaided.ai/easyocr/documentation/
        if ocr_type == OCRType.EASY_OCR:                        
            result_1 = ocr_instance.readtext(rotated_image_1, detail=0, allowlist=text_to_find, batch_size=200)[0]
            result_2 = ocr_instance.readtext(rotated_image_2, detail=0, allowlist=text_to_find, batch_size=200)[0]    
        
        if ocr_type == OCRType.KERAS_OCR:
            keras_image_1 = cv2.cvtColor(rotated_image_1, cv2.COLOR_BGR2RGB)
            keras_image_2 = cv2.cvtColor(rotated_image_2, cv2.COLOR_BGR2RGB)
            predictions = ocr_instance.recognize([keras_image_1, keras_image_2])            
            result_1 = predictions[0][0][0].replace("s", "5").replace("i", "1").replace("a", "4").replace("o", "0").upper()
            result_2 = predictions[1][0][0].replace("s", "5").replace("i", "1").replace("a", "4").replace("o", "0").upper()        

        # Remove unnecessary symbols
        result_1 = remove_unnecessary_symbols(result_1, len(text_to_find))
        result_2 = remove_unnecessary_symbols(result_2, len(text_to_find))

        text_found = False
        # Count and print result
        if (result_1 == text_to_find and result_2 == text_to_find):            
            text_found = True
        
        if result_1 == text_to_find:
            count_positive += 1

        if result_2 == text_to_find:
            count_positive += 1

        if disable_print == False:
            if text_found:
                print(Fore.GREEN + get_string_from_results(result_1, result_2, image_names[ind]))        
            else:
                print(Fore.RED + get_string_from_results(result_1, result_2, image_names[ind]))

    number_of_rele = len(image_names) * 2
    # Print final results    
    if disable_print == False:
        print("\n")
        print(Fore.MAGENTA + "Number of relays: " + str(number_of_rele))
        print("Positive found: " + str(count_positive))
        print("Not found: " + str(number_of_rele - count_positive))        
        print("Reliability: " + str(count_positive / number_of_rele))


    return "ok"

    if debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        

    

