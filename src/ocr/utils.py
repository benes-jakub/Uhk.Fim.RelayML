
import cv2
import numpy as np

def rotate_image(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def threshold_binary(img):
    ret,thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    return thresh

def threshold_adaptive_mean(img):
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, 10)
    return thresh

def threshold_adaptive_gaussian(img):
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY, 25, 10)
    return thresh

def threshold_adaptive_otsu(img):
    ret2,thresh = cv2.threshold(img, 70, 100, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def dilate(image):
    kernel = np.ones((4,4),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

def erode(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)   


# Removes unnecessary symbols from the string
def remove_unnecessary_symbols(result, searchTextLength):
    result = result.replace(" ", "").replace("|", "").replace("/", "").replace("\\", "").replace("\n", "").replace("!", "").replace(":", "")
    if(len(result) > searchTextLength):
        result = result[1:searchTextLength + 1]
    return result    

# Creates a string of results
def get_string_from_results(result_1, result_2, imageName):
    if not imageName:        
        return result_1 + ";" + result_2
    else:
        return imageName + ": " + result_1 + ";" + result_2