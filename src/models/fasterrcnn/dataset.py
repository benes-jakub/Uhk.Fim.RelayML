import os
import xml.etree.ElementTree as ET
import torch
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, annotation_dir, transforms=None):        
        self.root_dir = root_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.images = [f for f in os.listdir(root_dir) if f.endswith(".bmp")]
        self.annotations = [f for f in os.listdir(annotation_dir) if f.endswith(".xml")]                

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):        
        # Načtení obrázku
        img_path = os.path.join(self.root_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        # Načtení anotací
        annotation_file = os.path.join(self.annotation_dir, self.annotations[idx])
        boxes, labels = self.parse_voc_xml(annotation_file)

        # Vytvoření cílového slovníku (target)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }       

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def parse_voc_xml(self, xml_file):                
        tree = ET.parse(xml_file)
        root = tree.getroot()     

        boxes = []
        labels = []

        for obj in root.findall('object'):
            label = obj.find('name').text            
            labels.append(1 if label == "logo" else 0)  # Uprav podle svých tříd            

            bbox = obj.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)
            
            boxes.append([x_min, y_min, x_max, y_max])

        return boxes, labels
