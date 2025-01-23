import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(voc_dir, yolo_output_dir, class_name='logo'):
    # Zajistíme, že výstupní adresář existuje
    if not os.path.exists(yolo_output_dir):
        os.makedirs(yolo_output_dir)

    # Projdeme všechny XML soubory v zadaném adresáři
    for file in os.listdir(voc_dir):
        if file.endswith(".xml"):
            voc_file = os.path.join(voc_dir, file)
            tree = ET.parse(voc_file)
            root = tree.getroot()

            # Načteme velikost obrázku z VOC souboru
            width = int(root.find('size/width').text)
            height = int(root.find('size/height').text)

            yolo_labels = []

            # Pro každý objekt v anotaci
            for obj in root.findall('object'):
                name = obj.find('name').text

                if name == class_name:
                    # Načtení bounding boxu
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)

                    # Převod na YOLO formát (normované souřadnice)
                    x_center = (xmin + xmax) / 2 / width
                    y_center = (ymin + ymax) / 2 / height
                    box_width = (xmax - xmin) / width
                    box_height = (ymax - ymin) / height

                    # Class ID je 0, protože máme jen jednu třídu 'logo'
                    yolo_labels.append(f"0 {x_center} {y_center} {box_width} {box_height}")

            # Uložení YOLO labelu do .txt souboru
            output_filename = file.replace(".xml", ".txt")
            output_path = os.path.join(yolo_output_dir, output_filename)

            with open(output_path, 'w') as f:
                f.write("\n".join(yolo_labels))

# Cesty
voc_dir = '../../dataset/train/merged/labels'  # Adresář s XML soubory
yolo_output_dir = '../../dataset/train/merged/labels_yolo'  # Adresář pro výstupní YOLO soubory

# Spustíme převod pro všechny soubory v adresáři
convert_voc_to_yolo(voc_dir, yolo_output_dir)
