import os
import shutil
import random

image_dir = "../../dataset/train/merged_yolo/images"
txt_dir = "../../dataset/train/merged_yolo/labels"

# Získání seznamu všech obrázků v image_dir
image_files = [f for f in os.listdir(image_dir) if f.endswith(".bmp")]

if len(image_files) < 100:
    raise ValueError(f"V adresáři {image_dir} je méně než {num_samples} obrázků.")

# Náhodný výběr 100 obrázků
selected_images = random.sample(image_files, 100)

for image in selected_images:
    base_name = os.path.splitext(image)[0]  # Získání názvu souboru bez přípony
    txt_file = base_name + ".txt"
    
    image_path = os.path.join(image_dir, image)
    txt_path = os.path.join(txt_dir, txt_file)
    
    if os.path.exists(txt_path):
        # Kopírování obrázku
        shutil.move(image_path, os.path.join("../../dataset/train/merged_yolo/images/val", image))
        # Kopírování odpovídajícího txt souboru
        shutil.move(txt_path, os.path.join("../../dataset/train/merged_yolo/labels/val", txt_file))
    else:
        print(f"Varování: Pro {image} neexistuje odpovídající TXT soubor.")
