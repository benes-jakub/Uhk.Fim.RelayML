import os
import cv2
import imgaug.augmenters as iaa
from imgaug import parameters as iap
import numpy as np
from glob import glob

# Funkce pro načtení obrázku z cesty
def load_image(image_path):
    return cv2.imread(image_path)

# Funkce pro uložení obrázku do výstupní cesty
def save_image(image, output_path):
    cv2.imwrite(output_path, image)

# Definice augmentační sekvence
def augment_images(image):
    aug = iaa.Sequential([
        iaa.SomeOf((1, 3), [ # Použij 1 až 3 augmentační metody
            iaa.Affine(rotate=(-2, 2)),                 # Rotace o -10° až +10°            
            iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),# Gaussianův šum
            iaa.MotionBlur(k=3),                          # Motion blur
            iaa.Multiply((0.8, 1.2)),                     # Změna jasu
            iaa.LinearContrast((0.8, 1.2)),               # Změna kontrastu
            iaa.Crop(percent=(0, 0.05))                    # Oříznutí až 20% obrázku
        ], random_order=True)
    ])
    # Vygeneruje 3-5 augmentovaných verzí obrázku
    return [aug(image=image) for _ in range(np.random.randint(3, 6))]

# Funkce pro zpracování obrázků
def process_images(input_dir, output_dir):
    # Načte všechny obrázky z input_dir
    image_paths = glob(os.path.join(input_dir, "*.bmp"))  # Změň příponu podle formátu obrázků    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, image_path in enumerate(image_paths):
        image = load_image(image_path)
        
        if image is None:
            print(f"Chyba při načítání obrázku: {image_path}")
            continue
        
        # Vytvoř augmentované obrázky
        augmented_images = augment_images(image)
        
        # Ulož augmentované obrázky do output_dir
        for j, aug_img in enumerate(augmented_images):
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_filename}_aug_{j}.bmp")
            save_image(aug_img, output_path)
            print(f"Uložený obrázek: {output_path}")

# Hlavní funkce
if __name__ == "__main__":    
    input_dir = "../../dataset/train/PE514024"  # Cesta k adresáři s originálními obrázky
    output_dir = "../../dataset/train/PE514024_augmented" # Cesta k adresáři, kam uložit augmentované obrázky

    process_images(input_dir, output_dir)
