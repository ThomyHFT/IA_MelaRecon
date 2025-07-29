import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

#config

RAW_DIR = "../data/raw/archive"
PROCESSED_DIR = "../data/processed/"
IMG_SIZE = (300, 300)  # EfficientNetB3 use 300x300

os.makedirs(PROCESSED_DIR, exist_ok=True)

metadata_path = os.path.join(RAW_DIR, "HAM10000_metadata.csv")
df = pd.read_csv(metadata_path)

# image folders
image_folder_1 = os.path.join(RAW_DIR, "HAM10000_images_part_1")
image_folder_2 = os.path.join(RAW_DIR, "HAM10000_images_part_2")

for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
    image_id = row['image_id']
    label = row['dx']

    filename = image_id + ".jpg"
    img_path = os.path.join(image_folder_1, filename)
    if not os.path.exists(img_path):
        img_path = os.path.join(image_folder_2, filename)
    if not os.path.exists(img_path):
        continue

    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMG_SIZE)

        # bin qualify
        binary_label = 'melanoma' if label == 'mel' else 'no_melanoma'

        label_dir = os.path.join(PROCESSED_DIR, binary_label)
        os.makedirs(label_dir, exist_ok=True)

        save_path = os.path.join(label_dir, filename)
        img.save(save_path)

    except Exception as e:
        print(f"Error con {filename}: {e}")
