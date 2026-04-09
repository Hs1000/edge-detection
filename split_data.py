import os
import shutil
import random

random.seed(42)

image_dir = "train/images"
label_dir = "train/labels"

train_img_dir = "train_data/images"
train_lbl_dir = "train_data/labels"

val_img_dir = "val_data/images"
val_lbl_dir = "val_data/labels"

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(images)

split_idx = int(0.9 * len(images))

train_images = images[:split_idx]
val_images = images[split_idx:]

def safe_copy(src, dst):
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy(src, dst)

for img in train_images:
    src_img = os.path.join(image_dir, img)
    dst_img = os.path.join(train_img_dir, img)

    safe_copy(src_img, dst_img)

    lbl = img.replace(".jpg", ".txt").replace(".png", ".txt")
    src_lbl = os.path.join(label_dir, lbl)
    dst_lbl = os.path.join(train_lbl_dir, lbl)

    safe_copy(src_lbl, dst_lbl)

for img in val_images:
    src_img = os.path.join(image_dir, img)
    dst_img = os.path.join(val_img_dir, img)

    safe_copy(src_img, dst_img)

    lbl = img.replace(".jpg", ".txt").replace(".png", ".txt")
    src_lbl = os.path.join(label_dir, lbl)
    dst_lbl = os.path.join(val_lbl_dir, lbl)

    safe_copy(src_lbl, dst_lbl)

print(f"✅ Done! Train: {len(train_images)}, Val: {len(val_images)}")