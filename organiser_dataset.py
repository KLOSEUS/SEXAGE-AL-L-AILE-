"""
A lancer UNE SEULE FOIS pour organiser le dataset en train/val
"""
import os, shutil, random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

random.seed(42)

for cls in ["female", "male"]:
    src = os.path.join(DATA_DIR, cls)
    if not os.path.exists(src):
        print(f"Dossier introuvable : {src}")
        continue

    images = [f for f in os.listdir(src)
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    random.shuffle(images)

    split      = int(0.8 * len(images))
    train_imgs = images[:split]
    val_imgs   = images[split:]

    for subset, imgs in [("train", train_imgs), ("val", val_imgs)]:
        dest = os.path.join(DATA_DIR, subset, cls)
        os.makedirs(dest, exist_ok=True)
        for img in imgs:
            shutil.copy(os.path.join(src, img), os.path.join(dest, img))

    print(f"{cls} : {len(train_imgs)} train | {len(val_imgs)} val")

print("\nDataset organise !")
print(f"Structure creee dans : {DATA_DIR}")
