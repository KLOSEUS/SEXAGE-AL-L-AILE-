import os, shutil, random

BASE_DIR   = r"C:\Users\guill\Desktop\IA_Poussins"
MALE_SRC   = os.path.join(BASE_DIR, "data", "male")
FEMALE_SRC = os.path.join(BASE_DIR, "data", "female")

DOSSIERS = {
    "train/male"   : os.path.join(BASE_DIR, "data", "train", "male"),
    "train/female" : os.path.join(BASE_DIR, "data", "train", "female"),
    "val/male"     : os.path.join(BASE_DIR, "data", "val",   "male"),
    "val/female"   : os.path.join(BASE_DIR, "data", "val",   "female"),
}

# Creer les dossiers
for chemin in DOSSIERS.values():
    os.makedirs(chemin, exist_ok=True)

def splitter(source, dest_train, dest_val, label):
    extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")
    fichiers   = [f for f in os.listdir(source)
                  if f.lower().endswith(extensions)]
    random.seed(42)
    random.shuffle(fichiers)
    split = int(0.8 * len(fichiers))
    for f in fichiers[:split]:
        shutil.copy(os.path.join(source, f), dest_train)
    for f in fichiers[split:]:
        shutil.copy(os.path.join(source, f), dest_val)
    print(f"  {label} -> Train : {split} | Val : {len(fichiers)-split}")

print("=" * 45)
print("  ORGANISATION DU DATASET")
print("=" * 45)
splitter(MALE_SRC,   DOSSIERS["train/male"],
         DOSSIERS["val/male"],   "Males   ")
splitter(FEMALE_SRC, DOSSIERS["train/female"],
         DOSSIERS["val/female"], "Femelles")

print("\n  RESUME FINAL")
print("=" * 45)
for nom, chemin in DOSSIERS.items():
    print(f"  {nom:<20} : {len(os.listdir(chemin))} images")
print("\n✅ Organisation terminee !")