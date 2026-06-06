import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os, time, json
from model import creer_modele

# ══════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════
BASE_DIR  = r"C:\Users\guill\Desktop\IA SECOURS\IA_Poussins"
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
VAL_DIR   = os.path.join(BASE_DIR, "data", "val")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# APRES
EPOCHS     = 20
BATCH_SIZE = 16
LR         = 0.0005   # Plus petit car on decouvre plus de couches
DEVICE     = torch.device("cpu")

print(f"Appareil : {DEVICE}")

# ══════════════════════════════════════════
#  TRANSFORMATIONS
# ══════════════════════════════════════════
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ══════════════════════════════════════════
#  CHARGEMENT DONNEES
# ══════════════════════════════════════════
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=val_transforms)

# Gestion du desequilibre male/femelle
class_counts  = [0, 0]
for _, label in train_dataset:
    class_counts[label] += 1

weights       = [1.0 / class_counts[label] for _, label in train_dataset]
sampler       = WeightedRandomSampler(weights, len(weights))

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                           sampler=sampler, num_workers=0)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=0)

print(f"Classes  : {train_dataset.classes}")
print(f"Train    : {len(train_dataset)} images")
print(f"Val      : {len(val_dataset)} images")
print(f"Males    : {class_counts[0]} | Femelles : {class_counts[1]}")

# ══════════════════════════════════════════
#  MODELE
# ══════════════════════════════════════════
modele    = creer_modele().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, modele.parameters()),
    lr=LR
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
# ══════════════════════════════════════════
#  CHARGEMENT CHECKPOINT
# ══════════════════════════════════════════
BEST_MODEL = os.path.join(MODEL_DIR, "meilleur_modele.pth")

if os.path.exists(BEST_MODEL):
    modele.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
    print(f"Poids charges depuis meilleur_modele.pth")
else:
    print("Aucun checkpoint -> from scratch")

# ══════════════════════════════════════════
#  FONCTIONS
# ══════════════════════════════════════════
def train_epoch(modele, loader, optimizer, criterion):
    modele.train()
    total_loss, correct, total = 0, 0, 0
    for batch_idx, (images, labels) in enumerate(loader):
        optimizer.zero_grad()
        sorties = modele(images)
        loss    = criterion(sorties, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds       = sorties.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} "
                  f"| Loss: {total_loss/(batch_idx+1):.4f} "
                  f"| Acc: {100*correct/total:.2f}%")
    return total_loss / len(loader), 100 * correct / total

def val_epoch(modele, loader, criterion):
    modele.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            sorties     = modele(images)
            loss        = criterion(sorties, labels)
            total_loss += loss.item()
            preds       = sorties.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
    return total_loss / len(loader), 100 * correct / total

# ══════════════════════════════════════════
#  ENTRAINEMENT
# ══════════════════════════════════════════
meilleure_acc = 0.0
historique    = []

print("\n" + "="*50)
print("      DEBUT DE L'ENTRAINEMENT")
print("="*50)

for epoch in range(1, EPOCHS + 1):
    debut = time.time()
    print(f"\nEpoch {epoch}/{EPOCHS}")

    train_loss, train_acc = train_epoch(modele, train_loader,
                                        optimizer, criterion)
    val_loss,   val_acc   = val_epoch(modele, val_loader, criterion)
    scheduler.step()
    duree = time.time() - debut

    historique.append({
        "epoch": epoch,
        "train_loss": round(train_loss, 4),
        "train_acc" : round(train_acc,  2),
        "val_loss"  : round(val_loss,   4),
        "val_acc"   : round(val_acc,    2)
    })

    print(f"  Train  — Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
    print(f"  Val    — Loss: {val_loss:.4f}   | Acc: {val_acc:.2f}%")
    print(f"  Duree  : {duree:.1f}s")

    if val_acc > meilleure_acc:
        meilleure_acc = val_acc
        chemin = os.path.join(MODEL_DIR, "meilleur_modele.pth")
        torch.save(modele.state_dict(), chemin)
        print(f"  Meilleur modele sauvegarde ({val_acc:.2f}%)")

# ══════════════════════════════════════════
#  SAUVEGARDE HISTORIQUE
# ══════════════════════════════════════════
with open(os.path.join(BASE_DIR, "resultats_entrainement.json"), "w") as f:
    json.dump(historique, f, indent=4)

print("\n" + "="*50)
print("      ENTRAINEMENT TERMINE")
print("="*50)
print(f"  Meilleure Val Acc : {meilleure_acc:.2f}%")
print(f"  Historique sauvegarde : resultats_entrainement.json")

print(f"\n  {'Epoch':<8} {'Train Acc':<12} {'Val Acc':<12} {'Val Loss'}")
for h in historique:
    print(f"  {h['epoch']:<8} {h['train_acc']:<12} "
          f"{h['val_acc']:<12} {h['val_loss']}")