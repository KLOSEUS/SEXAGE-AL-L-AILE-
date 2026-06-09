import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc)
import os
from model import creer_modele

# ══════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VAL_DIR    = os.path.join(BASE_DIR, "data", "val")
MODEL_PATH = os.path.join(BASE_DIR, "models", "meilleur_modele.pth")
RESULTS    = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS, exist_ok=True)
DEVICE     = torch.device("cpu")
CLASSES    = ["Femelle", "Male"]
SEUIL      = 0.70

# ══════════════════════════════════════════
#  CHARGEMENT
# ══════════════════════════════════════════
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
val_loader  = DataLoader(val_dataset, batch_size=16,
                         shuffle=False, num_workers=0)

modele = creer_modele()
modele.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
modele.eval()
print("OK Modele charge !")
print(f"Classes : {val_dataset.classes}")
print(f"Images  : {len(val_dataset)}")

# ══════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════
all_labels    = []
all_preds     = []
all_probs     = []
all_confs     = []
all_incertains = 0

with torch.no_grad():
    for images, labels in val_loader:
        sorties     = modele(images)
        probs       = F.softmax(sorties, dim=1)
        confs, preds = torch.max(probs, dim=1)

        for i in range(len(labels)):
            conf  = confs[i].item()
            pred  = preds[i].item()
            label = labels[i].item()

            all_labels.append(label)
            all_probs.append(probs[i][1].item())
            all_confs.append(conf)

            if conf < SEUIL:
                all_preds.append(-1)
                all_incertains += 1
            else:
                all_preds.append(pred)

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)
all_confs  = np.array(all_confs)

# Filtrer les incertains pour les metriques
mask       = all_preds != -1
y_true     = all_labels[mask]
y_pred     = all_preds[mask]

# ══════════════════════════════════════════
#  METRIQUES
# ══════════════════════════════════════════
print("\n" + "="*55)
print("         RAPPORT D'EVALUATION")
print("="*55)
print(f"Total images       : {len(val_dataset)}")
print(f"Images certaines   : {mask.sum()} (seuil > {SEUIL*100:.0f}%)")
print(f"Images incertaines : {all_incertains}")
print(f"\n{classification_report(y_true, y_pred, target_names=CLASSES)}")

acc = (y_true == y_pred).mean() * 100
print(f"Precision globale  : {acc:.2f}%")

# ══════════════════════════════════════════
#  STYLE GRAPHIQUES
# ══════════════════════════════════════════
plt.style.use("dark_background")
COLORS = {
    "male"    : "#60A5FA",
    "femelle" : "#F472B6",
    "correct" : "#22C55E",
    "erreur"  : "#EF4444",
    "neutre"  : "#A78BFA"
}

# ══════════════════════════════════════════
#  1. MATRICE DE CONFUSION
# ══════════════════════════════════════════
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, linecolor="gray",
            annot_kws={"size": 18, "weight": "bold"}, ax=ax)
ax.set_xlabel("Predit",  fontsize=13, labelpad=10)
ax.set_ylabel("Reel",    fontsize=13, labelpad=10)
ax.set_title("Matrice de Confusion", fontsize=15, fontweight="bold", pad=15)

vf = cm[0][0]; fm = cm[0][1]
ff = cm[1][0]; vm = cm[1][1]
precision_m = vm/(vm+fm) if (vm+fm) > 0 else 0
rappel_m    = vm/(vm+ff) if (vm+ff) > 0 else 0
precision_f = vf/(vf+ff) if (vf+ff) > 0 else 0
rappel_f    = vf/(vf+fm) if (vf+fm) > 0 else 0

legende = (f"VF={vf}  FM={fm}  FF={ff}  VM={vm}\n"
           f"Prec. Male={precision_m*100:.1f}%  "
           f"Rappel Male={rappel_m*100:.1f}%\n"
           f"Prec. Femelle={precision_f*100:.1f}%  "
           f"Rappel Femelle={rappel_f*100:.1f}%")
fig.text(0.5, -0.05, legende, ha="center", fontsize=10,
         color="white", style="italic")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "1_matrice_confusion.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("\nSauvegarde : 1_matrice_confusion.png")

# ══════════════════════════════════════════
#  2. DISTRIBUTION DES CONFIANCES
# ══════════════════════════════════════════
conf_femelles = all_confs[all_labels == 0]
conf_males    = all_confs[all_labels == 1]

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(conf_males,    bins=20, alpha=0.7, color=COLORS["male"],
        label=f"Males ({len(conf_males)})",    edgecolor="white", lw=0.5)
ax.hist(conf_femelles, bins=20, alpha=0.7, color=COLORS["femelle"],
        label=f"Femelles ({len(conf_femelles)})", edgecolor="white", lw=0.5)
ax.axvline(SEUIL, color="yellow", linestyle="--", lw=2,
           label=f"Seuil {SEUIL*100:.0f}%")
ax.set_xlabel("Confiance du modele", fontsize=12)
ax.set_ylabel("Nombre d images",     fontsize=12)
ax.set_title("Distribution des Confiances par Sexe",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.set_xlim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "2_distribution_confiances.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Sauvegarde : 2_distribution_confiances.png")

# ══════════════════════════════════════════
#  3. REPARTITION MALE / FEMELLE
# ══════════════════════════════════════════
nb_femelles = (all_labels == 0).sum()
nb_males    = (all_labels == 1).sum()
pred_femelles = (all_preds[mask] == 0).sum()
pred_males    = (all_preds[mask] == 1).sum()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].bar(["Males", "Femelles"],
            [nb_males, nb_femelles],
            color=[COLORS["male"], COLORS["femelle"]],
            edgecolor="white", lw=0.5)
axes[0].set_title("Repartition Reelle", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Nombre d images")
for i, v in enumerate([nb_males, nb_femelles]):
    axes[0].text(i, v + 0.5, str(v), ha="center",
                 fontweight="bold", fontsize=13)

axes[1].bar(["Males", "Femelles"],
            [pred_males, pred_femelles],
            color=[COLORS["male"], COLORS["femelle"]],
            edgecolor="white", lw=0.5, alpha=0.8)
axes[1].set_title("Repartition Predite", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Nombre d images")
for i, v in enumerate([pred_males, pred_femelles]):
    axes[1].text(i, v + 0.5, str(v), ha="center",
                 fontweight="bold", fontsize=13)

plt.suptitle("Repartition Males / Femelles",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "3_repartition_sexes.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Sauvegarde : 3_repartition_sexes.png")

# ══════════════════════════════════════════
#  4. COURBE ROC
# ══════════════════════════════════════════
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc     = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color=COLORS["neutre"], lw=2,
        label=f"Courbe ROC (AUC = {roc_auc:.3f})")
ax.plot([0,1], [0,1], color="gray", linestyle="--", lw=1)
ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS["neutre"])
ax.set_xlabel("Taux Faux Positifs (FPR)", fontsize=12)
ax.set_ylabel("Taux Vrais Positifs (TPR)", fontsize=12)
ax.set_title("Courbe ROC", fontsize=14, fontweight="bold")
ax.legend(fontsize=12)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "4_courbe_roc.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Sauvegarde : 4_courbe_roc.png")

# ══════════════════════════════════════════
#  5. PRECISION PAR CLASSE
# ══════════════════════════════════════════
metriques = {
    "Prec. Male"    : precision_m * 100,
    "Rappel Male"   : rappel_m    * 100,
    "Prec. Femelle" : precision_f * 100,
    "Rappel Femelle": rappel_f    * 100,
    "Acc. Globale"  : acc
}
couleurs = [COLORS["male"], COLORS["male"],
            COLORS["femelle"], COLORS["femelle"], COLORS["correct"]]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(list(metriques.keys()),
               list(metriques.values()),
               color=couleurs, edgecolor="white", lw=0.5)
ax.axvline(90, color="yellow", linestyle="--", lw=1.5,
           label="Objectif 90%")
ax.set_xlim(0, 105)
ax.set_xlabel("Pourcentage (%)", fontsize=12)
ax.set_title("Metriques par Classe", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
for bar, val in zip(bars, metriques.values()):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontweight="bold", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "5_metriques_classes.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Sauvegarde : 5_metriques_classes.png")

# ══════════════════════════════════════════
#  RESUME FINAL
# ══════════════════════════════════════════
print("\n" + "="*55)
print("         RESUME FINAL")
print("="*55)
print(f"  Precision globale    : {acc:.2f}%")
print(f"  AUC-ROC              : {roc_auc:.3f}")
print(f"  Precision Male       : {precision_m*100:.2f}%")
print(f"  Rappel Male          : {rappel_m*100:.2f}%")
print(f"  Precision Femelle    : {precision_f*100:.2f}%")
print(f"  Rappel Femelle       : {rappel_f*100:.2f}%")
print(f"  Images incertaines   : {all_incertains}")
print(f"\n  Graphiques sauvegardes dans : results/")
print("  1_matrice_confusion.png")
print("  2_distribution_confiances.png")
print("  3_repartition_sexes.png")
print("  4_courbe_roc.png")
print("  5_metriques_classes.png")
print("\nEvaluation terminee !")