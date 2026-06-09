import torch
import os
import json
from datetime import datetime
from model import creer_modele

# ══════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "meilleur_modele.pth")
DEVICE     = torch.device("cpu")

# ══════════════════════════════════════════
#  SAUVEGARDE COMPLETE
# ══════════════════════════════════════════
def sauvegarder(modele, metriques):
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    nom_fichier = f"modele_poussins_{timestamp}.pth"
    chemin      = os.path.join(MODEL_DIR, nom_fichier)

    torch.save({
        "model_state_dict" : modele.state_dict(),
        "architecture"     : "ResNet18 + Transfer Learning (layer3+layer4+fc)",
        "classes"          : ["femelle", "male"],
        "precision_val"    : metriques["precision"],
        "auc_roc"          : metriques["auc"],
        "precision_male"   : metriques["prec_male"],
        "rappel_male"      : metriques["rappel_male"],
        "precision_femelle": metriques["prec_femelle"],
        "rappel_femelle"   : metriques["rappel_femelle"],
        "nb_epochs"        : metriques["epochs"],
        "date_creation"    : timestamp,
        "taille_image"     : "224x224",
        "seuil_confiance"  : 0.70,
    }, chemin)
    print(f"OK Modele sauvegarde : {nom_fichier}")
    return chemin

# ══════════════════════════════════════════
#  SAUVEGARDE JSON
# ══════════════════════════════════════════
def sauvegarder_json(metriques):
    infos = {
        "projet"           : "Sexage Poussins par Remiges",
        "architecture"     : "ResNet18 + Transfer Learning",
        "classes"          : ["femelle", "male"],
        "dataset"          : "426 males + 526 femelles (952 total)",
        "precision_val"    : f"{metriques['precision']}%",
        "auc_roc"          : metriques["auc"],
        "precision_male"   : f"{metriques['prec_male']}%",
        "rappel_male"      : f"{metriques['rappel_male']}%",
        "precision_femelle": f"{metriques['prec_femelle']}%",
        "rappel_femelle"   : f"{metriques['rappel_femelle']}%",
        "nb_epochs"        : metriques["epochs"],
        "seuil_confiance"  : "70%",
        "taille_image"     : "224x224",
        "framework"        : "PyTorch 2.12",
        "date"             : datetime.now().strftime("%d/%m/%Y %H:%M"),
        "contrainte_temps" : "< 5ms par image (CPU)"
    }
    chemin_json = os.path.join(MODEL_DIR, "infos_modele.json")
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump(infos, f, indent=4, ensure_ascii=False)
    print("OK Infos sauvegardees : infos_modele.json")
    return infos

# ══════════════════════════════════════════
#  VERIFICATION
# ══════════════════════════════════════════
def verifier(chemin):
    print("\nVerification du modele...")
    checkpoint = torch.load(chemin, map_location=DEVICE)
    modele     = creer_modele()
    modele.load_state_dict(checkpoint["model_state_dict"])
    modele.eval()
    test   = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        sortie = modele(test)
    print(f"  Architecture  : {checkpoint['architecture']}")
    print(f"  Classes       : {checkpoint['classes']}")
    print(f"  Precision     : {checkpoint['precision_val']}")
    print(f"  AUC-ROC       : {checkpoint['auc_roc']}")
    print(f"  Seuil         : {checkpoint['seuil_confiance']*100:.0f}%")
    print(f"  Test sortie   : {sortie.shape}")
    print("OK Modele verifie et fonctionnel !")

# ══════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════
if __name__ == "__main__":
    print("="*50)
    print("    SAUVEGARDE DU MODELE POUSSINS")
    print("="*50)

    modele = creer_modele()
    modele.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    metriques = {
        "precision"    : 96.69,
        "auc"          : 0.974,
        "prec_male"    : 96.12,
        "rappel_male"  : 98.02,
        "prec_femelle" : 97.44,
        "rappel_femelle": 95.00,
        "epochs"       : 30
    }

    chemin = sauvegarder(modele, metriques)
    infos  = sauvegarder_json(metriques)
    verifier(chemin)

    print("\n" + "="*50)
    print("    RESUME DU MODELE")
    print("="*50)
    for cle, val in infos.items():
        print(f"  {cle:<22} : {val}")
    print("\nModele pret pour l'interface !")