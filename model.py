import torch
import torch.nn as nn
from torchvision import models

def creer_modele():
    model = models.resnet18(weights=None)
    nb_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(nb_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )

    return model

if __name__ == "__main__":
    modele = creer_modele()
    modele.eval()
    total     = sum(p.numel() for p in modele.parameters())
    entraines = sum(p.numel() for p in modele.parameters()
                    if p.requires_grad)
    print("OK Modele ResNet18 ameliore !")
    print(f"Parametres totaux      : {total:,}")
    print(f"Parametres entrainables: {entraines:,}")
    test   = torch.randn(1, 3, 224, 224)
    sortie = modele(test)
    print(f"Test sortie            : {sortie.shape}")
    print(f"Classes                : [Male, Femelle]")