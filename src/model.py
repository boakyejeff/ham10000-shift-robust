"""Frozen pretrained CNN feature extractor + linear probe.

Design choice: HAM10000-sample (~300-500 images) is far too small to train a
CNN from scratch, and full fine-tuning on 2 CPU cores is slow. A frozen
torchvision backbone (pretrained ImageNet weights) as a feature extractor plus
a class-weighted logistic regression linear probe is the standard, cheap,
CPU-feasible option and keeps the experiment about the *evaluation protocol*
(source-disjoint vs pooled), not about squeezing the last point of accuracy.
"""

from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import PROCESSED_DIR, CLASS_TO_IDX

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_extractor(arch: str = "resnet18", img_size: int = 192):
    weights = torchvision.models.get_model_weights(arch).DEFAULT
    model = torchvision.models.get_model(arch, weights=weights)
    model.fc = torch.nn.Identity()  # strip classifier -> 512-d features
    model.eval().to(DEVICE)
    t = weights.transforms()  # ImageClassification preset; carries mean/std
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(t.mean, t.std),
    ])
    return model, preprocess


def featurize(df, extractor, preprocess, img_dir: Path = None,
              batch: int = 32) -> np.ndarray:
    img_dir = Path(img_dir) if img_dir else PROCESSED_DIR / "images"
    feats = []
    with torch.no_grad():
        paths = [img_dir / f"{i}.jpg" for i in df["image_id"]]
        for s in range(0, len(paths), batch):
            tensors = []
            for p in paths[s:s + batch]:
                with Image.open(p) as im:
                    tensors.append(preprocess(im.convert("RGB")))
            x = torch.stack(tensors).to(DEVICE)
            feats.append(extractor(x).cpu().numpy())
    return np.concatenate(feats)


def train_probe(X_train, y_train, X_val, y_val, seed: int = 0):
    scaler = StandardScaler().fit(X_train)
    best, best_acc = None, -1.0
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, class_weight="balanced",
                                 max_iter=2000, random_state=seed)
        clf.fit(scaler.transform(X_train), y_train)
        acc = clf.score(scaler.transform(X_val), y_val)
        if acc > best_acc:
            best, best_acc = clf, acc
    return best, scaler


def predict_proba(clf, scaler, X):
    return clf.predict_proba(scaler.transform(X))


def encode(df) -> np.ndarray:
    return df["dx"].map(CLASS_TO_IDX).to_numpy()
