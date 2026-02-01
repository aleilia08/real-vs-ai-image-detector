import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd # type: ignore
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # type: ignore
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm # type: ignore


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# Configuration
DATA_ROOT = Path("data/CIFAKE")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR  = DATA_ROOT / "test"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("runs/cifake_resnet50")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Transforms
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# Datasets (ImageFolder) and DataLoaders
full_train = datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_tfms)
class_to_idx = full_train.class_to_idx  # {'FAKE':0, 'REAL':1}
idx_to_class = {v: k for k, v in class_to_idx.items()}
print("Class mapping:", class_to_idx)

val_ratio = 0.2
val_len = int(len(full_train) * val_ratio)
train_len = len(full_train) - val_len
train_ds, val_ds = random_split(full_train, [train_len, val_len],
                                generator=torch.Generator().manual_seed(SEED))
val_ds.dataset.transform = val_tfms

test_ds = datasets.ImageFolder(root=str(TEST_DIR), transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Model: ResNet-50 + head binary classifier
class AIDetector(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.model(x)

model = AIDetector(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
best_val_acc = 0.0
best_path = OUT_DIR / "best_model.pth"

# Train / Validate function
def run_epoch(dataloader, model, train=True):
    model.train() if train else model.eval()
    losses, preds_all, targs_all = [], [], []
    for images, labels in tqdm(dataloader, disable=False):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        preds = outputs.argmax(1).detach().cpu().numpy()
        targs = labels.detach().cpu().numpy()
        preds_all.extend(preds)
        targs_all.extend(targs)
    acc = accuracy_score(targs_all, preds_all)
    return np.mean(losses), acc

# Main training loop
if __name__ == "__main__":
    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS} — device: {DEVICE}")
        t0 = time.time()
        train_loss, train_acc = run_epoch(train_loader, model, train=True)
        val_loss,   val_acc   = run_epoch(val_loader,   model, train=False)
        scheduler.step()
        dt = time.time() - t0
        print(f"Train  | loss={train_loss:.4f} acc={train_acc*100:.2f}%")
        print(f"Val    | loss={val_loss:.4f} acc={val_acc*100:.2f}%  (epoch time {dt:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"✅ New best model saved: {best_path} (val acc {val_acc*100:.2f}%)")

    # Evaluate on test set
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    model.eval()
    all_preds, all_targs = [], []
    for images, labels in tqdm(test_loader, disable=False):
        images = images.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_targs.extend(labels.numpy())

    test_acc = accuracy_score(all_targs, all_preds)
    cm = confusion_matrix(all_targs, all_preds)
    print(f"\nTest accuracy: {test_acc*100:.2f}%")
    print("Confusion matrix (rows=true, cols=pred):\n", cm)
    print("\nClassification report:\n", classification_report(all_targs, all_preds, target_names=[idx_to_class[0], idx_to_class[1]]))

    # Save CSV with predictions
    test_paths = [p for p, _ in test_loader.dataset.samples]
    pred_labels = [idx_to_class[p] for p in all_preds]

    df = pd.DataFrame({
        "image_path": test_paths,
        "pred_class": pred_labels
    })
    csv_path = OUT_DIR / "test_predictions.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n📄 Predictions saved to: {csv_path}")
