import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

# Configuration
MODEL_PATH = Path("runs/cifake_resnet50/best_model.pth")
DATA_DIR = Path("data/CIFAKE/test")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# Load Data
dataset = datasets.ImageFolder(root=str(DATA_DIR), transform=tfms)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

# Model Definition
class AIDetector(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load model
model = AIDetector(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Predictions
all_preds = []
all_labels = []
all_paths = [s[0] for s in dataset.samples]

with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Metrics
acc = accuracy_score(all_labels, all_preds)
print(f"\nFinal Test Accuracy: {acc * 100:.2f}%\n")

print("\nCLASSIFICATION REPORT:")
print(classification_report(all_labels, all_preds, target_names=list(idx_to_class.values())))

cm = confusion_matrix(all_labels, all_preds)

# Save Confusion Matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["FAKE", "REAL"])
plt.yticks([0, 1], ["FAKE", "REAL"])
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.tight_layout()

out_cm = Path("runs/cifake_resnet50/confusion_matrix.png")
plt.savefig(out_cm, dpi=150)
print(f"Confusion matrix saved at: {out_cm.resolve()}")

# Save CSV
df = pd.DataFrame({
    "image": all_paths,
    "true": [idx_to_class[i] for i in all_labels],
    "pred": [idx_to_class[i] for i in all_preds],
})
csv_path = Path("runs/cifake_resnet50/test_predictions.csv")
df.to_csv(csv_path, index=False)
print(f"Predictions saved at: {csv_path.resolve()}")

# Save Correct/Wrong Lists
correct = df[df["true"] == df["pred"]]
wrong = df[df["true"] != df["pred"]]

correct.to_csv("runs/cifake_resnet50/correct_predictions.csv", index=False)
wrong.to_csv("runs/cifake_resnet50/wrong_predictions.csv", index=False)

print("\nSaved correct and wrong prediction CSV files!")
