import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
import random

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

# Load test data
dataset = ImageFolder(root=str(DATA_DIR), transform=tfms)
class_to_idx = dataset.class_to_idx  # {'FAKE': 0, 'REAL': 1}
idx_to_class = {v: k for k, v in class_to_idx.items()}
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Model definition
class AIDetector(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(AIDetector, self).__init__()
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

# Load weights
model = AIDetector(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Select 10 random images from test set
indices = random.sample(range(len(dataset)), 10)
images, labels = zip(*[dataset[i] for i in indices])
paths = [dataset.samples[i][0] for i in indices]

# Predictions
with torch.no_grad():
    imgs_batch = torch.stack(images).to(DEVICE)
    outputs = model(imgs_batch)
    preds = outputs.argmax(1).cpu().numpy()

# Final figure  
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i, ax in enumerate(axes):
    img = images[i].permute(1, 2, 0).numpy()
    img = np.clip((img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)

    # labels coming from ImageFolder are plain ints; handle both int and tensor cases
    true_idx = labels[i].item() if hasattr(labels[i], "item") else int(labels[i])
    pred_idx = int(preds[i])
    true_label = idx_to_class[true_idx]
    pred_label = idx_to_class[pred_idx]
    color = "green" if true_label == pred_label else "red"

    ax.imshow(img)
    ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=color, fontsize=10)
    ax.axis("off")

plt.suptitle("ResNet-50 Model – Accuracy 97.14% – CIFAKE Set (AI vs Real)",
             fontsize=14, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save image
out_path = Path("runs/cifake_resnet50/sample_predictions.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Figure saved at: {out_path.resolve()}")
