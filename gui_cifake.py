import os
os.environ["TCL_LIBRARY"] = r"C:\Users\Ale\AppData\Local\Programs\Python\Python313\tcl\tcl8.6"
os.environ["TK_LIBRARY"]  = r"C:\Users\Ale\AppData\Local\Programs\Python\Python313\tcl\tk8.6"

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

import threading
import time
import datetime

import torch
import torch.nn as nn
from torchvision import models, transforms

# Theme
DARK_BG = "#1E1E1E"
DARKER_BG = "#121212"
TEXT_COLOR = "#E0E0E0"
HIGHLIGHT_COLOR = "#FF7D00"
ACCENT_BG = "#2A2A2A"
INFO_BG = "#252525"
BANNER_BG = "#101010"

# GUI class
class CIFAKEImageDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("REAL vs AI Image Detector (CIFAKE)")
        self.root.state("zoomed")
        self.root.configure(bg=DARK_BG)
        self.root.minsize(900, 600)

        # Stats
        self.best_model_accuracy = 97.14
        self.detection_count = 0
        self.is_processing = False

        # Paths
        self.model_path = os.path.join("runs", "cifake_resnet50", "best_model.pth")

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Classes 
        self.classes = ["FAKE", "REAL"]

        # Build + load model 
        self.model = self._build_model()
        self._load_model()

        # Transforms 
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.image_path = None
        self.display_image = None

        self._setup_layout()

    # Model
    def _build_model(self):
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        return model

    def _load_model(self):
        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", f"Model not found:\n{self.model_path}")
            self.root.destroy()
            return

        state = torch.load(self.model_path, map_location=self.device)

        clean_state = {}
        for k, v in state.items():
            clean_state[k.replace("model.", "")] = v

        self.model.load_state_dict(clean_state, strict=True)
        self.model.to(self.device)
        self.model.eval()

    # Layout
    def _setup_layout(self):
        main = tk.Frame(self.root, bg=DARK_BG)
        main.pack(fill=tk.BOTH, expand=True)

        content = tk.Frame(main, bg=DARK_BG)
        content.pack(fill=tk.BOTH, expand=True)

        self.left = tk.Frame(content, bg=DARK_BG, padx=20, pady=20)
        self.left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right = tk.Frame(content, bg=INFO_BG, width=260, padx=15, pady=20)
        self.right.pack(side=tk.RIGHT, fill=tk.Y)
        self.right.pack_propagate(False)

        banner = tk.Frame(main, bg=BANNER_BG, height=40)
        banner.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Label(
            banner,
            text="Real image or AI-generated? Let the model decide.",
            font=("Arial", 12, "italic"),
            bg=BANNER_BG,
            fg=HIGHLIGHT_COLOR
        ).pack(pady=8)

        self._left_panel()
        self._right_panel()

    def _right_panel(self):
        tk.Label(
            self.right, text="DETECTION STATS",
            font=("Arial", 14, "bold"),
            bg=INFO_BG, fg=HIGHLIGHT_COLOR
        ).pack(pady=(0, 20))

        ttk.Separator(self.right).pack(fill=tk.X, pady=5)

        def info(label, value):
            row = tk.Frame(self.right, bg=INFO_BG)
            row.pack(fill=tk.X, pady=4)
            tk.Label(row, text=label, bg=INFO_BG, fg=TEXT_COLOR, width=16, anchor="w").pack(side=tk.LEFT)
            val = tk.Label(row, text=value, bg=INFO_BG, fg=HIGHLIGHT_COLOR, font=("Arial", 10, "bold"))
            val.pack(side=tk.LEFT)
            return val

        self.device_lbl = info("Device:", str(self.device).upper())
        self.acc_lbl = info("Model Acc:", f"{self.best_model_accuracy:.2f}%")
        self.conf_lbl = info("Confidence:", "-- %")
        self.time_lbl = info("Time:", "-- ms")
        self.count_lbl = info("Detections:", "0")

        ttk.Separator(self.right).pack(fill=tk.X, pady=10)

        self.time_stamp = tk.Label(
            self.right, text="--",
            bg=INFO_BG, fg=TEXT_COLOR, wraplength=220, justify="left"
        )
        self.time_stamp.pack(anchor="w")

    def _left_panel(self):
        tk.Label(
            self.left,
            text="REAL vs AI IMAGE DETECTOR",
            font=("Arial", 22, "bold"),
            bg=DARK_BG, fg=HIGHLIGHT_COLOR
        ).pack(pady=(0, 20))

        self.img_frame = tk.Frame(self.left, width=350, height=350, bg=DARKER_BG)
        self.img_frame.pack(pady=15)
        self.img_frame.pack_propagate(False)

        self.img_lbl = tk.Label(
            self.img_frame, text="Image will appear here",
            bg=DARKER_BG, fg=TEXT_COLOR, font=("Arial", 12, "italic")
        )
        self.img_lbl.pack(expand=True, fill=tk.BOTH)

        btns = tk.Frame(self.left, bg=DARK_BG)
        btns.pack(pady=15)

        style = {
            "bg": ACCENT_BG,
            "fg": TEXT_COLOR,
            "activebackground": HIGHLIGHT_COLOR,
            "font": ("Arial", 11),
            "width": 15,
            "bd": 0
        }

        tk.Button(btns, text="Upload Image", command=self.upload_image, **style).pack(side=tk.LEFT, padx=10)
        self.detect_btn = tk.Button(btns, text="Predict", command=self.start_detection, **style)
        self.detect_btn.pack(side=tk.LEFT, padx=10)

        self.result = tk.Label(
            self.left, text="",
            font=("Arial", 18, "bold"),
            bg=ACCENT_BG, fg=HIGHLIGHT_COLOR, pady=15
        )
        self.result.pack(fill=tk.X)
        self.result.pack_forget()

    # Actions
    def upload_image(self):
        self.result.pack_forget()
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if not path:
            return

        self.image_path = path
        img = Image.open(path).convert("RGB")
        img.thumbnail((350, 350))
        self.display_image = ImageTk.PhotoImage(img)
        self.img_lbl.config(image=self.display_image, text="")

    def start_detection(self):
        if not self.image_path or self.is_processing:
            return

        self.is_processing = True
        self.detect_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.detect).start()

    def detect(self):
        start = time.time()
        try:
            img = Image.open(self.image_path).convert("RGB")
            x = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[0]
                pred = torch.argmax(probs).item()
                conf = probs[pred].item() * 100

            elapsed = (time.time() - start) * 1000
            self.detection_count += 1
            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            self.root.after(0, lambda: self.update_ui(pred, conf, elapsed, stamp))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, self.reset_ui)

    def update_ui(self, pred, conf, elapsed, stamp):
        label = self.classes[pred]
        self.result.config(
            text="REAL IMAGE" if label == "REAL" else "AI-GENERATED (FAKE)",
            fg=HIGHLIGHT_COLOR if label == "REAL" else TEXT_COLOR
        )
        self.result.pack(fill=tk.X)

        self.conf_lbl.config(text=f"{conf:.2f}%")
        self.time_lbl.config(text=f"{elapsed:.2f} ms")
        self.count_lbl.config(text=str(self.detection_count))
        self.time_stamp.config(text=stamp)

    def reset_ui(self):
        self.detect_btn.config(state=tk.NORMAL)
        self.is_processing = False


# Run
if __name__ == "__main__":
    root = tk.Tk()
    app = CIFAKEImageDetectorGUI(root)
    root.mainloop()
