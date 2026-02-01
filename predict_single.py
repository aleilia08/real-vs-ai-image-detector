import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

import os
import threading
import time
import datetime

import torch
import torch.nn as nn
from torchvision import models, transforms

# Theme colors
DARK_BG = "#1E1E1E"
DARKER_BG = "#121212"
TEXT_COLOR = "#E0E0E0"
HIGHLIGHT_COLOR = "#FF7D00"  
ACCENT_BG = "#2A2A2A"
INFO_BG = "#252525"
BANNER_BG = "#101010"


class CIFAKEDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("REAL vs AI Image Detector (CIFAKE)")
        self.root.state('zoomed')
        self.root.configure(bg=DARK_BG)
        self.root.minsize(900, 600)

        # Stats
        self.detection_time = 0.0
        self.confidence_score = 0.0
        self.best_model_accuracy = 97.14  
        self.detection_count = 0
        self.is_processing = False

        # Paths
        self.model_path = os.path.join("runs", "cifake_resnet50", "best_model.pth")
        self.acc_path = os.path.join("runs", "cifake_resnet50", "best_model_accuracy.txt")

        if os.path.exists(self.acc_path):
            try:
                with open(self.acc_path, "r") as f:
                    self.best_model_accuracy = float(f.read().strip())
            except:
                pass

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Classes 
        self.classes = ["FAKE", "REAL"]

        # Build + load model 
        self.model = self._build_model()
        self._load_model()

        # Transformations 
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # UI state
        self.image_path = None
        self.display_image = None

        # Layout
        self.setup_layout()

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
            messagebox.showerror("Error", f"Model file not found:\n{self.model_path}")
            return

        state = torch.load(self.model_path, map_location=self.device)

        # Remove "model." prefix 
        clean_state = {}
        for k, v in state.items():
            new_key = k.replace("model.", "")
            clean_state[new_key] = v

        try:
            self.model.load_state_dict(clean_state, strict=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model weights:\n{e}")
            return

        self.model.to(self.device)
        self.model.eval()

    # Layout
    def setup_layout(self):
        self.main_container = tk.Frame(self.root, bg=DARK_BG)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.content_frame = tk.Frame(self.main_container, bg=DARK_BG)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        self.left_panel = tk.Frame(self.content_frame, bg=DARK_BG, padx=20, pady=20)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_panel = tk.Frame(self.content_frame, bg=INFO_BG, width=250, padx=15, pady=20)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_panel.pack_propagate(False)

        self.banner_frame = tk.Frame(self.main_container, bg=BANNER_BG, height=40)
        self.banner_frame.pack(side=tk.BOTTOM, fill=tk.X)

        tagline = tk.Label(
            self.banner_frame,
            text="Is it a real photo or AI-generated? That is the question.",
            font=("Arial", 12, "italic"),
            bg=BANNER_BG,
            fg=HIGHLIGHT_COLOR,
            pady=10
        )
        tagline.pack()

        self.create_left_panel_widgets()
        self.create_right_panel_widgets()

    def create_right_panel_widgets(self):
        info_title = tk.Label(
            self.right_panel,
            text="DETECTION STATS",
            font=("Arial", 14, "bold"),
            bg=INFO_BG,
            fg=HIGHLIGHT_COLOR
        )
        info_title.pack(pady=(0, 20))

        separator = ttk.Separator(self.right_panel, orient='horizontal')
        separator.pack(fill=tk.X, pady=5)

        model_section = tk.Frame(self.right_panel, bg=INFO_BG, pady=10)
        model_section.pack(fill=tk.X)

        model_title = tk.Label(
            model_section,
            text="Model Information",
            font=("Arial", 12, "bold"),
            bg=INFO_BG,
            fg=TEXT_COLOR
        )
        model_title.pack(anchor=tk.W)

        device_frame = tk.Frame(model_section, bg=INFO_BG, pady=5)
        device_frame.pack(fill=tk.X)

        device_label = tk.Label(
            device_frame,
            text="Device:",
            font=("Arial", 10),
            bg=INFO_BG,
            fg=TEXT_COLOR,
            width=15,
            anchor=tk.W
        )
        device_label.pack(side=tk.LEFT)

        self.device_value = tk.Label(
            device_frame,
            text=str(self.device).upper(),
            font=("Arial", 10, "bold"),
            bg=INFO_BG,
            fg=HIGHLIGHT_COLOR
        )
        self.device_value.pack(side=tk.LEFT)

        accuracy_frame = tk.Frame(model_section, bg=INFO_BG, pady=5)
        accuracy_frame.pack(fill=tk.X)

        accuracy_label = tk.Label(
            accuracy_frame,
            text="Model Accuracy:",
            font=("Arial", 10),
            bg=INFO_BG,
            fg=TEXT_COLOR,
            width=15,
            anchor=tk.W
        )
        accuracy_label.pack(side=tk.LEFT)

        self.accuracy_value = tk.Label(
            accuracy_frame,
            text=f"{self.best_model_accuracy:.2f}%",
            font=("Arial", 10, "bold"),
            bg=INFO_BG,
            fg=HIGHLIGHT_COLOR
        )
        self.accuracy_value.pack(side=tk.LEFT)

        separator2 = ttk.Separator(self.right_panel, orient='horizontal')
        separator2.pack(fill=tk.X, pady=10)

        detection_section = tk.Frame(self.right_panel, bg=INFO_BG, pady=10)
        detection_section.pack(fill=tk.X)

        detection_title = tk.Label(
            detection_section,
            text="Current Detection",
            font=("Arial", 12, "bold"),
            bg=INFO_BG,
            fg=TEXT_COLOR
        )
        detection_title.pack(anchor=tk.W)

        confidence_frame = tk.Frame(detection_section, bg=INFO_BG, pady=5)
        confidence_frame.pack(fill=tk.X)

        confidence_label = tk.Label(
            confidence_frame,
            text="Confidence:",
            font=("Arial", 10),
            bg=INFO_BG,
            fg=TEXT_COLOR,
            width=15,
            anchor=tk.W
        )
        confidence_label.pack(side=tk.LEFT)

        self.confidence_value = tk.Label(
            confidence_frame,
            text="-- %",
            font=("Arial", 10, "bold"),
            bg=INFO_BG,
            fg=HIGHLIGHT_COLOR
        )
        self.confidence_value.pack(side=tk.LEFT)

        time_frame = tk.Frame(detection_section, bg=INFO_BG, pady=5)
        time_frame.pack(fill=tk.X)

        time_label = tk.Label(
            time_frame,
            text="Process Time:",
            font=("Arial", 10),
            bg=INFO_BG,
            fg=TEXT_COLOR,
            width=15,
            anchor=tk.W
        )
        time_label.pack(side=tk.LEFT)

        self.time_value = tk.Label(
            time_frame,
            text="-- ms",
            font=("Arial", 10, "bold"),
            bg=INFO_BG,
            fg=HIGHLIGHT_COLOR
        )
        self.time_value.pack(side=tk.LEFT)

        count_frame = tk.Frame(detection_section, bg=INFO_BG, pady=5)
        count_frame.pack(fill=tk.X)

        count_label = tk.Label(
            count_frame,
            text="Total Detections:",
            font=("Arial", 10),
            bg=INFO_BG,
            fg=TEXT_COLOR,
            width=15,
            anchor=tk.W
        )
        count_label.pack(side=tk.LEFT)

        self.count_value = tk.Label(
            count_frame,
            text="0",
            font=("Arial", 10, "bold"),
            bg=INFO_BG,
            fg=HIGHLIGHT_COLOR
        )
        self.count_value.pack(side=tk.LEFT)

        separator3 = ttk.Separator(self.right_panel, orient='horizontal')
        separator3.pack(fill=tk.X, pady=10)

        timestamp_section = tk.Frame(self.right_panel, bg=INFO_BG, pady=10)
        timestamp_section.pack(fill=tk.X)

        timestamp_title = tk.Label(
            timestamp_section,
            text="Last Detection",
            font=("Arial", 12, "bold"),
            bg=INFO_BG,
            fg=TEXT_COLOR
        )
        timestamp_title.pack(anchor=tk.W)

        self.timestamp_value = tk.Label(
            timestamp_section,
            text="--",
            font=("Arial", 10),
            bg=INFO_BG,
            fg=TEXT_COLOR,
            wraplength=220,
            justify=tk.LEFT
        )
        self.timestamp_value.pack(anchor=tk.W, pady=5)

        style = ttk.Style()
        style.configure("TSeparator", background=HIGHLIGHT_COLOR)

    def create_left_panel_widgets(self):
        title_label = tk.Label(
            self.left_panel,
            text="REAL vs AI DETECTION",
            font=("Arial", 22, "bold"),
            bg=DARK_BG,
            fg=HIGHLIGHT_COLOR
        )
        title_label.pack(pady=(0, 20))

        instructions = tk.Label(
            self.left_panel,
            text="Upload an image to classify it as REAL or AI-GENERATED (FAKE).",
            font=("Arial", 12),
            bg=DARK_BG,
            fg=TEXT_COLOR
        )
        instructions.pack(pady=(0, 20))

        self.image_container = tk.Frame(
            self.left_panel,
            bd=2,
            relief=tk.GROOVE,
            bg=DARKER_BG,
            highlightbackground=HIGHLIGHT_COLOR,
            highlightthickness=1
        )
        self.image_container.pack(pady=15)

        self.image_frame = tk.Frame(
            self.image_container,
            width=350,
            height=350,
            bg=DARKER_BG
        )
        self.image_frame.pack(padx=10, pady=10)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(
            self.image_frame,
            text="Image will appear here",
            bg=DARKER_BG,
            fg=TEXT_COLOR,
            font=("Arial", 12, "italic")
        )
        self.image_label.pack(expand=True, fill=tk.BOTH)

        button_frame = tk.Frame(self.left_panel, bg=DARK_BG)
        button_frame.pack(pady=15)

        button_style = {
            "bg": ACCENT_BG,
            "fg": TEXT_COLOR,
            "activebackground": HIGHLIGHT_COLOR,
            "activeforeground": DARKER_BG,
            "font": ("Arial", 11),
            "bd": 0,
            "padx": 15,
            "pady": 8,
            "width": 15,
            "cursor": "hand2"
        }

        upload_button = tk.Button(
            button_frame,
            text="Upload Image",
            command=self.upload_image,
            **button_style
        )
        upload_button.pack(side=tk.LEFT, padx=10)

        self.detect_button = tk.Button(
            button_frame,
            text="Predict",
            command=self.start_detection,
            **button_style
        )
        self.detect_button.pack(side=tk.LEFT, padx=10)

        self.progress_frame = tk.Frame(self.left_panel, bg=DARK_BG)
        self.progress_frame.pack(pady=(5, 15), fill=tk.X)

        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            length=350,
            mode='indeterminate'
        )
        self.progress.pack(pady=5)
        self.progress.pack_forget()

        self.processing_label = tk.Label(
            self.progress_frame,
            text="Processing image...",
            bg=DARK_BG,
            fg=HIGHLIGHT_COLOR,
            font=("Arial", 10, "italic")
        )
        self.processing_label.pack()
        self.processing_label.pack_forget()

        self.result_frame = tk.Frame(self.left_panel, bg=ACCENT_BG, bd=1, relief=tk.FLAT)
        self.result_frame.pack(pady=10, fill=tk.X)
        self.result_frame.pack_forget()

        self.result_label = tk.Label(
            self.result_frame,
            text="",
            font=("Arial", 16, "bold"),
            bg=ACCENT_BG,
            fg=HIGHLIGHT_COLOR,
            padx=20,
            pady=15
        )
        self.result_label.pack()

        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            "TProgressbar",
            thickness=10,
            troughcolor=ACCENT_BG,
            background=HIGHLIGHT_COLOR
        )

    # Actions
    def upload_image(self):
        self.hide_result()
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path).convert("RGB")

            width, height = image.size
            max_size = 350

            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            image = image.resize((new_width, new_height), Image.LANCZOS)
            self.display_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.display_image, text="")

    def start_detection(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please upload an image first!")
            return

        if self.is_processing:
            return

        self.is_processing = True
        self.detect_button.config(state=tk.DISABLED)

        self.progress.pack(pady=5)
        self.processing_label.pack()
        self.progress.start(10)

        self.time_value.config(text="-- ms")
        self.confidence_value.config(text="-- %")

        threading.Thread(target=self.detect_image, daemon=True).start()

    def detect_image(self):
        start_time = time.time()
        try:
            img = Image.open(self.image_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(img_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = int(torch.argmax(probs).item())
                confidence = float(probs[pred_idx].item()) * 100.0

            elapsed_ms = (time.time() - start_time) * 1000.0
            self.detection_time = elapsed_ms
            self.confidence_score = confidence
            self.detection_count += 1

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.root.after(0, lambda: self.update_result(pred_idx, confidence, elapsed_ms, timestamp))

        except Exception as e:
            self.root.after(0, lambda: self.show_error(str(e)))
        finally:
            self.root.after(0, self.reset_ui)

    def update_result(self, pred_idx, confidence, elapsed_time, timestamp):
        self.result_frame.pack(pady=10, fill=tk.X)

        pred_label = self.classes[pred_idx]

        if pred_label == "REAL":
            self.result_label.config(text="REAL IMAGE", fg=HIGHLIGHT_COLOR)
        else:
            self.result_label.config(text="AI-GENERATED (FAKE)", fg=TEXT_COLOR)

        self.confidence_value.config(text=f"{confidence:.2f}%")
        self.time_value.config(text=f"{elapsed_time:.2f} ms")
        self.count_value.config(text=str(self.detection_count))
        self.timestamp_value.config(text=timestamp)

    def reset_ui(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.processing_label.pack_forget()
        self.detect_button.config(state=tk.NORMAL)
        self.is_processing = False

    def hide_result(self):
        self.result_frame.pack_forget()

    def show_error(self, error_msg):
        messagebox.showerror("Error", f"An error occurred: {error_msg}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CIFAKEDetectorGUI(root)
    root.mainloop()
