import os
import threading
import time
import datetime

import customtkinter as ctk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk, ImageOps

import torch
import torch.nn as nn
from torchvision import models, transforms


# -----------------------------
# THEME (Neon Blue / Glassy Dark)
# -----------------------------
ctk.set_appearance_mode("dark")

BG         = "#070A12"
CARD       = "#0B1020"
CARD_2     = "#0A0F1E"
BORDER     = "#1B2A4A"
TEXT       = "#EAF2FF"
MUTED      = "#8FA3BF"

NEON_BLUE  = "#4DA3FF"
NEON_CYAN  = "#2EF2FF"

GOOD       = "#3DFFB5"  # REAL
BAD        = "#FF4D8D"  # FAKE

APP_TITLE  = "REAL vs AI Image Detector (CIFAKE)"


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


class CIFAKEImageDetectorGUI:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1200x740")
        self.root.minsize(1050, 650)
        self.root.configure(fg_color=BG)

        # Stats
        self.best_model_accuracy = 97.14
        self.detection_count = 0
        self.is_processing = False

        # Paths
        self.model_path = os.path.join("runs", "cifake_resnet50", "best_model.pth")

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Classes (keep your mapping)
        self.classes = ["FAKE", "REAL"]

        # Build + load model
        self.model = self._build_model()
        self._load_model()

        # Transforms (same as your val_tfms)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.image_path = None
        self.preview_photo = None  # keep reference for Tk image

        self._setup_layout()

        # init stats
        self._update_stats(conf=None, infer_ms=None, stamp=None)

    # -----------------------------
    # Model
    # -----------------------------
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

    # -----------------------------
    # Layout
    # -----------------------------
    def _setup_layout(self):
        # root grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_rowconfigure(0, weight=0)  # header
        self.root.grid_rowconfigure(1, weight=1)  # body
        self.root.grid_rowconfigure(2, weight=0)  # footer/log

        self._header()
        self._main_body()
        self._footer_log()

    def _header(self):
        header = ctk.CTkFrame(self.root, fg_color=BG)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=(16, 10))
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text="REAL vs AI IMAGE DETECTOR",
            font=ctk.CTkFont("Segoe UI", 26, "bold"),
            text_color=NEON_BLUE
        )
        title.grid(row=0, column=0, sticky="w")

        subtitle = ctk.CTkLabel(
            header,
            text="Upload an image and let the model decide.",
            font=ctk.CTkFont("Segoe UI", 13),
            text_color=MUTED
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(4, 0))

        # thin neon line
        line = ctk.CTkFrame(header, fg_color=BORDER, height=2, corner_radius=0)
        line.grid(row=2, column=0, sticky="ew", pady=(10, 0))

    def _main_body(self):
        # left main + right sidebar
        self.main = ctk.CTkFrame(self.root, fg_color=BG)
        self.main.grid(row=1, column=0, sticky="nsew", padx=(18, 10), pady=(0, 10))
        self.main.grid_rowconfigure(0, weight=1)
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_columnconfigure(1, weight=1)

        self.sidebar = ctk.CTkFrame(
            self.root, fg_color=CARD, corner_radius=16, border_width=1, border_color=BORDER
        )
        self.sidebar.grid(row=1, column=1, sticky="nsew", padx=(0, 18), pady=(0, 10))
        self.sidebar.grid_rowconfigure(99, weight=1)

        # left card: Image Preview
        self.preview_card = ctk.CTkFrame(
            self.main, fg_color=CARD, corner_radius=16, border_width=1, border_color=BORDER
        )
        self.preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.preview_card.grid_rowconfigure(1, weight=1)
        self.preview_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.preview_card, text="IMAGE PREVIEW",
            font=ctk.CTkFont("Segoe UI", 14, "bold"),
            text_color=MUTED
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))

        # actual preview area
        self.preview_area = ctk.CTkFrame(
            self.preview_card, fg_color=CARD_2, corner_radius=14, border_width=1, border_color=BORDER
        )
        self.preview_area.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 14))
        self.preview_area.grid_rowconfigure(0, weight=1)
        self.preview_area.grid_columnconfigure(0, weight=1)

        self.preview_label = ctk.CTkLabel(
            self.preview_area,
            text="No image uploaded",
            font=ctk.CTkFont("Segoe UI", 13),
            text_color=MUTED
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)

        # info row (optional)
        self.meta_label = ctk.CTkLabel(
            self.preview_card,
            text="Resolution: --   |   Format: --",
            font=ctk.CTkFont("Segoe UI", 12),
            text_color=MUTED
        )
        self.meta_label.grid(row=2, column=0, sticky="w", padx=16, pady=(0, 14))

        # middle/right card: Result + actions
        self.result_card = ctk.CTkFrame(
            self.main, fg_color=CARD, corner_radius=16, border_width=1, border_color=BORDER
        )
        self.result_card.grid(row=0, column=1, sticky="nsew")
        self.result_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.result_card, text="RESULT",
            font=ctk.CTkFont("Segoe UI", 14, "bold"),
            text_color=MUTED
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))

        # Big result label
        self.result_big = ctk.CTkLabel(
            self.result_card,
            text="—",
            font=ctk.CTkFont("Segoe UI", 34, "bold"),
            text_color=TEXT
        )
        self.result_big.grid(row=1, column=0, sticky="w", padx=16, pady=(12, 4))

        self.result_small = ctk.CTkLabel(
            self.result_card,
            text="Upload an image to start.",
            font=ctk.CTkFont("Segoe UI", 13),
            text_color=MUTED
        )
        self.result_small.grid(row=2, column=0, sticky="w", padx=16, pady=(0, 16))

        # confidence bar + percent
        bar_wrap = ctk.CTkFrame(self.result_card, fg_color="transparent")
        bar_wrap.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 16))
        bar_wrap.grid_columnconfigure(0, weight=1)

        self.conf_bar = ctk.CTkProgressBar(
            bar_wrap, height=12, corner_radius=999,
            fg_color="#0E1730", progress_color=NEON_BLUE
        )
        self.conf_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.conf_bar.set(0)

        self.conf_text = ctk.CTkLabel(
            bar_wrap, text="Confidence: -- %",
            font=ctk.CTkFont("Segoe UI", 12),
            text_color=MUTED
        )
        self.conf_text.grid(row=1, column=0, sticky="w")

        # buttons
        btns = ctk.CTkFrame(self.result_card, fg_color="transparent")
        btns.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 14))
        btns.grid_columnconfigure(0, weight=1)
        btns.grid_columnconfigure(1, weight=1)

        self.upload_btn = ctk.CTkButton(
            btns, text="Upload Image",
            fg_color="transparent", hover_color="#0E1730",
            border_width=1, border_color=BORDER,
            text_color=TEXT,
            height=42,
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            command=self.upload_image
        )
        self.upload_btn.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.predict_btn = ctk.CTkButton(
            btns, text="Predict",
            fg_color=NEON_BLUE,
            hover_color=NEON_CYAN,
            text_color="black",
            height=42,
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            command=self.start_detection,
            state="disabled"
        )
        self.predict_btn.grid(row=0, column=1, sticky="ew")

        # small status line
        self.status_line = ctk.CTkLabel(
            self.result_card,
            text="",
            font=ctk.CTkFont("Segoe UI", 12),
            text_color=MUTED
        )
        self.status_line.grid(row=5, column=0, sticky="w", padx=16, pady=(0, 16))

        # sidebar
        self._sidebar_panel()

    def _sidebar_panel(self):
        ctk.CTkLabel(
            self.sidebar, text="DETECTION STATS",
            font=ctk.CTkFont("Segoe UI", 15, "bold"),
            text_color=NEON_BLUE
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 10))

        sep = ctk.CTkFrame(self.sidebar, fg_color=BORDER, height=2)
        sep.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 12))

        def stat_row(r, label, value):
            row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
            row.grid(row=r, column=0, sticky="ew", padx=16, pady=6)
            row.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(row, text=label, text_color=MUTED, font=ctk.CTkFont("Segoe UI", 12)).grid(
                row=0, column=0, sticky="w"
            )
            val = ctk.CTkLabel(row, text=value, text_color=TEXT, font=ctk.CTkFont("Segoe UI", 12, "bold"))
            val.grid(row=0, column=1, sticky="e")
            return val

        self.device_lbl = stat_row(2, "Device", str(self.device).upper())
        self.acc_lbl    = stat_row(3, "Model Acc", f"{self.best_model_accuracy:.2f}%")
        self.conf_lbl   = stat_row(4, "Confidence", "-- %")
        self.time_lbl   = stat_row(5, "Time", "-- ms")
        self.count_lbl  = stat_row(6, "Detections", "0")

        sep2 = ctk.CTkFrame(self.sidebar, fg_color=BORDER, height=2)
        sep2.grid(row=7, column=0, sticky="ew", padx=16, pady=(10, 12))

        self.time_stamp = ctk.CTkLabel(
            self.sidebar,
            text="—",
            text_color=MUTED,
            font=ctk.CTkFont("Segoe UI", 12),
            justify="left"
        )
        self.time_stamp.grid(row=8, column=0, sticky="w", padx=16)

        self.metrics_btn = ctk.CTkButton(
            self.sidebar, text="Show Model Metrics",
            fg_color="transparent", hover_color="#0E1730",
            border_width=1, border_color=BORDER,
            text_color=TEXT,
            height=40,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
            command=self._show_metrics_popup
        )
        self.metrics_btn.grid(row=98, column=0, sticky="ew", padx=16, pady=(10, 16))

    def _footer_log(self):
        # bottom log card spanning both columns
        self.log_card = ctk.CTkFrame(
            self.root, fg_color=CARD, corner_radius=16, border_width=1, border_color=BORDER
        )
        self.log_card.grid(row=2, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 16))
        self.log_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.log_card, text="LOG",
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            text_color=MUTED
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 6))

        self.log_box = ctk.CTkTextbox(
            self.log_card, height=90,
            fg_color=CARD_2, text_color=TEXT,
            border_width=1, border_color=BORDER,
            corner_radius=12
        )
        self.log_box.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 14))
        self.log_box.insert("end", "• Ready.\n")
        self.log_box.configure(state="disabled")

    # -----------------------------
    # Actions
    # -----------------------------
    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if not path:
            return

        self.image_path = path

        # Load + clean preview (EXIF fix, center crop, high-quality resize)
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")

        # meta
        w, h = img.size
        fmt = os.path.splitext(path)[1].lower().replace(".", "").upper() or "IMG"
        self.meta_label.configure(text=f"Resolution: {w}×{h}   |   Format: {fmt}")

        # preview
        max_size = 420
        w_orig, h_orig = img.size
        scale = min(max_size / w_orig, max_size / h_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        prev = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.preview_photo = ctk.CTkImage(
            light_image=prev,
            dark_image=prev,
            size=(prev.width, prev.height)
        )
        self.preview_label.configure(image=self.preview_photo, text="")

        # enable predict
        self.predict_btn.configure(state="normal")
        self.status_line.configure(text="")
        self._set_result_neutral()

    def start_detection(self):
        if self.is_processing or not self.image_path:
            return

        self.is_processing = True
        self.predict_btn.configure(state="disabled")
        self.upload_btn.configure(state="disabled")
        self.status_line.configure(text="Running inference...", text_color=MUTED)

        threading.Thread(target=self.detect, daemon=True).start()

    def detect(self):
        start = time.time()
        try:
            img = ImageOps.exif_transpose(Image.open(self.image_path)).convert("RGB")
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
            self.root.after(0, self._reset_buttons)
        finally:
            self.root.after(0, lambda: setattr(self, "is_processing", False))

    def update_ui(self, pred, conf, elapsed, stamp):
        label = self.classes[pred]  # "FAKE" or "REAL"

        if label == "REAL":
            big = "✓ REAL"
            small = "Likely a genuine image."
            color = GOOD
            bar_color = GOOD
        else:
            big = "⚠ AI-GENERATED"
            small = "Likely synthetic / generated."
            color = BAD
            bar_color = BAD

        self.result_big.configure(text=big, text_color=color)
        self.result_small.configure(text=small, text_color=MUTED)

        self.conf_bar.configure(progress_color=bar_color)
        self.conf_bar.set(max(0.0, min(1.0, conf / 100.0)))
        self.conf_text.configure(text=f"Confidence: {conf:.2f} %", text_color=MUTED)

        # Update confidence label in sidebar with color coding
        self.conf_lbl.configure(text=f"{conf:.2f}%", text_color=color)

        self._update_stats(conf=conf, infer_ms=elapsed, stamp=stamp)
        self.status_line.configure(text="Inference completed.", text_color=MUTED)

        # log line
        self._append_log(f"Detected {label}  |  {conf:.2f}%  |  {elapsed:.0f} ms  |  {stamp}")

        self._reset_buttons()

    def _update_stats(self, conf, infer_ms, stamp):
        self.acc_lbl.configure(text=f"{self.best_model_accuracy:.2f}%")
        self.count_lbl.configure(text=str(self.detection_count))

        if conf is None:
            self.conf_lbl.configure(text="-- %")
        else:
            self.conf_lbl.configure(text=f"{conf:.2f}%")

        if infer_ms is None:
            self.time_lbl.configure(text="-- ms")
        else:
            self.time_lbl.configure(text=f"{infer_ms:.0f} ms")

        if stamp is None:
            self.time_stamp.configure(text="—")
        else:
            self.time_stamp.configure(text=stamp)

    def _append_log(self, line: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"• {line}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _set_result_neutral(self):
        self.result_big.configure(text="—", text_color=TEXT)
        self.result_small.configure(text="Upload an image to start.", text_color=MUTED)
        self.conf_bar.configure(progress_color=NEON_BLUE)
        self.conf_bar.set(0)
        self.conf_text.configure(text="Confidence: -- %", text_color=MUTED)
        self.conf_lbl.configure(text="-- %")
        self.time_lbl.configure(text="-- ms")

    def _reset_buttons(self):
        self.upload_btn.configure(state="normal")
        self.predict_btn.configure(state=("normal" if self.image_path else "disabled"))

    def _show_metrics_popup(self):
        msg = (
            "Model: ResNet-50 (binary classifier)\n"
            f"Test Accuracy: {self.best_model_accuracy:.2f}%\n"
            f"Device: {str(self.device).upper()}\n\n"
            "Confusion Matrix (Test set):\n"
            "FAKE → FAKE: 9698\n"
            "FAKE → REAL:  302\n"
            "REAL → FAKE:  271\n"
            "REAL → REAL: 9729\n\n"
            "Classification Report (macro avg):\n"
            "Precision: 0.97 | Recall: 0.97 | F1-score: 0.97"
        )
        messagebox.showinfo("Model Metrics", msg)


if __name__ == "__main__":
    root = ctk.CTk()
    app = CIFAKEImageDetectorGUI(root)
    root.mainloop()