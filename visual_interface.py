import tkinter as tk
from tkinter import filedialog, messagebox
import torch, json
import numpy as np
import imageio
from skimage.transform import resize as resize_image
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn.functional as F
import torchvision.transforms as T

from single_image import plot_attention, beam_search_captioning

# Set up device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CaptionGUI:
    def __init__(self, root):
        self.root = root
        root.title("Image Captioning")

        # StringVars / BooleanVar
        self.image_path = tk.StringVar()
        self.checkpoint_path = tk.StringVar()
        self.map_path = tk.StringVar()
        self.beam_size = tk.StringVar(value="5")
        self.smooth = tk.BooleanVar(value=True)

        # Row 0: Image
        tk.Label(root, text="Image:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(root, textvariable=self.image_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self._browse_image).grid(row=0, column=2, padx=5)

        # Row 1: Checkpoint
        tk.Label(root, text="Checkpoint:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(root, textvariable=self.checkpoint_path, width=50).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self._browse_checkpoint).grid(row=1, column=2, padx=5)

        # Row 2: WordMap
        tk.Label(root, text="WordMap JSON:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(root, textvariable=self.map_path, width=50).grid(row=2, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self._browse_map).grid(row=2, column=2, padx=5)

        # Row 3: Beam size & Smooth checkbox
        tk.Label(root, text="Beam size:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(root, textvariable=self.beam_size, width=5).grid(row=3, column=1, sticky="w", padx=5, pady=5)
        tk.Checkbutton(root,
                       text="Smooth attention",
                       variable=self.smooth,
                       onvalue=True,
                       offvalue=False
                       ).grid(row=3, column=2, padx=5, pady=5)

        # Row 4: Generate button
        tk.Button(root,
                  text="Generate Caption",
                  command=self._generate_caption,
                  width=20
                  ).grid(row=4, column=1, pady=10)

    def _browse_image(self):
        p = filedialog.askopenfilename(title="Select an image",
                                       filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"),
                                                  ("All files", "*.*")])
        if p:
            self.image_path.set(p)

    def _browse_checkpoint(self):
        p = filedialog.askopenfilename(title="Select checkpoint (.pth.tar)",
                                       filetypes=[("PyTorch checkpoint", "*.pth.tar"),
                                                  ("All files", "*.*")])
        if p:
            self.checkpoint_path.set(p)

    def _browse_map(self):
        p = filedialog.askopenfilename(title="Select WORDMAP JSON",
                                       filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if p:
            self.map_path.set(p)

    def _generate_caption(self):
        img = self.image_path.get()
        ckpt = self.checkpoint_path.get()
        mp = self.map_path.get()
        try:
            bs = int(self.beam_size.get())
        except ValueError:
            messagebox.showerror("Invalid beam size", "Please enter a valid integer for beam size.")
            return
        sm = self.smooth.get()

        # Validate inputs
        if not all([img, ckpt, mp]):
            messagebox.showerror("Missing inputs", "You must select image, checkpoint and map.")
            return

        try:
            # Load models
            checkpoint_dict = torch.load(ckpt, map_location=DEVICE, weights_only=False)
            encoder = checkpoint_dict['encoder'].to(DEVICE).eval()
            decoder = checkpoint_dict['decoder'].to(DEVICE).eval()

            # Load vocabulary map
            with open(mp, 'r') as f:
                vocab_map = json.load(f)
            rev_map = {v: k for k, v in vocab_map.items()}

            # Generate
            seq_indices, alpha_maps = beam_search_captioning(
                encoder, decoder, img, vocab_map, beam_sz=bs
            )
            alpha_arr = torch.FloatTensor(alpha_maps).detach().cpu().numpy()

            # Plot result
            plot_attention(img, seq_indices, alpha_arr, rev_map, smooth=sm)

        except Exception as e:
            messagebox.showerror("Error during captioning", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = CaptionGUI(root)
    root.mainloop()
