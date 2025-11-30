# utils/dataset_loader.py  — clean, robust loader (no circular imports)
import os, random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile, ImageEnhance

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")

def is_image(fname: str) -> bool:
    low = fname.lower()
    if low in ("thumbs.db",):
        return False
    return low.endswith(IMG_EXTS)

def augment_rgb(img_pil: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.3:
        img_pil = img_pil.rotate(random.uniform(-10, 10), resample=Image.BILINEAR, fillcolor=(0,0,0))
    if random.random() < 0.5:
        img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.5:
        img_pil = ImageEnhance.Contrast(img_pil).enhance(random.uniform(0.85, 1.15))
    return img_pil

def to_chw_normalized(img_pil: Image.Image, size=(224,224)) -> np.ndarray:
    im = img_pil.convert("RGB").resize(size)
    arr = (np.array(im).astype(np.float32) / 255.0).transpose(2, 0, 1)  # CHW
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)
    arr = (arr - mean) / std
    return arr

def safe_open_image(path: str, size=(224,224)) -> np.ndarray:
    try:
        with Image.open(path) as im:
            return to_chw_normalized(im, size=size)
    except Exception:
        h, w = size[1], size[0]
        blank = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
        return to_chw_normalized(blank, size=size)

def sample_indices(num_frames: int, seq_len: int) -> list:
    if num_frames <= 0:
        return [0]*seq_len
    if num_frames >= seq_len:
        return np.linspace(0, num_frames-1, seq_len, dtype=int).tolist()
    else:
        idxs = list(range(num_frames))
        while len(idxs) < seq_len:
            idxs.append(num_frames-1)
        return idxs

class CartoonSeqDataset(Dataset):
    def __init__(self, root, seq_len=8, size=(224,224)):
        self.root = root
        self.seq_len = seq_len
        self.size = size
        self.samples = []   # list[(sample_dir, label_idx)]
        self.class_to_idx = {}

        if not os.path.isdir(root):
            return

        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes.sort()
        self.class_to_idx = {c:i for i,c in enumerate(classes)}

        for c in classes:
            cpath = os.path.join(root, c)
            for s in sorted(os.listdir(cpath)):
                sp = os.path.join(cpath, s)
                if os.path.isdir(sp):
                    frames = [f for f in os.listdir(sp) if is_image(f)]
                    if len(frames) >= 1:
                        self.samples.append((sp, self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sp, label = self.samples[idx]
        files = sorted([f for f in os.listdir(sp) if is_image(f)])
        n = len(files)
        idxs = sample_indices(n, self.seq_len)

        seq = []
        is_train_split = "train" in self.root.replace("\\", "/").lower()
        for i in idxs:
            if n > 0:
                fname = files[min(i, n-1)]
                path = os.path.join(sp, fname)
                try:
                    im = Image.open(path).convert("RGB").resize(self.size)
                except Exception:
                    im = Image.fromarray(np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8))
            else:
                im = Image.fromarray(np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8))

            if is_train_split:
                im = augment_rgb(im)

            arr = (np.array(im).astype(np.float32)/255.).transpose(2,0,1)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)
            arr = (arr - mean) / std

            seq.append(arr)

        x = torch.from_numpy(np.stack(seq, axis=0)).to(torch.float32)  # (T,C,H,W)
        y = torch.tensor(label, dtype=torch.long)
        return x, y
