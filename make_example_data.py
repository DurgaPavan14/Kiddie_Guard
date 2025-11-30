# make_example_data.py
import os, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image

# Optional: only needed if downloads are allowed
try:
    import requests
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False

# OpenCV is used to extract frames from mp4s
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "data"
CLIPS = PROJECT_ROOT / "clips"

CATEGORIES = ["violent", "non-violent", "comedic"]
URLS = {
    "violent":      "https://archive.org/download/skeletondance1929/SkeletonDance1929_512kb.mp4",
    "comedic":      "https://archive.org/download/BettyBoopDizzyDishes1930/BettyBoopDizzyDishes1930_512kb.mp4",
    "non-violent":  "https://archive.org/download/cartoons_202104/LittleDutchMill1934_512kb.mp4",
}

def ensure_dirs():
    for split in ["train", "test"]:
        for cls in CATEGORIES:
            (DATA / split / cls).mkdir(parents=True, exist_ok=True)
    for cls in CATEGORIES:
        (CLIPS / cls).mkdir(parents=True, exist_ok=True)

def download_file(url: str, out_path: Path, timeout=30) -> bool:
    """Download url to out_path. Returns True on success, False otherwise."""
    if not HAVE_REQUESTS:
        return False
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception as e:
        print(f"[WARN] Download failed: {url} -> {e}")
        return False

def evenly_spaced_indices(n, k):
    if n <= 0:  # guard
        return [0] * k
    return np.linspace(0, max(0, n - 1), k, dtype=int).tolist()

def extract_frames(video_path: Path, out_dir: Path, k=8, size=(224, 224)) -> bool:
    """Extract k frames from video into out_dir; return True if at least 1 frame written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        cap.release()
        return False

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = evenly_spaced_indices(n, k)
    wrote = 0
    for i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if not ok:
            # Fallback empty frame
            fr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        else:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            fr = cv2.resize(fr, size)
        Image.fromarray(fr).save(out_dir / f"frame_{i}.jpg")
        wrote += 1
    cap.release()
    return wrote > 0

def make_dummy_frames(out_dir: Path, k=8, size=(224,224)):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(k):
        arr = (np.random.rand(size[1], size[0], 3) * 255).astype("uint8")
        Image.fromarray(arr).save(out_dir / f"frame_{i}.jpg")

def ensure_train_and_test_for_class(cls: str):
    """Ensure data/train/<cls>/sample_001 and data/test/<cls>/sample_001 exist."""
    # Prefer local clip if user placed one in clips/<cls>/*.mp4; else try download; else dummy
    local_mp4s = sorted((CLIPS / cls).glob("*.mp4"))
    preferred = None

    # If user put a clip in clips/<cls>, use the first one
    if local_mp4s:
        preferred = local_mp4s[0]
    else:
        # Try to download our suggested sample (if requests exists)
        dst = (CLIPS / cls / f"{cls}_sample.mp4")
        if not dst.exists():
            ok = download_file(URLS.get(cls, ""), dst)
            if ok:
                preferred = dst

    # TRAIN
    train_out = DATA / "train" / cls / "sample_001"
    test_out  = DATA / "test"  / cls / "sample_001"

    if preferred and preferred.exists():
        ok_train = extract_frames(preferred, train_out, k=8, size=(224,224))
        if not ok_train:
            print(f"[INFO] Fallback to dummy frames for train/{cls}")
            make_dummy_frames(train_out, k=8)
    else:
        print(f"[INFO] No video for {cls}. Making dummy frames for train.")
        make_dummy_frames(train_out, k=8)

    # TEST (mirror train sample_001 for a smoke test)
    if test_out.exists():
        shutil.rmtree(test_out)
    shutil.copytree(train_out, test_out)
    print(f"[OK] Created train/test samples for {cls}")

def main():
    ensure_dirs()
    print("=== Creating Kiddie Guard Example Dataset ===")
    for cls in CATEGORIES:
        ensure_train_and_test_for_class(cls)

    # Show a quick listing
    print("\nTop files under data/:")
    shown = 0
    for p in sorted(DATA.rglob("*.jpg")):
        print(" ", p.relative_to(PROJECT_ROOT))
        shown += 1
        if shown >= 24:
            break
    print("\n✅ Dataset ready under data/train and data/test")

if __name__ == "__main__":
    main()
