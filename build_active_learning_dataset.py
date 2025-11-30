import os
import cv2
from typing import Dict, List, Tuple

FEEDBACK_DIR = "feedback"
UPLOADS_DIR = "uploads"
MODEL_DIR = "model"

# Where to put extracted frames for each label
LABEL_TO_SUBDIR = {
    "non_violent": "non_violent",
    "violent": "violent",
    "comedic": "comedic",
}

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_feedback_file(path: str) -> Dict[str, str]:
    """Parse a single feedback text file into a dict.

    Expected lines like:
      key: value
    """

    data: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
    return data


def load_all_feedback() -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    if not os.path.isdir(FEEDBACK_DIR):
        print(f"No feedback directory '{FEEDBACK_DIR}' found.")
        return entries

    for name in os.listdir(FEEDBACK_DIR):
        if not name.lower().endswith(".txt"):
            continue
        path = os.path.join(FEEDBACK_DIR, name)
        try:
            entry = parse_feedback_file(path)
            entry["__source_file"] = path
            entries.append(entry)
        except Exception as e:
            print(f"WARNING: failed to parse feedback file {path}: {e}")
    return entries


def ensure_label_dir(label: str) -> str:
    sub = LABEL_TO_SUBDIR.get(label)
    if not sub:
        raise ValueError(f"Unknown label '{label}' in feedback.")
    out_dir = os.path.join(MODEL_DIR, sub)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def is_video_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in SUPPORTED_VIDEO_EXTS


def extract_frame(video_path: str, frame_index: int, out_dir: str, prefix: str) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        raise RuntimeError(f"Video {video_path} has no frames.")

    # Clamp frame index into valid range
    idx = max(0, min(frame_index, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {idx} from {video_path}")

    # Save as PNG
    filename = f"{prefix}_f{idx:06d}.png"
    out_path = os.path.join(out_dir, filename)
    cv2.imwrite(out_path, frame)
    return out_path


def process_feedback_entries(entries: List[Dict[str, str]]) -> None:
    if not entries:
        print("No feedback entries found.")
        return

    extracted: List[Tuple[str, str]] = []  # (label, out_path)

    for i, entry in enumerate(entries, start=1):
        video_path = entry.get("video_path") or ""
        correct_label = (entry.get("correct_label") or "").strip()
        frame_id_str = entry.get("frame_id") or ""

        if not video_path:
            print(f"[{i}] Skipping feedback (no video_path): {entry.get('__source_file')}")
            continue
        if not correct_label:
            print(f"[{i}] Skipping feedback (no correct_label): {entry.get('__source_file')}")
            continue

        try:
            frame_id = int(frame_id_str) if frame_id_str else 0
        except ValueError:
            frame_id = 0

        # Map to uploads directory if necessary
        basename = os.path.basename(video_path)
        if os.path.isabs(video_path) or os.path.exists(video_path):
            candidate = video_path
        else:
            candidate = os.path.join(UPLOADS_DIR, basename)

        if not os.path.exists(candidate):
            print(f"[{i}] Video file not found for feedback: {candidate}")
            continue
        if not is_video_file(candidate):
            print(f"[{i}] Not a supported video for feedback: {candidate}")
            continue

        label_dir = ensure_label_dir(correct_label)
        prefix = os.path.splitext(basename)[0]

        try:
            out_path = extract_frame(candidate, frame_id, label_dir, prefix)
            extracted.append((correct_label, out_path))
            print(f"[{i}] Extracted frame {frame_id} from {candidate} -> {out_path}")
        except Exception as e:
            print(f"[{i}] Failed to extract frame from {candidate}: {e}")

    print("\nSummary:")
    if not extracted:
        print("  No frames extracted.")
    else:
        counts: Dict[str, int] = {}
        for label, _ in extracted:
            counts[label] = counts.get(label, 0) + 1
        for label, count in counts.items():
            print(f"  {label}: {count} frames added to {os.path.join(MODEL_DIR, LABEL_TO_SUBDIR[label])}")


def main() -> None:
    print("[Kiddie Guard] Building active-learning dataset from feedback and uploads...")
    entries = load_all_feedback()
    print(f"Loaded {len(entries)} feedback entries.")
    process_feedback_entries(entries)
    print("Done. You can now re-run train_kiddie_guard.py to retrain the model.")


if __name__ == "__main__":
    main()
