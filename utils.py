import cv2
import os
from model import model, CLASS_NAMES


def _overlay_prediction(frame, label_name, confidence):
    """Draw the prediction text on a frame or image."""
    if label_name == "violent":
        text = "Violent"
        color = (0, 0, 255)
    elif label_name == "comedic":
        text = "Comedic"
        color = (0, 255, 255)
    else:
        text = "Non-Violent"
        color = (0, 255, 0)

    cv2.putText(
        frame,
        f"Prediction: {text} ({confidence * 100:.1f}%)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
        cv2.LINE_AA,
    )


def process_video(video_path, progress_callback):
    """Process a video frame by frame and overlay predictions.

    Returns (output_path, summary, timeline) where summary is a dict with
    basic stats and timeline is a list of per-frame prediction records that
    can be stored in the database or visualised as a heatmap.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None, None, []

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = video_path.rsplit('.', 1)[0] + '_processed.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps or 1, (frame_width, frame_height))

    frame_count = 0
    class_counts = {name: 0 for name in CLASS_NAMES}
    timeline = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        label_name, confidence, probs = model.predict(frame)
        _overlay_prediction(frame, label_name, confidence)
        out.write(frame)

        if label_name in class_counts:
            class_counts[label_name] += 1

        time_sec = (frame_count / fps) if fps else 0.0
        timeline.append(
            {
                'frame_index': frame_count,
                'time_sec': time_sec,
                'label': label_name,
                'probs': probs,
            }
        )

        # Update progress
        frame_count += 1
        if total_frames > 0:
            progress_callback(int((frame_count / total_frames) * 100))

    cap.release()
    out.release()

    if frame_count == 0:
        percentages = {name: 0.0 for name in CLASS_NAMES}
    else:
        percentages = {
            name: (class_counts[name] / frame_count * 100.0)
            for name in CLASS_NAMES
        }

    summary = {
        'total_frames': frame_count,
        'class_counts': class_counts,
        'class_percentages': percentages,
    }

    return output_path, summary, timeline


def process_image(image_path, progress_callback):
    """Process a single image and overlay prediction.

    Returns (output_path, summary, timeline) similar to process_video.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not open image.")
        return None, None, []

    label_name, confidence, probs = model.predict(image)
    _overlay_prediction(image, label_name, confidence)

    # Save processed image next to original.
    base, ext = os.path.splitext(image_path)
    output_path = base + '_processed' + ext
    cv2.imwrite(output_path, image)

    class_counts = {name: 0 for name in CLASS_NAMES}
    if label_name in class_counts:
        class_counts[label_name] = 1

    percentages = {
        name: (100.0 if class_counts[name] == 1 else 0.0)
        for name in CLASS_NAMES
    }

    summary = {
        'total_frames': 1,
        'class_counts': class_counts,
        'class_percentages': percentages,
    }

    timeline = [
        {
            'frame_index': 0,
            'time_sec': 0.0,
            'label': label_name,
            'probs': probs,
        }
    ]

    # For a single image we can immediately mark progress as 100%.
    progress_callback(100)

    return output_path, summary, timeline

