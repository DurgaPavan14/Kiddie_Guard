# Kiddie Guard – Cartoon Violence Detection (CNN+LSTM)

Kiddie Guard is a web-based system that classifies cartoon content as **non-violent**, **violent**, or **comedic** using a deep-learning model (CNN + LSTM). It consists of:

- **Backend**: Flask + TensorFlow/Keras + OpenCV + SQLite
- **Frontend**: React (Vite) + Socket.IO
- **Model**: CNN+LSTM trained on frame sequences from `model/violent`, `model/non_violent` (and `model/comedic` when available)
- **Active learning**: User feedback is stored and can be used to retrain the model.

---

## 1. Repository Layout

- `app.py` – Flask backend API and Socket.IO server.
- `model.py` – Model wrapper that loads the trained Keras model and provides a unified `predict(frame)` interface.
- `utils.py` – Video/image processing:
  - Extracts frames with OpenCV
  - Calls the model for each frame
  - Writes processed video/image with overlayed predictions
  - Builds per-media `summary` and `timeline` objects.
- `train_kiddie_guard.py` – CNN+LSTM training script using frames in `model/violent`, `model/non_violent`, and (optionally) `model/comedic`.
- `build_active_learning_dataset.py` – Consumes feedback text files + archived uploads and extracts labeled frames into the `model/` folders.
- `feedback.db` – SQLite database (tables: `media`, `frame_predictions`, `feedback`).
- `uploads/` – Archived original uploads, used for retraining.
- `feedback/` – Plain-text feedback files, used by `build_active_learning_dataset.py`.
- `static/output/` – Processed media served by the backend.
- `frontend/frontend/` – Main React + Vite frontend application.

---

## 2. Running the System

### 2.1 Prerequisites

- Python 3.10+ (with `pip`)
- Node.js + npm

Python dependencies (simplified):

- Flask, Flask-SocketIO, Flask-CORS
- TensorFlow / Keras
- OpenCV (`opencv-python`)
- NumPy

### 2.2 Install Python dependencies

First, activate the provided virtual environment (Windows example):

```bash
venv311\Scripts\activate
```

Then, from the project root (where `app.py` is), install dependencies:

```bash
pip install -r requirements.txt
```

If you are using GPU, install a TensorFlow GPU build that matches your CUDA/CuDNN setup.

### 2.3 Install frontend dependencies

```bash
cd frontend/frontend
npm install
npm run dev
```

### 2.4 Start the backend (Flask + Socket.IO)

From the project root:

```bash
python app.py
```

- Backend runs on `http://localhost:5000`.
- Socket.IO is available on the same origin.

### 2.5 Start the frontend (React + Vite)

In a second terminal:

```bash
cd frontend/frontend
npm run dev
```

- Frontend runs on `http://localhost:5173` (Vite default).
- It communicates with the backend via:
  - `POST http://localhost:5000/api/upload`
  - `GET  http://localhost:5000/api/result`
  - `POST http://localhost:5000/feedback`
  - Socket.IO events (`progress`, `processing_complete`).

Open `http://localhost:5173` in your browser.

---

## 3. Features

### 3.1 Video/Frame Upload

- Users can upload **one video** (MP4, AVI, MOV, MKV) or **one image** (PNG, JPG, JPEG).
- The React upload form sends the selected file as `form-data` to `POST /api/upload`.
- The backend saves the file to `static/output/` and copies it to `uploads/`.
- For videos, frames are read with OpenCV; for images, the single image is processed directly.

### 3.2 Live Analysis & Progress

- Upload starts a background thread that processes the media frame by frame.
- Backend emits Socket.IO `progress` events with a `percent` field.
- React shows a progress bar that updates in real time.
- When processing completes, backend emits `processing_complete` and React fetches `/api/result`.

### 3.3 Per-media Summary & Scoreboard

For each processed media (video or image), backend returns a `summary` and `timeline`:

```json
{
  "media_path": "static/output/example_processed.mp4",
  "media_type": "video",
  "summary": {
    "total_frames": 123,
    "class_counts": {
      "non_violent": 80,
      "violent": 30,
      "comedic": 13
    },
    "class_percentages": {
      "non_violent": 65.0,
      "violent": 24.4,
      "comedic": 10.6
    }
  },
  "timeline": [
    {"frame_index": 0, "time_sec": 0.0, "label": "non_violent", "probs": {...}},
    ...
  ]
}
```

The React `ResultCard` shows:

- **Media name** (video or image file name).
- **Scoreboard** with counts and percentages for non-violent, violent, and comedic.
- Processed video/image preview with overlaid per-frame predictions.
- A **Violence Heatmap** timeline showing where violent segments are concentrated.

### 3.4 Analysis History (per session)

- Every time processing completes, the result is appended to an in-memory **history** list.
- Under the main result card, the "Analysis History" section lists all media analyzed in the current browser session:

  - Type (Video/Image)
  - File name
  - Percentages for non-violent, violent, and comedic frames.

This lets the user compare multiple videos/images without reloading the page.

### 3.5 Feedback & Active Learning

- Below the scoreboard, the React `FeedbackForm` allows the user to correct misclassifications:
  - Frame ID (optional)
  - Correct label (non_violent / violent / comedic)
  - Comment (optional)
  - Name or email (optional)
- `POST /feedback` stores feedback in two places:
  - SQLite `feedback` table (for queries and analytics).
  - Text file in `feedback/` directory (easier for offline scripts).
- `build_active_learning_dataset.py` reads feedback text files, finds the referenced video in `uploads/`, extracts the given frame, and saves it into `model/<label>/`.
- `train_kiddie_guard.py` uses all frames in `model/violent`, `model/non_violent`, and `model/comedic` to retrain the CNN+LSTM model.

### 3.6 Error Handling

- Unsupported file types or missing files result in JSON errors from `/api/upload`.
- React displays a clear error message to the user.
- `model.ViolenceDetectionModel` includes a **fallback heuristic** if the Keras model cannot be loaded.
- `cleanup_old_media()` periodically deletes media older than 12 hours (and related DB rows) to enforce privacy.

---

## 4. Model Training Workflow

1. Prepare your dataset:
   - Place frame images in:
     - `model/non_violent/`
     - `model/violent/`
     - `model/comedic/` (optional, for 3-class training)
2. Train the CNN+LSTM model:

   ```bash
   python train_kiddie_guard.py
   ```

   - Saves best model to `model/violent_non_violent_model.keras`.

3. Evaluate the model (if you add `evaluate_kiddie_guard.py`):

   ```bash
   python evaluate_kiddie_guard.py
   ```

4. Start `app.py`; the backend automatically loads the latest model via `model.py`.

5. As users submit feedback, periodically run:

   ```bash
   python build_active_learning_dataset.py
   python train_kiddie_guard.py
   ```

   to incorporate new labeled frames into the training set.

---

## 5. Database Schema (SQLite)

- `media`:
  - `id INTEGER PRIMARY KEY`
  - `path TEXT` – path to processed media (video/image)
  - `media_type TEXT` – 'video' or 'image'
  - `created_at TEXT`

- `frame_predictions`:
  - `id INTEGER PRIMARY KEY`
  - `media_id INTEGER` – FK to `media.id`
  - `frame_index INTEGER`
  - `time_sec REAL`
  - `label TEXT`
  - `prob_non_violent REAL`
  - `prob_violent REAL`
  - `prob_comedic REAL`
  - `max_conf REAL`
  - `is_uncertain INTEGER`
  - `created_at TEXT`

- `feedback`:
  - `id INTEGER PRIMARY KEY`
  - `video_path TEXT`
  - `frame_id INTEGER`
  - `correct_label TEXT`
  - `comment TEXT`
  - `submitted_by TEXT`
  - `created_at TEXT`

---

## 6. Notes for Deployment

- For production, it is recommended to:
  - Run Flask under Gunicorn or a similar WSGI/ASGI server.
  - Serve the built React app (`npm run build`) via Nginx or another web server.
  - Optionally migrate from SQLite to PostgreSQL using an ORM (e.g., SQLAlchemy).
  - Use a GPU-enabled machine (e.g., AWS EC2 g4dn) for faster training and inference.

This document should give developers and operators enough context to run, understand, and extend the Kiddie Guard system.
