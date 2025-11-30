from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import sqlite3
from datetime import datetime
from threading import Thread

from utils import process_video, process_image
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__)
# Folder used by the web app to serve processed media back to the user.
app.config['UPLOAD_FOLDER'] = 'static/output/'
# Additional archive folder where we keep original uploads for retraining.
ARCHIVE_UPLOAD_FOLDER = 'uploads'
# Folder where we also store feedback in plain text files for offline retraining.
FEEDBACK_TEXT_FOLDER = 'feedback'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(ARCHIVE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FEEDBACK_TEXT_FOLDER, exist_ok=True)

# Allow React frontend (running on a different port) to call the API.
# We now allow CORS on all routes so React can post to /feedback as well.
CORS(app, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(app, cors_allowed_origins="*")

progress = 0
last_result = {}
DB_PATH = 'feedback.db'

ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_VIDEO_EXTENSIONS or ext in ALLOWED_IMAGE_EXTENSIONS


def is_image_file(filename):
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_IMAGE_EXTENSIONS


def is_video_file(filename):
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_VIDEO_EXTENSIONS


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # User feedback table.
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_path TEXT,
            frame_id INTEGER,
            correct_label TEXT,
            comment TEXT,
            submitted_by TEXT,
            created_at TEXT
        )
        """
    )
    # Media table: one row per processed video/image.
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS media (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT,
            media_type TEXT,
            created_at TEXT
        )
        """
    )
    # Per-frame (or per-image) prediction table for active learning.
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS frame_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER,
            frame_index INTEGER,
            time_sec REAL,
            label TEXT,
            prob_non_violent REAL,
            prob_violent REAL,
            prob_comedic REAL,
            max_conf REAL,
            is_uncertain INTEGER,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()
    # Clean up old media and associated predictions/feedback.
    cleanup_old_media()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global progress
    progress = 0  # Reset progress
    cleanup_old_media()

    if 'video' not in request.files:
        # In templates we use the same field name for videos and images.
        return redirect(url_for('index'))

    file = request.files['video']

    if file.filename == '':
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        # Unsupported file type.
        return render_template('index.html', error='Unsupported file type. Please upload MP4, AVI, PNG, or JPEG.')

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        # Also archive the original upload for future retraining.
        try:
            archive_path = os.path.join(ARCHIVE_UPLOAD_FOLDER, filename)
            if not os.path.exists(archive_path):
                import shutil
                shutil.copy2(filepath, archive_path)
        except Exception as e:
            print(f"WARNING: could not archive upload to '{ARCHIVE_UPLOAD_FOLDER}': {e}")

        is_image = is_image_file(filename)

        # Background processing so user can see live progress
        def background_task(path, is_image_flag):
            global last_result
            if is_image_flag:
                output_path, summary, timeline = process_image(path, update_progress)
            else:
                output_path, summary, timeline = process_video(path, update_progress)

            media_type = 'image' if is_image_flag else 'video'
            media_id = save_predictions_to_db(output_path, media_type, timeline)

            last_result = {
                'media_path': output_path,
                'summary': summary,
                'media_type': media_type,
                'timeline': timeline,
                'media_id': media_id,
            }
            socketio.emit('processing_complete', {'video_path': output_path})

        Thread(target=background_task, args=(filepath, is_image), daemon=True).start()

        return render_template('processing.html')

    return redirect(url_for('index'))


@app.route('/progress')
def progress_status():
    global progress
    return jsonify({'progress': progress})


@app.route('/result')
def result():
    video_path = request.args.get('video_path')
    summary = None
    media_type = 'video'

    if last_result:
        # If no video_path is provided, fall back to the last processed media.
        if not video_path:
            video_path = last_result.get('media_path') or last_result.get('video_path')
            summary = last_result.get('summary')
            media_type = last_result.get('media_type', 'video')
        # If a video_path is provided and matches the last result, attach summary.
        elif last_result.get('media_path') == video_path or last_result.get('video_path') == video_path:
            summary = last_result.get('summary')
            media_type = last_result.get('media_type', 'video')

    return render_template('result.html', video_path=video_path, summary=summary, media_type=media_type)


# --------- JSON API for React frontend ---------

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """API endpoint for React frontend to upload media.

    Expects form-data with field name 'file'. Starts background processing and
    immediately returns a JSON response indicating that processing has
    started. Progress and completion are reported via Socket.IO events.
    """
    global progress
    progress = 0
    cleanup_old_media()

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded', 'code': 'NO_FILE'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename', 'code': 'EMPTY_FILENAME'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type', 'code': 'BAD_TYPE'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    # Also archive the original upload for future retraining.
    try:
        archive_path = os.path.join(ARCHIVE_UPLOAD_FOLDER, filename)
        if not os.path.exists(archive_path):
            import shutil
            shutil.copy2(filepath, archive_path)
    except Exception as e:
        print(f"WARNING: could not archive upload to '{ARCHIVE_UPLOAD_FOLDER}': {e}")

    is_image = is_image_file(filename)

    def background_task(path, is_image_flag):
        global last_result
        if is_image_flag:
            output_path, summary, timeline = process_image(path, update_progress)
        else:
            output_path, summary, timeline = process_video(path, update_progress)

        media_type = 'image' if is_image_flag else 'video'
        media_id = save_predictions_to_db(output_path, media_type, timeline)

        last_result = {
            'media_path': output_path,
            'summary': summary,
            'media_type': media_type,
            'timeline': timeline,
            'media_id': media_id,
        }
        socketio.emit('processing_complete', {'video_path': output_path})

    Thread(target=background_task, args=(filepath, is_image), daemon=True).start()

    return jsonify({'status': 'processing', 'media_type': 'image' if is_image else 'video'})


@app.route('/api/result')
def api_result():
    """Return the last processed result as JSON for the React frontend."""
    if not last_result:
        return jsonify({'error': 'No result available'}), 404
    return jsonify(last_result)


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Accept feedback from either HTML form or React frontend.

    - Always writes to SQLite + text file.
    - If the client prefers JSON (React), return a JSON response.
    - Otherwise, redirect back to the result page (legacy HTML flow).
    """
    video_path = request.form.get('video_path')
    frame_id = request.form.get('frame_id')
    correct_label = request.form.get('correct_label')
    comment = request.form.get('comment')
    submitted_by = request.form.get('submitted_by')

    try:
        frame_id_int = int(frame_id) if frame_id else None
    except ValueError:
        frame_id_int = None

    now_iso = datetime.utcnow().isoformat()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO feedback (video_path, frame_id, correct_label, comment, submitted_by, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            video_path,
            frame_id_int,
            correct_label,
            comment,
            submitted_by,
            now_iso,
        ),
    )
    conn.commit()
    conn.close()

    # Also write a plain-text feedback record for offline retraining scripts.
    try:
        os.makedirs(FEEDBACK_TEXT_FOLDER, exist_ok=True)
        safe_label = correct_label or "unknown"
        basename = os.path.basename(video_path) if video_path else "unknown_media"
        filename = f"feedback_{basename}_{now_iso.replace(':', '-').replace('T', '_')}_.txt"
        filepath = os.path.join(FEEDBACK_TEXT_FOLDER, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"video_path: {video_path}\n")
            f.write(f"frame_id: {frame_id_int}\n")
            f.write(f"correct_label: {safe_label}\n")
            f.write(f"comment: {comment or ''}\n")
            f.write(f"submitted_by: {submitted_by or ''}\n")
            f.write(f"created_at: {now_iso}\n")
    except Exception as e:
        print(f"WARNING: could not write feedback text file: {e}")

    # If the client expects JSON (e.g., React fetch), return JSON instead of redirect.
    accepts_json = 'application/json' in (request.headers.get('Accept') or '')
    if accepts_json:
        return jsonify({'status': 'ok'}), 200

    # Legacy HTML flow: redirect back to result page.
    return redirect(url_for('result', video_path=video_path))


def update_progress(percent):
    global progress
    progress = percent
    socketio.emit('progress', {'percent': percent})


def save_predictions_to_db(media_path, media_type, timeline):
    """Persist per-frame predictions for active learning and metrics.

    Returns the media_id for later reference.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    created_at = datetime.utcnow().isoformat()

    c.execute(
        "INSERT INTO media (path, media_type, created_at) VALUES (?, ?, ?)",
        (media_path, media_type, created_at),
    )
    media_id = c.lastrowid

    threshold = 0.6  # uncertainty threshold on max probability
    rows = []
    for entry in timeline:
        probs = entry.get('probs', {})
        p_nv = float(probs.get('non_violent', 0.0))
        p_v = float(probs.get('violent', 0.0))
        p_c = float(probs.get('comedic', 0.0))
        max_conf = max(p_nv, p_v, p_c)
        is_uncertain = 1 if max_conf < threshold else 0
        rows.append(
            (
                media_id,
                int(entry.get('frame_index', 0)),
                float(entry.get('time_sec', 0.0)),
                str(entry.get('label', 'unknown')),
                p_nv,
                p_v,
                p_c,
                max_conf,
                is_uncertain,
                created_at,
            )
        )

    c.executemany(
        """
        INSERT INTO frame_predictions (
            media_id, frame_index, time_sec, label,
            prob_non_violent, prob_violent, prob_comedic,
            max_conf, is_uncertain, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    conn.commit()
    conn.close()
    return media_id


def cleanup_old_media():
    """Delete media and predictions older than 12 hours from disk and DB."""
    from datetime import timedelta

    cutoff = datetime.utcnow() - timedelta(hours=12)
    cutoff_iso = cutoff.isoformat()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, path, media_type FROM media WHERE created_at < ?", (cutoff_iso,))
    rows = c.fetchall()

    for media_id, path, media_type in rows:
        # Delete files from disk (processed and original if possible).
        for p in {path, path.replace('_processed', '')}:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

        # Remove associated frame predictions and feedback rows.
        c.execute("DELETE FROM frame_predictions WHERE media_id = ?", (media_id,))
        c.execute("DELETE FROM feedback WHERE video_path = ?", (path,))

        c.execute("DELETE FROM media WHERE id = ?", (media_id,))

    conn.commit()
    conn.close()


if __name__ == '__main__':
    init_db()
    socketio.run(app, debug=True)
