# Kiddie Guard (Grad 699)
Detecting cartoon violence using a CNN + LSTM hybrid model.

## Structure
- model/: model architecture (cnn_lstm), training, evaluation
- utils/: preprocessing (video→tensor), dataset loader
- app/: Flask web UI
- data/: put your train/test samples here
- reports/: placeholders for docs
- requirements.txt: dependencies

## Quickstart
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
python model/train_model.py  # after adding data
python model/evaluate.py
python app/app.py  # open http://localhost:5000