from flask import Flask, render_template, request, flash
import torch, os
from model.cnn_lstm import CNN_LSTM
from utils.preprocess import preprocess_video

app = Flask(__name__); app.secret_key = 'demo'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_LSTM().to(device)
w = 'model/weights/kiddie_guard.pth'
if os.path.exists(w): model.load_state_dict(torch.load(w, map_location=device))
model.eval()
labels = ['Violent','Non-Violent','Comedic']

@app.route('/', methods=['GET','POST'])
def index():
  res = None; conf = None
  if request.method == 'POST' and 'video' in request.files:
    v = request.files['video']
    try:
      seq = preprocess_video(v)
      with torch.no_grad():
        p = model(seq.to(device)); c, i = p.max(1)
        res, conf = labels[i.item()], float(c)
    except Exception as e:
      flash(str(e))
  return render_template('index.html', result=res, conf=conf)

if __name__ == '__main__':
  app.run(port=5000)