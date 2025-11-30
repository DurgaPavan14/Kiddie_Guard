import cv2, numpy as np, torch, os

def preprocess_video(fileobj, seq_len=8, resize=(224,224)):
  if hasattr(fileobj, 'read'):
    os.makedirs('uploads', exist_ok=True)
    tmp = 'uploads/tmp.mp4'
    with open(tmp, 'wb') as f: f.write(fileobj.read())
    cap = cv2.VideoCapture(tmp)
  else:
    cap = cv2.VideoCapture(str(fileobj))
  n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  if n <= 0:
    cap.release(); raise ValueError('Empty/unreadable video')
  idxs = np.linspace(0, max(1, n-1), seq_len, dtype=int)
  frames = []
  for i in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, fr = cap.read()
    if not ok: fr = np.zeros((224,224,3), np.uint8)
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    fr = cv2.resize(fr, resize)
    fr = (fr.astype(np.float32)/255.).transpose(2,0,1)
    frames.append(fr)
  cap.release()
  arr = np.stack(frames, 0)
  return torch.from_numpy(arr).unsqueeze(0)