import os, sys, numpy as np, traceback
import torch
from torch.utils.data import DataLoader
from model.cnn_lstm import CNN_LSTM
from utils.dataset_loader import CartoonSeqDataset

def train():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  ds = CartoonSeqDataset('data/train', seq_len=8)
  if len(ds) == 0:
    print('Add data to data/train (see README).'); return

  batch_size = 4
  epochs = 25
  dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

  model = CNN_LSTM().to(device)

  # ---- class weights (inverse frequency) ----
  num_classes = len(ds.class_to_idx)
  counts = np.zeros(num_classes, dtype=np.int64)
  for _, lbl in ds.samples:
    counts[lbl] += 1
  counts = np.where(counts == 0, 1, counts)  # avoid div-by-zero
  class_weights = torch.tensor(1.0 / counts, dtype=torch.float32, device=device)

  opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
  crit = torch.nn.CrossEntropyLoss(weight=class_weights)

  # Optional LR schedule (nice-to-have)
  steps_per_epoch = max(1, len(dl))
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)

  for ep in range(epochs):
    model.train(); total = 0.0
    for x, y in dl:
      x, y = x.to(device), y.to(device)
      opt.zero_grad()
      logits = model(x)             # raw logits now
      loss = crit(logits, y)
      loss.backward()
      opt.step()
      scheduler.step()              # if you keep the scheduler
      total += loss.item()
    print(f"Epoch {ep+1}/{epochs} - Loss: {total/len(dl):.4f}")

  os.makedirs('model/weights', exist_ok=True)
  torch.save(model.state_dict(), 'model/weights/kiddie_guard.pth')
  print(' Saved model/weights/kiddie_guard.pth')

if __name__ == '__main__':
  train()
