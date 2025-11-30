import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
  def __init__(self, num_classes=3, hidden=256):
    super().__init__()
    self.cnn = nn.Sequential(
      nn.Conv2d(3,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2),
      nn.Conv2d(32,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2,2)
    )
    # If input is 224x224 -> after two poolings -> 56x56 with 64 channels
    self.lstm = nn.LSTM(64*56*56, hidden, batch_first=True)
    self.fc = nn.Linear(hidden, num_classes)

  def forward(self, x):
    # x: (B, T, C, H, W)
    b, t, c, h, w = x.size()
    feats = []
    for i in range(t):
      f = self.cnn(x[:, i]).reshape(b, -1)
      feats.append(f)
    seq = torch.stack(feats, dim=1)  # (B, T, F)
    out, _ = self.lstm(seq)          # (B, T, H)
    logits = self.fc(out[:, -1, :])  # (B, C)
    return logits