# model/evaluate.py
import os, csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from model.cnn_lstm import CNN_LSTM
from utils.dataset_loader import CartoonSeqDataset

def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def precision_recall_f1(cm):
    eps = 1e-12
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    return prec, rec, f1

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = CartoonSeqDataset('data/test', seq_len=8)
    if len(ds) == 0:
        print(" No test data found in data/test."); return

    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    model = CNN_LSTM(num_classes=len(ds.class_to_idx)).to(device)
    weights_path = 'model/weights/kiddie_guard.pth'
    if not os.path.exists(weights_path):
        print(f" Weights not found at {weights_path}. Train first."); return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    labels_sorted = sorted(ds.class_to_idx.keys(), key=lambda k: ds.class_to_idx[k])
    idx_to_label = {i: lbl for i, lbl in enumerate(labels_sorted)}

    y_true, y_pred, confidences = [], [], []
    sample_paths = []  # for CSV
    with torch.no_grad():
        for i, (x, y) in enumerate(dl):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

            y_true.append(int(y.item()))
            y_pred.append(int(pred.item()))
            confidences.append(float(conf.item()))
            # stash sample dir (for report clarity)
            sp, _ = ds.samples[i]
            sample_paths.append(sp)

    y_true_np, y_pred_np = np.array(y_true), np.array(y_pred)
    acc = (y_true_np == y_pred_np).mean() if len(y_true_np) else 0.0
    cm = confusion_matrix_np(y_true_np, y_pred_np, num_classes=len(labels_sorted))
    prec, rec, f1 = precision_recall_f1(cm)

    print(f"Test Accuracy: {acc*100:.2f}%")
    print("Classes (index → label):", idx_to_label)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    for row in cm:
        print(" ", row.tolist())

    print("\nPer-class metrics:")
    for i, lbl in idx_to_label.items():
        print(f"  {lbl:12s}  Prec: {prec[i]*100:5.1f}%  Rec: {rec[i]*100:5.1f}%  F1: {f1[i]*100:5.1f}%  (n={cm[i].sum()})")

    if confidences:
        print(f"\nAvg confidence (predicted class): {np.mean(confidences)*100:.1f}%")

    # Save predictions to CSV for your report
    os.makedirs("reports", exist_ok=True)
    out_csv = "reports/test_predictions.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_path", "true_label", "pred_label", "confidence"])
        for sp, t, p, c in zip(sample_paths, y_true_np, y_pred_np, confidences):
            w.writerow([sp, idx_to_label[t], idx_to_label[p], f"{c:.4f}"])
    print(f"\n📄 Wrote per-sample predictions: {out_csv}")

if __name__ == '__main__':
    evaluate()

