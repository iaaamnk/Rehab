"""
Train the bi‑directional LSTM on the Kinect data_new/ dataset.

Usage:
    python -m backend.models.train          (from project root)
    python train.py                         (from backend/models/)
"""

import sys, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.data_loader import MotionDataLoader
from core.feature_extraction import extract_features_batch, FEATURE_NAMES
from core.sequence_model import LSTMCompensationDetector

# ── hyper‑parameters ──────────────────────────────────────────────────
SEQUENCE_LENGTH = 30
BATCH_SIZE      = 32
EPOCHS          = 50
LR              = 1e-3
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_sequences(features, labels, seq_len):
    """Overlapping sliding window; label = majority vote."""
    # Map KInAReT labels: 1 -> 0 (Healthy), >1 -> 1 (Compensatory)
    binary_labels = np.where(labels > 1, 1, 0)
    seqs, lbls = [], []
    for i in range(len(features) - seq_len + 1):
        seqs.append(features[i : i + seq_len])
        lbls.append(1 if np.mean(binary_labels[i : i + seq_len]) > 0.5 else 0)
    return np.array(seqs), np.array(lbls)


def train_model():
    print(f"Device : {DEVICE}")
    print("Loading data_new/ …")

    loader = MotionDataLoader()
    all_pos, all_lbl, info = loader.load_all_trials()
    print(f"\n{len(info)} trials loaded.\n")

    all_seqs, all_seq_lbls = [], []

    for positions, labels, meta in zip(all_pos, all_lbl, info):
        print(f"  ▸ {meta['subject']}/{meta['exercise']} "
              f"({meta['num_frames']} frames)")

        smoothed = loader.apply_lowpass_filter(positions)
        feats = extract_features_batch(smoothed)
        feats = np.nan_to_num(feats, nan=0.0, posinf=180.0, neginf=0.0)

        if len(feats) >= SEQUENCE_LENGTH:
            s, l = make_sequences(feats, labels, SEQUENCE_LENGTH)
            all_seqs.append(s)
            all_seq_lbls.append(l)

    X = np.concatenate(all_seqs)
    y = np.concatenate(all_seq_lbls)

    print(f"\nSequences  : {len(X)}")
    print(f"Classes    : {dict(zip(*np.unique(y.astype(int), return_counts=True)))}")
    print(f"Features/f : {X.shape[-1]}")

    # ── normalise ─────────────────────────────────────────────────────
    flat = X.reshape(-1, X.shape[-1])
    mean, std = flat.mean(0), flat.std(0) + 1e-8
    X = (X - mean) / std

    norm_path = os.path.join(os.path.dirname(__file__), "norm_params.npz")
    np.savez(norm_path, mean=mean, std=std)
    print(f"Norm params → {norm_path}")

    # ── split ─────────────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Class weight to address 74/26 imbalance: pos_weight = n_neg / n_pos
    n_neg = (y_tr == 0).sum()
    n_pos = (y_tr == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    print(f"Class weight   : {pos_weight.item():.3f}  (neg={n_neg}, pos={n_pos})")

    tr_loader = DataLoader(SeqDataset(X_tr, y_tr), BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    te_loader = DataLoader(SeqDataset(X_te, y_te), BATCH_SIZE, num_workers=2, pin_memory=True)

    # ── model ─────────────────────────────────────────────────────────
    model = LSTMCompensationDetector(
        input_size=len(FEATURE_NAMES), hidden_size=64,
        num_layers=2, dropout=0.3,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5, min_lr=1e-5)

    print(f"Params     : {sum(p.numel() for p in model.parameters()):,}")
    print("Training …\n")

    best_acc = 0.0
    model_path = os.path.join(os.path.dirname(__file__), "lstm_model.pth")

    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        t_loss = 0.0
        for bx, by in tr_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE).unsqueeze(1)
            optimiser.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            t_loss += loss.item()

        # eval
        model.eval()
        preds, trues, v_loss = [], [], 0.0
        with torch.no_grad():
            for bx, by in te_loader:
                bx, by_d = bx.to(DEVICE), by.to(DEVICE).unsqueeze(1)
                out = model(bx)
                v_loss += criterion(out, by_d).item()
                # BCEWithLogitsLoss outputs raw logits; apply sigmoid before threshold
                probs = torch.sigmoid(out).cpu()
                preds.extend((probs > 0.5).float().numpy().ravel())
                trues.extend(by.numpy().ravel())

        avg_v_loss = v_loss / len(te_loader)
        acc = accuracy_score(trues, preds)
        scheduler.step(avg_v_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"TrLoss {t_loss/len(tr_loader):.4f}  "
                  f"VaLoss {avg_v_loss:.4f}  "
                  f"Acc {acc:.4f}  "
                  f"LR {optimiser.param_groups[0]['lr']:.2e}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)

    # ── final report ──────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Best validation accuracy: {best_acc:.4f}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in te_loader:
            out = model(bx.to(DEVICE))
            probs = torch.sigmoid(out).cpu()
            preds.extend((probs > 0.5).float().numpy().ravel())
            trues.extend(by.numpy().ravel())

    print(f"Accuracy : {accuracy_score(trues, preds):.4f}")
    print(classification_report(trues, preds,
          target_names=["Healthy", "Compensatory"]))

    cm = confusion_matrix(trues, preds)
    print("Confusion matrix:\n", cm)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"Sensitivity : {tp/(tp+fn):.3f}")
        print(f"Specificity : {tn/(tn+fp):.3f}")

    print(f"\nModel saved → {model_path}")


if __name__ == "__main__":
    train_model()
