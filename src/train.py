"""
src/train.py -- Training procedure for the Physics-Informed Dual-Branch GAT (Phase 10).

Diabetic foot thermography project. Written out programmatically by
09_training.ipynb Section 9 (via inspect.getsource on the live functions
defined in that notebook's Sections 2-4), so this file and the notebook stay
in sync.

IMPORTANT:
  - evaluate() uses PLAIN model.eval() -- it does NOT use MC-Dropout. MC-Dropout
    (src/mc_dropout.py, Phase 9) is reserved for Phase 11's final evaluation.
  - This module never touches the test split itself -- what a caller passes as
    `loader` is entirely the caller's responsibility. 09_training.ipynb only
    ever passes train_loader/val_loader.
  - pos_weight convention: labels are 1=DM, 0=Control. DM is the majority class
    in train, so its OWN (smaller, <1) inverse-frequency weight is what should
    be passed as BCEWithLogitsLoss's pos_weight -- see 09_training.ipynb
    Section 1 markdown for the full explanation.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, accuracy_score

from src.model import DualBranchGAT


def build_model(config: dict) -> DualBranchGAT:
    """Maps a config dict (PRIMARY_CONFIG-style) to a DualBranchGAT instance.
    Only the constructor-relevant keys are used; graph_topology/image_key are
    forward()-time arguments and are intentionally ignored here.
    """
    return DualBranchGAT(
        use_graph_branch=config["use_graph_branch"],
        use_image_branch=config["use_image_branch"],
        image_in_channels=config["image_in_channels"],
    )


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Moves every tensor value in the batch dict to `device`; leaves the
    plain-Python string metadata fields (subject_id, foot_side, ...) as-is.
    """
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_one_epoch(model, loader, optimizer, criterion, device, graph_topology, image_key):
    model.train()
    total_loss, n_seen = 0.0, 0
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad()
        logits = model(batch, graph_topology=graph_topology, image_key=image_key)
        labels = batch["label"].float()
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        bs = labels.shape[0]
        total_loss += loss.item() * bs
        n_seen += bs
    return total_loss / n_seen


def evaluate(model, loader, criterion, device, graph_topology, image_key):
    """Plain model.eval() evaluation -- NOT MC-Dropout. Returns a dict with
    loss/accuracy/balanced_accuracy/roc_auc/f1 plus the raw (labels, probs)
    arrays for later inspection.
    """
    model.eval()
    total_loss, n_seen = 0.0, 0
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            logits = model(batch, graph_topology=graph_topology, image_key=image_key)
            labels = batch["label"].float()
            loss = criterion(logits, labels)
            bs = labels.shape[0]
            total_loss += loss.item() * bs
            n_seen += bs
            probs = torch.sigmoid(logits)
            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    hard_preds = (all_probs >= 0.5).astype(int)

    return {
        "loss": total_loss / n_seen,
        "accuracy": accuracy_score(all_labels, hard_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, hard_preds),
        "roc_auc": roc_auc_score(all_labels, all_probs),
        "f1": f1_score(all_labels, hard_preds, zero_division=0),
        "labels": all_labels,
        "probs": all_probs,
    }


def train_model(config, train_loader, val_loader, criterion, device,
                 max_epochs=150, patience=20, lr=1e-3, weight_decay=1e-4,
                 checkpoint_path=None):
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    history = []
    best_val_roc_auc = -float("inf")
    best_epoch = -1
    epochs_since_improvement = 0
    print_every = 1 if max_epochs <= 30 else 5

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                      config["graph_topology"], config["image_key"])
        val_metrics = evaluate(model, val_loader, criterion, device,
                                config["graph_topology"], config["image_key"])
        scheduler.step(val_metrics["roc_auc"])

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_f1": val_metrics["f1"],
        })

        improved = val_metrics["roc_auc"] > best_val_roc_auc
        if improved:
            best_val_roc_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            epochs_since_improvement = 0
            if checkpoint_path is not None:
                torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_since_improvement += 1

        if epoch % print_every == 0 or epoch == max_epochs - 1 or improved:
            marker = " *" if improved else ""
            print(f"epoch {epoch:4d} | train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
                  f"val_roc_auc={val_metrics['roc_auc']:.4f} "
                  f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}{marker}")

        if epochs_since_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch} "
                  f"(no val_roc_auc improvement for {patience} epochs).")
            break

    return history, best_epoch, best_val_roc_auc

