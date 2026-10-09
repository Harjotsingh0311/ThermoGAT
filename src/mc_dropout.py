"""
src/mc_dropout.py -- MC-Dropout uncertainty mechanism (Phase 9).

Diabetic foot thermography project. Written out programmatically by
08_mc_dropout.ipynb Section 7 (via inspect.getsource on the live functions
defined in that notebook's Sections 2 and 4), so this file and the notebook
stay in sync.

IMPORTANT: every function here was validated in 08_mc_dropout.ipynb against a
freshly-initialized, UNTRAINED DualBranchGAT. That notebook confirmed:
  - enable_mc_dropout() correctly isolates Dropout/Dropout2d in train mode
    while leaving every other layer (GroupNorm, Linear, Conv2d, ...) in eval
    mode (Section 2).
  - The toggle produces genuinely stochastic outputs when active, and
    genuinely deterministic outputs when inactive (Section 3).
  - mc_dropout_predict() produces NaN-free, correctly-shaped, non-negative-
    epistemic-uncertainty statistics (Section 5), and a stable estimate by
    around T=30 (Section 6).

None of that validation used real (trained) predictive signal -- it only
proves the plumbing is correct. Actual uncertainty numbers only become
meaningful once these functions are reused on a trained model.
"""

import torch
import torch.nn as nn


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """Puts `model` in eval mode, then re-enables train-mode (stochastic)
    behavior specifically on Dropout / Dropout2d layers, leaving every other
    layer (GroupNorm, Linear, Conv2d, ...) in eval mode.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()
    return model


def binary_entropy(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary entropy -[p*log(p) + (1-p)*log(1-p)], with p clamped away from
    exactly 0 or 1 to avoid NaN from log(0).
    """
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def mc_dropout_predict(model: nn.Module, batch: dict, T: int = 30,
                        graph_topology: str = "base",
                        image_key: str = "colormap_image") -> dict:
    """Runs T stochastic MC-Dropout forward passes on `batch` and returns a
    dict of per-sample uncertainty statistics: mean_prob, std_prob,
    predictive_entropy (total), expected_entropy (aleatoric proxy), and
    epistemic_uncertainty (predictive_entropy - expected_entropy).
    """
    enable_mc_dropout(model)

    per_pass_probs = []
    with torch.no_grad():
        for _ in range(T):
            logits = model(batch, graph_topology=graph_topology, image_key=image_key)
            per_pass_probs.append(torch.sigmoid(logits))

    probs = torch.stack(per_pass_probs, dim=0)  # (T, B)

    mean_prob = probs.mean(dim=0)               # (B,)
    std_prob = probs.std(dim=0)                 # (B,)

    predictive_entropy = binary_entropy(mean_prob)          # (B,) total uncertainty
    per_pass_entropy = binary_entropy(probs)                # (T, B)
    expected_entropy = per_pass_entropy.mean(dim=0)         # (B,) aleatoric proxy
    epistemic_uncertainty = predictive_entropy - expected_entropy  # (B,)

    return {
        "mean_prob": mean_prob,
        "std_prob": std_prob,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "epistemic_uncertainty": epistemic_uncertainty,
    }

