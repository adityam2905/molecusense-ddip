"""
utils/calibration.py  —  Probability calibration and risk bands
────────────────────────────────────────────────────────────────
Two separate things live here:

1. Temperature scaling: one number T fitted on the validation set so that
   sigmoid(logit / T) is a calibrated probability. Scored with expected
   calibration error (ECE) and Brier score.

2. Risk bands by percentile: training uses a 50/50 mix of interacting and
   non-interacting pairs, so even a calibrated probability assumes half of all
   pairs interact. Instead, each score is placed against the scores of
   validation pairs NOT known to interact. "HIGH" means the pair scores above
   95% of those pairs, so roughly 5% of non-interacting pairs are flagged HIGH
   by construction, whatever the class mix of the training data was.
"""

import numpy as np
import torch
import torch.nn.functional as F

HIGH_PERCENTILE = 95.0
MEDIUM_PERCENTILE = 80.0

# Neutral wording on purpose: this is a research model's score, not advice.
RISK_TEXT = {
    "HIGH":   "Model score: high. Scores above {pct:.0f}% of drug pairs not known to interact.",
    "MEDIUM": "Model score: medium. Scores above {pct:.0f}% of drug pairs not known to interact.",
    "LOW":    "Model score: low. In the range of drug pairs not known to interact (percentile {pct:.0f}).",
}


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature T > 0 minimising binary cross-entropy of sigmoid(logits / T)."""
    x = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Weighted mean |accuracy − confidence| over equal-width probability bins."""
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(probs, float) - np.asarray(labels, float)) ** 2))


def calibration_report(logits: np.ndarray, labels: np.ndarray, temperature: float) -> dict:
    raw = 1 / (1 + np.exp(-logits))
    cal = 1 / (1 + np.exp(-logits / temperature))
    return {
        "ece_before": expected_calibration_error(raw, labels),
        "ece_after": expected_calibration_error(cal, labels),
        "brier_before": brier_score(raw, labels),
        "brier_after": brier_score(cal, labels),
    }


def percentile(logit: float, reference_logits: np.ndarray) -> float:
    """
    Share (0–100) of reference (non-interacting) pairs scoring below `logit`,
    rounded DOWN to one decimal so a displayed "95" can never sit in a lower
    band than 95 (94.97 would otherwise print as 95 but band as MEDIUM).
    """
    ref = np.asarray(reference_logits, float)
    raw = 100.0 * np.searchsorted(np.sort(ref), logit, side="left") / len(ref)
    return float(np.floor(raw * 10) / 10)


def risk_from_percentile(pct: float) -> dict:
    if pct >= HIGH_PERCENTILE:
        level = "HIGH"
    elif pct >= MEDIUM_PERCENTILE:
        level = "MEDIUM"
    else:
        level = "LOW"
    # int() rounds down, keeping the text consistent with the band thresholds.
    return {"level": level, "description": RISK_TEXT[level].format(pct=int(pct)), "percentile": pct}


def band_rates(logits: np.ndarray, labels: np.ndarray, reference_logits: np.ndarray) -> dict:
    """Share of interacting / non-interacting pairs landing in each risk band."""
    ref = np.sort(np.asarray(reference_logits, float))
    pcts = 100.0 * np.searchsorted(ref, logits, side="left") / len(ref)
    levels = np.where(pcts >= HIGH_PERCENTILE, "HIGH",
                      np.where(pcts >= MEDIUM_PERCENTILE, "MEDIUM", "LOW"))
    labels = np.asarray(labels)
    out = {}
    for name, mask in (("interacting", labels == 1), ("non_interacting", labels == 0)):
        if mask.any():
            out[name] = {lvl: float((levels[mask] == lvl).mean()) for lvl in ("HIGH", "MEDIUM", "LOW")}
    return out
