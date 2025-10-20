"""Calibration reliability metrics for probability predictions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error.
    
    ECE = sum_i (|acc_i - conf_i| * n_i / N)
    where acc_i = actual positive rate in bin i
          conf_i = average predicted probability in bin i
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_pred[mask].mean()
        ece += np.abs(acc - conf) * mask.sum()
    
    return float(ece / len(y_true))


def compute_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Brier score: mean squared error of probabilities."""
    return float(np.mean((y_true - y_pred) ** 2))


def calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve data.
    
    Returns:
        mean_pred_per_bin: average predicted probability per bin
        fraction_pos_per_bin: actual positive rate per bin
        counts_per_bin: number of samples per bin
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    mean_pred = np.zeros(n_bins)
    frac_pos = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            mean_pred[i] = bins[i]
            frac_pos[i] = 0.0
            counts[i] = 0
        else:
            mean_pred[i] = y_pred[mask].mean()
            frac_pos[i] = y_true[mask].mean()
            counts[i] = mask.sum()
    
    return mean_pred, frac_pos, counts


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    n_bins: int = 10,
    title: str = "Reliability Diagram"
) -> None:
    """
    Plot and save reliability diagram (calibration curve).
    
    Args:
        y_true: Binary labels
        y_pred: Predicted probabilities
        output_path: Where to save the plot
        n_bins: Number of bins for calibration curve
        title: Plot title
    """
    mean_pred, frac_pos, counts = calibration_curve(y_true, y_pred, n_bins)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calibration curve
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)
    ax1.plot(mean_pred, frac_pos, 'o-', label='Model', markersize=8)
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Distribution histogram
    ax2.hist(y_pred, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Prediction Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Reliability diagram saved to {output_path}")


def export_calibration_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
    name: str = "calibration_report",
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Export comprehensive calibration report with metrics and plots.
    
    Args:
        y_true: Binary labels
        y_pred: Predicted probabilities
        output_dir: Directory to save report files
        name: Base name for output files
        n_bins: Number of bins
    
    Returns:
        Dictionary with reliability metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute metrics
    ece = compute_ece(y_true, y_pred, n_bins)
    brier = compute_brier(y_true, y_pred)
    mean_pred, frac_pos, counts = calibration_curve(y_true, y_pred, n_bins)
    
    # Build report
    report = {
        "ece": float(ece),
        "brier": float(brier),
        "n_samples": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "mean_prediction": float(y_pred.mean()),
        "bins": {
            "n_bins": int(n_bins),
            "mean_pred": mean_pred.tolist(),
            "fraction_pos": frac_pos.tolist(),
            "counts": counts.tolist(),
        }
    }
    
    # Save JSON report
    report_path = output_dir / f"{name}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"📋 Calibration report: {report_path}")
    print(f"   ECE: {ece:.4f}, Brier: {brier:.4f}")
    
    # Save reliability diagram
    plot_path = output_dir / f"{name}_reliability.png"
    plot_reliability_diagram(
        y_true, y_pred, plot_path, n_bins,
        title=f"Reliability Diagram (ECE={ece:.4f}, Brier={brier:.4f})"
    )
    
    return report

