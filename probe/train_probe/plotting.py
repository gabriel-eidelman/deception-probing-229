"""Accuracy-vs-layer plots per pooling method with bootstrap CI bands."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

_COLORS = {"behavior": "#e76f51", "strategy": "#2a9d8f",
           "stake_structure": "#264653"}


def _chance(label_name: str) -> float:
    return 1.0 / len(C.LABELS[label_name]["classes"])


def plot_accuracy_vs_layer(results: list[dict], pooling: str, outdir):
    rows = [r for r in results if r["pooling"] == pooling]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for label_name in C.LABELS:
        lr = sorted([r for r in rows if r["label"] == label_name],
                    key=lambda r: r["layer"])
        if not lr:
            continue
        layers = [r["layer"] for r in lr]
        acc = [r["accuracy"] for r in lr]
        lo = [r["acc_ci_lo"] for r in lr]
        hi = [r["acc_ci_hi"] for r in lr]
        col = _COLORS.get(label_name, None)
        ax.plot(layers, acc, "-o", ms=3, color=col, label=label_name)
        ax.fill_between(layers, lo, hi, color=col, alpha=0.15)
        ax.axhline(_chance(label_name), color=col, ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("layer")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Decodability by layer — {pooling}\n"
                 "(bands = 95% bootstrap CI; dotted = chance)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = outdir / f"accuracy_vs_layer_{pooling}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
