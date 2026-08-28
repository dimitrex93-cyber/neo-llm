#!/usr/bin/env python3
"""Loss-Verlauf-Graph für das neo1.1-Training (aus tfevents)."""
import glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

def lese_verlauf(pfade):
    steps, losses, lrs, best = [], [], [], []
    for path in sorted(pfade):
        loader = EventFileLoader(path)
        b = float("inf")
        for ev in loader.Load():
            if not ev.summary.value:
                continue
            for v in ev.summary.value:
                val = None
                if v.tensor and len(v.tensor.float_val) > 0:
                    val = v.tensor.float_val[0]
                elif v.simple_value is not None:
                    val = v.simple_value
                if val is None:
                    continue
                if v.tag == "train/loss":
                    steps.append(ev.step)
                    losses.append(val)
                    b = min(b, val)
                    best.append(b)
                elif v.tag == "train/learning_rate":
                    lrs.append((ev.step, val))
    return steps, losses, best, lrs

def main():
    runs = sorted(glob.glob("out/runs/Aug27_17-*"))
    if not runs:
        print("Kein neo1.1-Run gefunden")
        sys.exit(1)
    ev_files = glob.glob(runs[-1] + "/*.tfevents*")
    steps, losses, best, lrs = lese_verlauf(ev_files)
    if not steps:
        print("Keine Loss-Events")
        sys.exit(1)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("neo1.1:3b - QLoRA-Training (3 Epochen, 1.580 Steps, seq 512)",
                 fontsize=13, fontweight="bold")

    # Loss
    ax1.plot(steps, losses, color="#1f77b4", alpha=0.45, linewidth=0.9,
             label="Loss (pro 5 Steps)")
    ax1.plot(steps, best, color="#d62728", linewidth=2.2,
             label="Bester Loss (kumulativ)")
    ax1.set_ylabel("Loss")
    ax1.set_ylim(0, max(losses) * 1.05)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.text(0.99, 0.97,
             f"Bester Loss: {min(losses):.4f} @ Step {steps[losses.index(min(losses))]}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#d62728", alpha=0.12))

    # Learning-Rate
    if lrs:
        lr_steps = [s for s, _ in lrs]
        lr_vals = [v for _, v in lrs]
        ax2.plot(lr_steps, lr_vals, color="#2ca02c", linewidth=1.6)
        ax2.set_ylabel("Learning Rate")
        ax2.grid(alpha=0.3)

    ax2.set_xlabel("Step")
    # Epochen-Grenzen
    total = 1580
    for ep in (total // 3, 2 * total // 3):
        for ax in (ax1, ax2):
            ax.axvline(ep, color="gray", linestyle="--", alpha=0.5)
    ax1.text(total // 6, ax1.get_ylim()[1] * 0.98, "Epoche 1", ha="center", fontsize=8, color="gray")
    ax1.text(total // 2, ax1.get_ylim()[1] * 0.98, "Epoche 2", ha="center", fontsize=8, color="gray")
    ax1.text(5 * total // 6, ax1.get_ylim()[1] * 0.98, "Epoche 3", ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    out = "neo11_loss_verlauf.png"
    plt.savefig(out, dpi=130)
    print(f"Gespeichert: {out} | {len(steps)} Messpunkte, best={min(losses):.4f}")

if __name__ == "__main__":
    main()
