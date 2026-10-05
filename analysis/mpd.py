"""
Compute the mean Pareto distance metrics (MPD-100 and MPD-20) per generation
and plot them for one or more Auto-DDM result folders.

For generation g, the script takes all parent pools from generation 0 to g,
finds the (epsilon-)Pareto front, and computes the distance of each Pareto
solution to the ideal point (all objectives = 0).
  MPD-100 = mean distance over the full Pareto front
  MPD-20  = mean distance over the 20% closest Pareto solutions
The shaded band is the 2.5-97.5 percentile interval of the distances.

Examples:
    python analysis/mpd.py results/image_toy/seed_0 --reference 0.4498
    python analysis/mpd.py results/image_toy/seed_0 results/image_toy/seed_42 \\
        --labels "seed 0" "seed 42" --out results/image_toy/mpd.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.NSGA_related import get_pareto_front_indices


def mpd_curves(trace, epsilon=0.0, top_frac=0.20):
    """
    trace: parent-pool objectives per generation, shape (T, P, M).
    Returns a dict of arrays with one value per generation.
    """
    T, P, M = trace.shape
    all_objs = trace.reshape(-1, M)
    keys = ["mpd100", "mpd100_lo", "mpd100_hi", "mpd20", "mpd20_lo", "mpd20_hi", "pareto_size"]
    out = {k: np.full(T, np.nan) for k in keys}

    for g in range(T):
        cur = all_objs[: (g + 1) * P]
        idx = get_pareto_front_indices(cur, epsilon=epsilon)
        if idx.size == 0:
            continue
        d_sorted = np.sort(np.linalg.norm(cur[idx], axis=1))
        top = d_sorted[: max(1, int(len(d_sorted) * top_frac))]

        out["mpd100"][g] = np.mean(d_sorted)
        out["mpd100_lo"][g], out["mpd100_hi"][g] = np.percentile(d_sorted, [2.5, 97.5])
        out["mpd20"][g] = np.mean(top)
        out["mpd20_lo"][g], out["mpd20_hi"][g] = np.percentile(top, [2.5, 97.5])
        out["pareto_size"][g] = len(d_sorted)
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot MPD-100 and MPD-20 per generation.")
    parser.add_argument("run_dirs", nargs="+", help="Result folders that contain history_objectives.npy.")
    parser.add_argument("--labels", nargs="+", default=None, help="Legend label for each folder.")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Epsilon for the Pareto sorting (default: 0).")
    parser.add_argument("--reference", type=float, default=None,
                        help="Draw a reference line (image toy optimum: 0.4498).")
    parser.add_argument("--out", default=None, help="Output figure (default: <first run dir>/mpd.png).")
    args = parser.parse_args()

    labels = args.labels or [Path(d).name for d in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        parser.error("Give one label per result folder.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for run_dir, label in zip(args.run_dirs, labels):
        trace = np.load(Path(run_dir) / "history_objectives.npy")
        c = mpd_curves(trace, epsilon=args.epsilon)
        x = np.arange(len(c["mpd100"]))

        np.savetxt(Path(run_dir) / "mpd.csv",
                   np.column_stack([x] + [c[k] for k in c]),
                   delimiter=",", header="generation," + ",".join(c), comments="")
        print(f"{label}: final MPD-100 = {c['mpd100'][-1]:.4f}, final MPD-20 = {c['mpd20'][-1]:.4f}, "
              f"Pareto size = {int(c['pareto_size'][-1])}")

        for ax, key in zip(axes, ["mpd100", "mpd20"]):
            line, = ax.plot(x, c[key], label=label)
            ax.fill_between(x, c[f"{key}_lo"], c[f"{key}_hi"], color=line.get_color(), alpha=0.15)

    for ax, title in zip(axes, ["MPD-100 (full Pareto front)", "MPD-20 (top 20%)"]):
        if args.reference is not None:
            ax.axhline(args.reference, linestyle="--", color="green", lw=1.5, label="reference")
        ax.set_ylabel(title)
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
    axes[-1].set_xlabel("Generation")
    plt.tight_layout()

    out = Path(args.out or Path(args.run_dirs[0]) / "mpd.png")
    fig.savefig(out, dpi=150)
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
