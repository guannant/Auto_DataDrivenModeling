"""
Plot the MPD-100 (full Pareto front) and MPD-20 (top 20%) curves for the
image toy and Cu-Mg CALPHAD case studies from the trace files in this folder.

Run from the repository root:
    python paper_data/plot_mpd.py            # save the figures to paper_data/figures/
    python paper_data/plot_mpd.py --show     # also open the figures
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR.parent))
from utils.NSGA_related import get_pareto_front_indices  # noqa: E402

FIG_DIR = DATA_DIR / "figures"


# ============================================================
# Image toy
# ============================================================
def pareto_cdf_area_curve(trace, get_pareto_front_indices, epsilon_pf=0.0):

    T, P, M = trace.shape
    all_objs = trace.reshape(-1, M)

    area_full = []
    std_full = []

    area_20 = []
    std_20 = []

    k20size = []
    pareto_size = []

    for g in range(T):

        cur = all_objs[: (g + 1) * P]
        idx = get_pareto_front_indices(cur, epsilon=epsilon_pf)

        if idx.size == 0:
            area_full.append(0)
            std_full.append(0)
            area_20.append(0)
            std_20.append(0)
            continue

        F = cur[idx]
        dists = np.linalg.norm(F, axis=1)
        pareto_size.append(len(dists))
        d_sorted = np.sort(dists)

        k20 = max(1, int(len(d_sorted) * 0.20))
        # full Pareto
        area_full.append(np.mean(d_sorted))
        std_full.append(np.std(d_sorted))

        # top 20%
        top20 = d_sorted[:k20]
        area_20.append(np.mean(top20))
        std_20.append(np.std(top20))
        k20size.append(k20)
    return (
        np.array(area_full), np.array(std_full),
        np.array(area_20), np.array(std_20),
        k20size, pareto_size
    )


def plot_image_toy():
    toy_dir = DATA_DIR / "image_toy"
    trace0 = np.load(toy_dir / "trace0.npy")
    trace42 = np.load(toy_dir / "trace42.npy")
    trace147 = np.load(toy_dir / "trace147.npy")

    trace0_no_llm = np.load(toy_dir / "trace0_no_llm.npy")
    trace42_no_llm = np.load(toy_dir / "trace42_no_llm.npy")
    trace147_no_llm = np.load(toy_dir / "trace147_no_llm.npy")

    groups = {
        "Seed:0": (trace0_no_llm, trace0),
        "Seed:42": (trace42_no_llm, trace42),
        "Seed:147": (trace147_no_llm, trace147)
    }

    tiers = ["Full Pareto", "Top 20%"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=True)

    for col, (group_name, (trace_no_llm, trace_llm)) in enumerate(groups.items()):

        full_no, std_full_no, top20_no, std20_no, k20size_no, pareto_size_no = pareto_cdf_area_curve(
            trace_no_llm,
            get_pareto_front_indices,
            epsilon_pf=0.0,
        )

        full_yes, std_full_yes, top20_yes, std20_yes, k20size_yes, pareto_size_yes = pareto_cdf_area_curve(
            trace_llm,
            get_pareto_front_indices,
            epsilon_pf=0.0,
        )

        if group_name == "Seed:147":
            k20size_no_plot = k20size_no
            k20size_yes_plot = k20size_yes
            pareto_size_no_plot = pareto_size_no
            pareto_size_yes_plot = pareto_size_yes

        metrics_no = [(full_no, std_full_no), (top20_no, std20_no)]
        metrics_yes = [(full_yes, std_full_yes), (top20_yes, std20_yes)]

        for row in range(2):
            ax = axes[row, col]

            mean_no, std_no = metrics_no[row]
            mean_yes, std_yes = metrics_yes[row]

            x = np.arange(len(mean_no))

            # Baseline
            ax.plot(x, mean_no, color="red", label="Baseline")
            ax.fill_between(
                x,
                mean_no - std_no,
                mean_no + std_no,
                color="red",
                alpha=0.15,
                linestyle='--',
            )

            # AutoDDM
            ax.plot(x, mean_yes, color="blue", label="AutoDDM")
            ax.fill_between(
                x,
                mean_yes - std_yes,
                mean_yes + std_yes,
                color="blue",
                alpha=0.15,
                linestyle='--',
            )

            ax.axhline(y=0.4498, linestyle='--', color='green', lw=2)

            if row == 0:
                ax.set_title(group_name)

            if col == 0:
                ax.set_ylabel(tiers[row])

            ax.set_yscale("log")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.tick_params(axis='both', labelsize=15)

            ax.legend(fontsize=12)

    for ax in axes[-1]:
        ax.set_xlabel("Generation")

    # enforce shared y-scale per row
    for row in range(2):

        row_axes = axes[row, :]

        ymin = min(ax.get_ylim()[0] for ax in row_axes)
        ymax = max(ax.get_ylim()[1] for ax in row_axes)

        for ax in row_axes:
            ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "image_toy_mpd.png", dpi=200)

    fig2 = plt.figure()
    plt.plot(k20size_no_plot, color="red", label="Baseline")
    plt.plot(k20size_yes_plot, color="blue", label="AutoDDM")
    plt.plot(pareto_size_no_plot, color="green", label="Baseline_full")
    plt.plot(pareto_size_yes_plot, color="yellow", label="AutoDDM_full")
    plt.xlabel("Generation")
    plt.ylabel("Number of Pareto solutions (seed 147)")
    plt.legend()
    fig2.savefig(FIG_DIR / "image_toy_pareto_size_seed147.png", dpi=200)


# ============================================================
# Cu-Mg CALPHAD
# ============================================================
def pareto_cdf_area_curve_calphad(
        raw_objs,
        gen_end_indices,
        get_pareto_front_indices,
        epsilon_array=0.0,
        top_frac=0.2,
        has_llm=True):

    area_full = []
    low_full = []
    high_full = []

    area_top = []
    low_top = []
    high_top = []

    pareto_size = []
    k20size = []

    for g_idx, end_idx in enumerate(gen_end_indices):

        # epsilon scheduling
        if has_llm:
            epsilon_pf = 0.0 if g_idx == 0 else epsilon_array[g_idx - 1]
        else:
            epsilon_pf = epsilon_array

        # Pareto front selection
        cur = raw_objs[:end_idx]
        pf_idx = get_pareto_front_indices(cur, epsilon=epsilon_pf)

        if pf_idx.size == 0:
            area_full.append(0)
            low_full.append(0)
            high_full.append(0)

            area_top.append(0)
            low_top.append(0)
            high_top.append(0)

            pareto_size.append(0)
            k20size.append(0)
            continue

        F = cur[pf_idx]

        # Distance calculation
        dists = np.linalg.norm(F, axis=1)
        d_sorted = np.sort(dists)

        pareto_size.append(len(dists))

        k = max(1, int(len(d_sorted) * top_frac))
        k20size.append(k)

        # Full Pareto statistics
        area_full.append(np.mean(d_sorted))
        low_full.append(np.percentile(d_sorted, 2.5))
        high_full.append(np.percentile(d_sorted, 97.5))

        # Top 20%
        topk = d_sorted[:k]

        area_top.append(np.mean(topk))
        low_top.append(np.percentile(topk, 2.5))
        high_top.append(np.percentile(topk, 97.5))

    return (
        np.array(area_full),
        np.array(low_full),
        np.array(high_full),
        np.array(area_top),
        np.array(low_top),
        np.array(high_top),
        k20size,
        pareto_size
    )


def plot_calphad():
    cal_dir = DATA_DIR / "calphad"
    all_objs_loaded = np.load(cal_dir / "all_objs.npy")
    eps_val = np.load(cal_dir / "epsilon.npy")
    all_objs_loaded_no_llm = np.load(cal_dir / "all_objs_no_llm.npy")

    # Generation indexing: 20 initial candidates, then 10 offspring per generation
    initial_size = 20
    gen_size = 10

    total_points = all_objs_loaded.shape[0]

    gen_end_indices = [initial_size]

    while gen_end_indices[-1] < total_points:
        gen_end_indices.append(gen_end_indices[-1] + gen_size)

    gen_end_indices[-1] = total_points

    full_no, low_full_no, high_full_no, top_no, low_top_no, high_top_no, k20size_no, pareto_size_no = pareto_cdf_area_curve_calphad(
        raw_objs=all_objs_loaded_no_llm,
        gen_end_indices=gen_end_indices,
        get_pareto_front_indices=get_pareto_front_indices,
        epsilon_array=40,
        has_llm=False
    )

    full_yes, low_full_yes, high_full_yes, top_yes, low_top_yes, high_top_yes, k20size_yes, pareto_size_yes = pareto_cdf_area_curve_calphad(
        raw_objs=all_objs_loaded,
        gen_end_indices=gen_end_indices,
        get_pareto_front_indices=get_pareto_front_indices,
        epsilon_array=eps_val,
        has_llm=True
    )

    generations = np.arange(len(gen_end_indices))

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    tiers = ["Full Pareto", "Top 20%"]

    metrics_no = [(full_no, low_full_no, high_full_no),
                  (top_no, low_top_no, high_top_no)]

    metrics_yes = [(full_yes, low_full_yes, high_full_yes),
                   (top_yes, low_top_yes, high_top_yes)]

    for row in range(2):

        ax = axes[row]

        mean_no, low_no, high_no = metrics_no[row]
        mean_yes, low_yes, high_yes = metrics_yes[row]

        # Baseline
        ax.plot(generations, mean_no, color="red", label="Baseline")
        ax.fill_between(generations, low_no, high_no, color="red", alpha=0.15)

        # AutoDDM
        ax.plot(generations, mean_yes, color="blue", label="AutoDDM")
        ax.fill_between(generations, low_yes, high_yes, color="blue", alpha=0.15)

        ax.set_ylabel(tiers[row])
        ax.set_yscale("log")

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis='both', labelsize=15)

        ax.legend(fontsize=12)

    axes[-1].set_xlabel("Generation")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "calphad_mpd.png", dpi=200)


def main():
    parser = argparse.ArgumentParser(description="Plot MPD-100 and MPD-20 from the paper trace files.")
    parser.add_argument("--show", action="store_true", help="Open the figures after they are saved.")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    FIG_DIR.mkdir(exist_ok=True)

    plot_image_toy()
    plot_calphad()

    print(f"Figures saved to {FIG_DIR}:")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"  {f.name}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
