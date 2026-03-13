#!/usr/bin/env python3
"""Plot AlphaEvolve optimization progress.

Usage:
    python plot_progress.py experiments/run_001
    python plot_progress.py experiments/run_001 --island first_try
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_experiment_data(exp_dir: Path, island_filter: str | None = None):
    """Load all variant fitness data from an experiment directory."""
    variants = []  # (experiment_num, name, fitness, kept)
    elites = []    # (experiment_num, name, fitness)

    exp_num = 0
    islands_dir = exp_dir / "islands"
    if not islands_dir.exists():
        print(f"No islands directory found at {islands_dir}")
        return [], []

    for island_dir in sorted(islands_dir.iterdir()):
        if not island_dir.is_dir():
            continue
        island_name = island_dir.name
        if island_filter and island_name != island_filter:
            continue

        gens_dir = island_dir / "generations"
        if not gens_dir.exists():
            continue

        for gen_dir in sorted(gens_dir.iterdir()):
            if not gen_dir.is_dir() or not gen_dir.name.startswith("gen_"):
                continue

            result_path = gen_dir / "result.json"
            if not result_path.exists():
                continue

            result = json.loads(result_path.read_text())
            selected = result.get("selected", "")
            fitnesses = result.get("variant_fitnesses", {})

            for vname, fitness in fitnesses.items():
                exp_num += 1
                kept = vname == selected and result.get("elite_updated", False)
                variants.append({
                    "exp_num": exp_num,
                    "name": vname,
                    "fitness": fitness,
                    "kept": kept,
                    "island": island_name,
                    "generation": result.get("generation", 0),
                })
                if kept:
                    elites.append({
                        "exp_num": exp_num,
                        "name": vname,
                        "fitness": fitness,
                        "island": island_name,
                    })

    return variants, elites


def plot_progress(exp_dir: Path, island_filter: str | None = None, output: str | None = None):
    variants, elites = load_experiment_data(exp_dir, island_filter)

    if not variants:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    # All variants (grey dots = discarded)
    discarded = [v for v in variants if not v["kept"]]
    kept = [v for v in variants if v["kept"]]

    if discarded:
        ax.scatter(
            [v["exp_num"] for v in discarded],
            [v["fitness"] for v in discarded],
            c="#cccccc", s=30, alpha=0.6, label=f"Discarded ({len(discarded)})",
            zorder=1,
        )

    # Kept variants (green dots)
    if kept:
        ax.scatter(
            [v["exp_num"] for v in kept],
            [v["fitness"] for v in kept],
            c="#2ecc71", s=60, zorder=3, label=f"Kept ({len(kept)})",
            edgecolors="white", linewidth=0.5,
        )
        # Labels for kept variants
        for v in kept:
            ax.annotate(
                v["name"],
                (v["exp_num"], v["fitness"]),
                fontsize=6, color="#27ae60", alpha=0.8,
                xytext=(5, 5), textcoords="offset points",
                rotation=15,
            )

    # Running best line (step function)
    if kept:
        sorted_kept = sorted(kept, key=lambda v: v["exp_num"])
        running_best_x = [0]
        running_best_y = [sorted_kept[0]["fitness"]]
        best_so_far = sorted_kept[0]["fitness"]

        for v in sorted_kept:
            if v["fitness"] > best_so_far:
                # Step to new best
                running_best_x.extend([v["exp_num"], v["exp_num"]])
                running_best_y.extend([best_so_far, v["fitness"]])
                best_so_far = v["fitness"]

        # Extend to the end
        max_exp = max(v["exp_num"] for v in variants)
        running_best_x.append(max_exp)
        running_best_y.append(best_so_far)

        ax.plot(running_best_x, running_best_y, c="#2ecc71", linewidth=2,
                alpha=0.7, label="Running best", zorder=2)

    # Title and labels
    title_parts = [f"{len(variants)} Experiments, {len(kept)} Kept Improvements"]
    if island_filter:
        title_parts.insert(0, f"Island: {island_filter}")
    else:
        title_parts.insert(0, "All Islands")
    ax.set_title(f"Optimization Progress: {' — '.join(title_parts)}", fontsize=14)
    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Fitness (higher is better)", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = output or str(exp_dir / "progress.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot optimization progress")
    parser.add_argument("exp_dir", type=Path, help="Experiment directory")
    parser.add_argument("--island", default=None, help="Filter to one island")
    parser.add_argument("--output", "-o", default=None, help="Output path (default: exp_dir/progress.png)")
    args = parser.parse_args()

    plot_progress(args.exp_dir, island_filter=args.island, output=args.output)


if __name__ == "__main__":
    main()
