#!/usr/bin/env python3
"""
Generate per-repository duration distributions from profiling logs.

The script expects profiler outputs in JSON format (one file per instance) and
draws a half-violin plot for each repository, showing the spread of total
durations on a shared figure.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw total duration distributions per repository.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path.home() / ".codeminer" / "profile_log",
        help="Directory containing profiler JSON logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/profile_plots"),
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="total_duration",
        choices=["total_duration"],
        help="Metric to visualize from each profiler log.",
    )
    parser.add_argument(
        "--section-label",
        type=str,
        default=None,
        help="If provided, use the total duration recorded for this profiler section label instead of the root metric.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Skip repositories with fewer than this many samples.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution (dots per inch) for saved plots.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Plot metric using a log scale on the y-axis.",
    )
    return parser.parse_args()


def sanitize_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def load_profiles(
    log_dir: Path, metric: str, section_label: Optional[str]
) -> Tuple[List[Dict[str, Any]], str, str]:
    if not log_dir.exists():
        raise FileNotFoundError(f"Profiler log directory not found: {log_dir}")

    metric_display = "Total Duration (seconds)"
    metric_slug = metric
    if section_label:
        metric_display = f"{section_label} Duration (seconds)"
        metric_slug = f"{section_label}_duration"
    metric_slug = sanitize_slug(metric_slug)

    records: List[Dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            # Skip corrupted entries but continue processing.
            continue

        repo = payload.get("repo")
        if repo is None:
            continue

        value: Optional[float] = None
        if section_label:
            for section in payload.get("sections", []):
                if section.get("label") == section_label:
                    section_total = section.get("total")
                    if isinstance(section_total, (int, float)):
                        value = float(section_total)
                    break
        else:
            metric_value = payload.get(metric)
            if isinstance(metric_value, (int, float)):
                value = float(metric_value)

        if value is None:
            continue

        records.append({"repo": repo, "value": value})

    return records, metric_display, metric_slug


def draw_repo_violin(
    records: Iterable[Dict[str, Any]],
    output_dir: Path,
    metric_display: str,
    metric_slug: str,
    min_samples: int,
    dpi: int,
    log_scale: bool,
) -> Dict[str, Path]:
    sns.set_theme(style="whitegrid")
    df = pd.DataFrame(records)
    if df.empty:
        return {}

    counts = df.groupby("repo")["value"].transform("size")
    df = df[counts >= min_samples]
    if df.empty:
        return {}

    if log_scale:
        df = df[df["value"] > 0]
        if df.empty:
            return {}

    repo_order = (
        df.groupby("repo")["value"].median().sort_values(ascending=True).index.tolist()
    )

    fig_width = max(8, 0.6 * len(repo_order))
    fig_height = 5.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.violinplot(
        data=df,
        x="repo",
        y="value",
        order=repo_order,
        cut=0,
        scale="width",
        bw_adjust=0.5,
        linewidth=1,
        ax=ax,
        color="#4C72B0",
        inner=None,
    )

    # Collapse each violin to its right half for a lopsided distribution view.
    for collection in ax.collections:
        if not hasattr(collection, "get_paths"):
            continue
        for path in collection.get_paths():
            vertices = path.vertices
            if len(vertices) == 0:
                continue
            x_center = np.median(vertices[:, 0])
            vertices[:, 0] = np.maximum(vertices[:, 0], x_center)

    repo_to_values = {
        repo: df.loc[df["repo"] == repo, "value"].values for repo in repo_order
    }

    median_color = "#DD8452"
    for idx, repo in enumerate(repo_order):
        values = np.asarray(repo_to_values[repo], dtype=float)
        if len(values) == 0:
            continue
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        ax.scatter(
            idx + 0.18,
            median,
            color=median_color,
            s=25,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.vlines(
            idx + 0.18,
            q1,
            q3,
            colors=median_color,
            linewidth=1.2,
            zorder=2.5,
        )

        y_offset = q3 if q3 > 0 else values.max()
        if not np.isfinite(y_offset) or y_offset <= 0:
            y_offset = median if median > 0 else values.max() if values.size else 1.0
        if log_scale:
            y_text = y_offset * 1.4
        else:
            y_text = y_offset + max(0.05 * y_offset, 0.5)
        ax.text(
            idx + 0.22,
            y_text,
            f"n={len(values)}",
            ha="left",
            va="bottom",
            fontsize=7,
            rotation=90,
            color="#444444",
        )

    ax.set_xlabel("Repository")
    ylabel = metric_display
    if log_scale:
        ax.set_yscale("log")
        ylabel += " (log scale)"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric_display} by Repository", fontsize=12, pad=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=9)

    ax.margins(x=0.02)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"repo_{metric_slug}_half_violin.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    return {"aggregate": output_path}


def main() -> None:
    args = parse_args()
    records, metric_display, metric_slug = load_profiles(
        args.log_dir, args.metric, args.section_label
    )

    if not records:
        if args.section_label:
            print(
                f"No profiler records contained the section label '{args.section_label}'. Nothing to plot."
            )
        else:
            print("No valid profiler records found. Nothing to plot.")
        return

    saved_plots = draw_repo_violin(
        records=records,
        output_dir=args.output_dir,
        metric_display=metric_display,
        metric_slug=metric_slug,
        min_samples=args.min_samples,
        dpi=args.dpi,
        log_scale=args.log_scale,
    )

    if not saved_plots:
        print(f"No repositories met the minimum sample threshold ({args.min_samples}).")
        return

    print("Generated plots:")
    for repo, path in saved_plots.items():
        print(f"- {repo}: {path}")


if __name__ == "__main__":
    main()
