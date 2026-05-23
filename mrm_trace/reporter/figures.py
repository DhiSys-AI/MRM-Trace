"""
Optional figure generation — requires matplotlib.

Produces:
  retention_histogram.png  — per-region_type retention distribution
  working_set.png          — total working-set bytes per region_type (bar chart)
  write_once.png           — write-once ratio per region (scatter/bar)

All functions return the output Path, or None if matplotlib is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe in WSL / CI
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _require_mpl() -> bool:
    return _HAS_MPL


def plot_retention_histogram(
    retention_per_region: pd.DataFrame,
    run_dir: Union[str, Path],
) -> Optional[Path]:
    """
    Histogram of retention_s grouped by region_type.

    Returns path written, or None if matplotlib is unavailable.
    """
    if not _require_mpl():
        return None

    out = Path(run_dir) / "retention_histogram.png"
    fig, ax = plt.subplots(figsize=(8, 4))

    for rtype, grp in retention_per_region.groupby("region_type"):
        ax.hist(grp["retention_s"], bins=30, alpha=0.7, label=rtype)

    ax.set_xlabel("Retention (s)")
    ax.set_ylabel("Region count")
    ax.set_title("Retention distribution by region type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_working_set(
    working_set_summary: pd.DataFrame,
    run_dir: Union[str, Path],
) -> Optional[Path]:
    """
    Bar chart of total working-set bytes per region_type.

    Returns path written, or None if matplotlib is unavailable.
    """
    if not _require_mpl():
        return None

    out = Path(run_dir) / "working_set.png"
    df = working_set_summary.copy()
    df["working_set_gb"] = df["total_bytes"] / 1_073_741_824

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["region_type"], df["working_set_gb"], color="steelblue")
    ax.set_xlabel("Region type")
    ax.set_ylabel("Working set (GB)")
    ax.set_title("Working set by region type")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_write_once(
    write_once: pd.DataFrame,
    run_dir: Union[str, Path],
) -> Optional[Path]:
    """
    Bar chart of mean write-once ratio per region_type.

    Returns path written, or None if matplotlib is unavailable.
    """
    if not _require_mpl():
        return None

    out = Path(run_dir) / "write_once.png"
    agg = (
        write_once.groupby("region_type")["write_once_ratio"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(agg["region_type"], agg["write_once_ratio"], color="coral")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Region type")
    ax.set_ylabel("Write-once ratio")
    ax.set_title("Write-once ratio by region type")
    ax.axhline(0.8, color="darkred", linestyle="--", linewidth=0.8, label="high_mrm threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def generate_all_figures(
    analysis: dict,
    run_dir: Union[str, Path],
) -> list:
    """
    Generate all figures.  Returns list of Paths written (empty if no matplotlib).
    """
    run_dir = Path(run_dir)
    written = []
    for fn, key1, key2 in (
        (plot_retention_histogram, "retention_per_region", None),
        (plot_working_set,         "working_set_summary",  None),
        (plot_write_once,          "write_once",           None),
    ):
        df = analysis[key1]
        p = fn(df, run_dir)
        if p is not None:
            written.append(p)
    return written
