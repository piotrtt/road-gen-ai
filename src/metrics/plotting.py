"""
Plot generation for evaluation runs.

Each function takes a path to a benchmark `aggregated.json` (and the matching
`raw/` directory) and writes a single PNG. They are designed to be called
on-demand by the Flask results viewer.

Plots:
    - plot_diversity_bars: per-method mean diversity with error bars (across repeats)
    - plot_similarity_distribution: per-method violin/histogram of pairwise similarities
    - plot_topo_vs_geo: scatter of topological vs geometric similarity per pair
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt


# ----- helpers --------------------------------------------------------------

LLM_PREFIX = "llm:"


def _load_aggregated(aggregated_path: str | Path) -> dict:
    return json.loads(Path(aggregated_path).read_text())


def _raw_dir_for(aggregated_path: str | Path) -> Path:
    return Path(aggregated_path).parent / "raw"


def _is_llm(method_key: str) -> bool:
    return method_key.startswith(LLM_PREFIX)


def _is_per_model(method_key: str) -> bool:
    """True for any method whose key has the ``<approach>:<family>`` shape."""
    return ":" in method_key


def _color_for(method_key: str) -> str:
    """Stable color assignment: non-LLM in blues, LLM families in distinct hues."""
    if not _is_llm(method_key) and not method_key.startswith("hybrid:"):
        return {
            "random": "#7faecb",
            "least_generated": "#3b78b8",
            "diversity_driven": "#1f4e79",
        }.get(method_key, "#888888")
    # ``llm:<fam>`` and ``hybrid:<fam>`` both carry a family suffix
    prefix, fam = method_key.split(":", 1)
    base = {
        "gpt5": "#d62728",
        "gemini": "#2ca02c",
        "claude": "#9467bd",
    }.get(fam, "#ff7f0e")
    # Hybrid runs use a green palette to distinguish them from plain LLM runs.
    hybrid_base = {
        "gpt5": "#0e8f60",
        "gemini": "#1a7a3a",
        "claude": "#1f6b4a",
    }
    return hybrid_base.get(fam, "#0e8f60") if prefix == "hybrid" else base


def _ordered_methods(agg: dict) -> list[str]:
    """Stable ordering: plain methods first, then ``llm:*`` and ``hybrid:*`` buckets."""
    keys = list(agg.get("methods", {}).keys())
    return [k for k in keys if not _is_per_model(k)] + [k for k in keys if _is_per_model(k)]


def _gather_pairwise(aggregated_path: str | Path, method_key: str, kind: str) -> list[float]:
    """
    Concatenate pairwise similarities for a method across all its repeats.

    `kind` ∈ {"combined", "topological", "geometric"}.
    """
    agg = _load_aggregated(aggregated_path)
    raw_dir = _raw_dir_for(aggregated_path)
    files = agg["methods"][method_key].get("raw_files", [])
    out: list[float] = []
    for rel in files:
        path = (raw_dir.parent / rel)
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        pw = (report.get("diversity") or {}).get("pairwise") or {}
        out.extend(pw.get(kind, []))
    return out


# ----- public plotting API --------------------------------------------------

def plot_diversity_bars(aggregated_path: str | Path, out_path: str | Path) -> Path:
    """Bar chart of mean combined similarity per method, error bars from repeats."""
    agg = _load_aggregated(aggregated_path)
    methods = _ordered_methods(agg)
    means, stds, colors, labels = [], [], [], []
    for k in methods:
        s = (agg["methods"][k].get("stats") or {}).get("diversity_combined_mean")
        if not s or s.get("mean") is None:
            continue
        means.append(s["mean"])
        stds.append(s.get("std") or 0.0)
        colors.append(_color_for(k))
        labels.append(k)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4.5))
    x = list(range(len(labels)))
    ax.bar(x, means, yerr=stds, color=colors, capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean pairwise combined similarity\n(lower = more diverse)")
    ax.set_title(f"Diversity by method  (run {agg.get('run_id', '')})")
    ax.set_ylim(0, max(means + [0.4]) * 1.15 if means else 0.5)
    ax.axhline(0.3, ls="--", color="gray", alpha=0.6, label="thesis target = 0.3")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_similarity_distribution(aggregated_path: str | Path, out_path: str | Path) -> Path:
    """Violin plot of pairwise combined similarities per method (one violin per method)."""
    agg = _load_aggregated(aggregated_path)
    methods = _ordered_methods(agg)
    data, colors, labels = [], [], []
    for k in methods:
        sims = _gather_pairwise(aggregated_path, k, "combined")
        if not sims:
            continue
        data.append(sims)
        colors.append(_color_for(k))
        labels.append(k)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.0), 4.5))
    if data:
        parts = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=True)
        for body, c in zip(parts["bodies"], colors):
            body.set_facecolor(c)
            body.set_alpha(0.65)
            body.set_edgecolor("black")
        for key in ("cmins", "cmaxes", "cbars", "cmeans"):
            if key in parts:
                parts[key].set_edgecolor("black")
        ax.set_xticks(list(range(1, len(labels) + 1)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.axhline(0.3, ls="--", color="gray", alpha=0.6, label="thesis target = 0.3")
    ax.set_ylabel("Pairwise combined similarity")
    ax.set_title("Distribution of pairwise similarities per method")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_topo_vs_geo(aggregated_path: str | Path, out_path: str | Path) -> Path:
    """Scatter of (topological, geometric) similarity per pair, colored by method."""
    agg = _load_aggregated(aggregated_path)
    methods = _ordered_methods(agg)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for k in methods:
        topo = _gather_pairwise(aggregated_path, k, "topological")
        geom = _gather_pairwise(aggregated_path, k, "geometric")
        if not topo or not geom:
            continue
        n = min(len(topo), len(geom))
        ax.scatter(topo[:n], geom[:n], s=18, alpha=0.55, color=_color_for(k), label=k,
                   edgecolor="white", linewidth=0.3)
    # Reference: combined-similarity-equals-0.3 line is geom = 0.6 - topo
    xs = [0, 0.6]
    ax.plot(xs, [0.6 - x for x in xs], ls="--", color="gray", alpha=0.6,
            label="combined = 0.3")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Topological similarity")
    ax.set_ylabel("Geometric similarity")
    ax.set_title("Topological vs geometric similarity (per pair)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_temperature_sweep(summary_paths: Iterable[str | Path], out_path: str | Path) -> Path:
    """
    Line plot: temperature on x-axis, mean similarity on y-axis, one line per family.

    `summary_paths` is an iterable of `summary.json` files written by
    `scripts/temperature_sweep.py`. The model_key field of each summary is used
    as the legend entry.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    found_any = False
    for sp in summary_paths:
        path = Path(sp)
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        rows = sorted(
            (r for r in data.get("rows", []) if r.get("mean_similarity") is not None),
            key=lambda r: r["temperature"],
        )
        if not rows:
            continue
        xs = [r["temperature"] for r in rows]
        ys = [r["mean_similarity"] for r in rows]
        family = data.get("model_key", path.parent.name)
        ax.plot(xs, ys, marker="o", color=_color_for(f"llm:{family}"), label=family)
        # Mark the best
        best_t = data.get("best_temperature")
        best_y = data.get("best_mean_similarity")
        if best_t is not None and best_y is not None:
            ax.scatter([best_t], [best_y], s=120, facecolors="none",
                       edgecolors=_color_for(f"llm:{family}"), linewidth=2)
        found_any = True

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Mean pairwise combined similarity\n(lower = more diverse)")
    ax.set_title("Temperature sweep per model family")
    ax.axhline(0.3, ls="--", color="gray", alpha=0.6, label="thesis target = 0.3")
    if found_any:
        ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
