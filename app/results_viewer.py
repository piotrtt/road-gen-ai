"""
Flask viewer for road network evaluation runs.

Routes:
    GET /                       Index of evaluation runs
    GET /run/<run_id>           Detail view: config + results table + embedded plots
    GET /run/<run_id>/plot/<n>  PNG bytes for plot N (regenerated on demand)
    GET /temperature            Temperature-sweep results across model families

Run with:
    uv run python -m app.results_viewer
    -> http://localhost:5000
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

from flask import Flask, abort, render_template, send_file

from src.metrics.plotting import (
    plot_diversity_bars,
    plot_similarity_distribution,
    plot_temperature_sweep,
    plot_topo_vs_geo,
)

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_ROOT = PROJECT_ROOT / "outputs" / "evaluation"
TEMP_SWEEP_ROOT = PROJECT_ROOT / "experiments" / "temperature_sweep"

PLOTS = {
    "diversity_bars": plot_diversity_bars,
    "similarity_distribution": plot_similarity_distribution,
    "topo_vs_geo": plot_topo_vs_geo,
}


def _list_runs() -> list[dict]:
    if not EVAL_ROOT.exists():
        return []
    runs = []
    for child in sorted(EVAL_ROOT.iterdir(), reverse=True):
        agg = child / "aggregated.json"
        if not (child.is_dir() and agg.exists()):
            continue
        try:
            data = json.loads(agg.read_text())
        except Exception:
            continue
        # Headline numbers: best (lowest mean similarity)
        best_method, best_mean = None, None
        for k, m in (data.get("methods") or {}).items():
            stats = m.get("stats") or {}
            mean = (stats.get("diversity_combined_mean") or {}).get("mean")
            if mean is None:
                continue
            if best_mean is None or mean < best_mean:
                best_mean, best_method = mean, k
        cfg = data.get("config") or {}
        runs.append({
            "run_id": child.name,
            "quantity": cfg.get("quantity"),
            "components": cfg.get("components"),
            "repeats": cfg.get("repeats"),
            "method_count": len(data.get("methods") or {}),
            "best_method": best_method,
            "best_mean": best_mean,
        })
    return runs


def _load_run(run_id: str) -> tuple[Path, dict]:
    run_dir = EVAL_ROOT / run_id
    agg = run_dir / "aggregated.json"
    if not agg.exists():
        abort(404, description=f"run_id {run_id} not found")
    return run_dir, json.loads(agg.read_text())


def _table_rows(data: dict) -> list[dict]:
    rows = []
    for k, m in (data.get("methods") or {}).items():
        stats = m.get("stats") or {}

        def cell(stat_key, scale=1.0, prec=3):
            stat = stats.get(stat_key)
            if not stat or stat.get("mean") is None:
                return None
            return {
                "mean": round(stat["mean"] * scale, prec),
                "std": round((stat.get("std") or 0.0) * scale, prec),
            }

        rows.append({
            "method": k,
            "is_llm": k.startswith("llm:"),
            "model": m.get("model"),
            "temperature": m.get("temperature"),
            "time_s": cell("time_to_quantity_seconds", prec=2),
            "reject_pct": cell("reject_rate", scale=100.0, prec=1),
            "diversity_combined": cell("diversity_combined_mean"),
            "diversity_topo": cell("diversity_topological_mean"),
            "diversity_geo": cell("diversity_geometric_mean"),
        })
    # Sort by combined diversity ascending (best first); None goes to bottom
    rows.sort(key=lambda r: (r["diversity_combined"] or {}).get("mean", float("inf")))
    return rows


def _temperature_summaries() -> list[dict]:
    if not TEMP_SWEEP_ROOT.exists():
        return []
    out = []
    for child in sorted(TEMP_SWEEP_ROOT.iterdir()):
        summary = child / "summary.json"
        if not summary.exists():
            continue
        try:
            data = json.loads(summary.read_text())
        except Exception:
            continue
        out.append({
            "model_key": data.get("model_key", child.name),
            "model": data.get("model"),
            "best_temperature": data.get("best_temperature"),
            "best_mean_similarity": data.get("best_mean_similarity"),
            "rows": data.get("rows", []),
            "summary_path": str(summary),
        })
    return out


# ---------------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )

    @app.route("/")
    def index():
        return render_template("index.html", runs=_list_runs())

    @app.route("/run/<run_id>")
    def run_detail(run_id: str):
        run_dir, data = _load_run(run_id)
        return render_template(
            "run.html",
            run_id=run_id,
            data=data,
            rows=_table_rows(data),
            plot_names=list(PLOTS.keys()),
        )

    @app.route("/run/<run_id>/plot/<plot_name>.png")
    def run_plot(run_id: str, plot_name: str):
        if plot_name not in PLOTS:
            abort(404)
        run_dir, _ = _load_run(run_id)
        agg_path = run_dir / "aggregated.json"
        # Render to a temp PNG inside the run dir's plots/ cache
        cache = run_dir / "plots"
        cache.mkdir(exist_ok=True)
        out = cache / f"{plot_name}.png"
        PLOTS[plot_name](agg_path, out)
        return send_file(out, mimetype="image/png", max_age=0)

    @app.route("/temperature")
    def temperature():
        summaries = _temperature_summaries()
        return render_template("temperature.html", summaries=summaries)

    @app.route("/temperature/plot.png")
    def temperature_plot():
        summary_paths = [s["summary_path"] for s in _temperature_summaries()]
        if not summary_paths:
            abort(404, description="no temperature sweep summaries found")
        out = TEMP_SWEEP_ROOT / "_combined_plot.png"
        plot_temperature_sweep(summary_paths, out)
        return send_file(out, mimetype="image/png", max_age=0)

    return app


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    create_app().run(host="127.0.0.1", port=port, debug=True)
