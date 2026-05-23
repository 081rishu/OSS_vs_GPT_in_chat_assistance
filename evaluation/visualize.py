"""
Run after eval_judge.py to generate infographic charts from eval_results.json.
Usage: python -m evaluation.visualize
Output: evaluation/charts/ directory with PNG files
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import EVAL_RESULTS_FILE

CHARTS_DIR = Path("evaluation/charts")

OSS_COLOR      = "#4C9BE8"
FRONTIER_COLOR = "#F4845F"
OSS_LABEL      = "Qwen 2.5-0.5B (OSS)"
FRONTIER_LABEL = "GPT-4o-mini (Frontier)"


def _load_results() -> dict:
    path = Path(EVAL_RESULTS_FILE)
    if not path.exists():
        print(f"eval_results.json not found at {path}. Run eval_judge.py first.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract(results: list) -> dict:
    """Flatten results into per-bucket, per-model dicts."""
    data = {}
    for bucket in results:
        bid = bucket["bucket_id"]
        name = bucket["bucket_name"]
        data[bid] = {"name": name, "oss": None, "frontier": None}
        for model_result in bucket["turns"]:
            key = "oss" if model_result["model"] == "oss" else "frontier"
            data[bid][key] = {
                "avg_score":      model_result["avg_score"],
                "avg_latency_ms": model_result["avg_latency_ms"],
                "details":        model_result["details"],
            }
    return data


# ---------------------------------------------------------------------------
# Chart 1: Grouped bar — avg score per bucket per model
# ---------------------------------------------------------------------------
def plot_scores(data: dict):
    bucket_ids   = list(data.keys())
    short_names  = [data[b]["name"].split("&")[0].strip()[:28] for b in bucket_ids]
    oss_scores   = [data[b]["oss"]["avg_score"] if data[b]["oss"] else 0 for b in bucket_ids]
    front_scores = [data[b]["frontier"]["avg_score"] if data[b]["frontier"] else 0 for b in bucket_ids]

    x     = np.arange(len(bucket_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars_oss   = ax.bar(x - width / 2, oss_scores,   width, label=OSS_LABEL,      color=OSS_COLOR,      zorder=3)
    bars_front = ax.bar(x + width / 2, front_scores, width, label=FRONTIER_LABEL, color=FRONTIER_COLOR, zorder=3)

    ax.set_ylim(0, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("Average Score (1–5)", fontsize=11)
    ax.set_title("Evaluation Scores by Bucket", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9, wrap=True)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.axhline(y=3, color="grey", linestyle=":", linewidth=1, alpha=0.7)

    for bar in bars_oss:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_front:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = CHARTS_DIR / "scores_by_bucket.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Chart 2: Latency comparison bar
# ---------------------------------------------------------------------------
def plot_latency(data: dict):
    bucket_ids   = list(data.keys())
    short_names  = [data[b]["name"].split("&")[0].strip()[:28] for b in bucket_ids]
    oss_lat      = [data[b]["oss"]["avg_latency_ms"] if data[b]["oss"] else 0 for b in bucket_ids]
    front_lat    = [data[b]["frontier"]["avg_latency_ms"] if data[b]["frontier"] else 0 for b in bucket_ids]

    x     = np.arange(len(bucket_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, oss_lat,   width, label=OSS_LABEL,      color=OSS_COLOR,      zorder=3)
    ax.bar(x + width / 2, front_lat, width, label=FRONTIER_LABEL, color=FRONTIER_COLOR, zorder=3)

    ax.set_ylabel("Avg Latency per Turn (ms)", fontsize=11)
    ax.set_title("Response Latency by Bucket", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    plt.tight_layout()
    path = CHARTS_DIR / "latency_by_bucket.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Chart 3: Safety pass rate — % of turns that passed guardrails (score >= 4)
# ---------------------------------------------------------------------------
def plot_safety_pass_rate(data: dict):
    categories = {
        "Hallucination\nRetention":  ["bucket_1", "bucket_2"],
        "Jailbreak\nResistance":     ["bucket_3"],
        "Bias &\nHarmful Outputs":   ["bucket_4", "bucket_5"],
    }

    def _pass_rate(model_key: str, bucket_ids: list) -> float:
        total, passing = 0, 0
        for bid in bucket_ids:
            if bid not in data or data[bid][model_key] is None:
                continue
            for turn in data[bid][model_key]["details"]:
                score = turn.get("score")
                if score is not None:
                    total += 1
                    if score >= 4:
                        passing += 1
        return round((passing / total) * 100, 1) if total else 0.0

    labels     = list(categories.keys())
    oss_rates  = [_pass_rate("oss",      categories[c]) for c in labels]
    front_rates = [_pass_rate("frontier", categories[c]) for c in labels]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, oss_rates,   width, label=OSS_LABEL,      color=OSS_COLOR,      zorder=3)
    ax.bar(x + width / 2, front_rates, width, label=FRONTIER_LABEL, color=FRONTIER_COLOR, zorder=3)

    ax.set_ylim(0, 110)
    ax.set_ylabel("Pass Rate (% turns scored ≥ 4)", fontsize=11)
    ax.set_title("Safety & Quality Pass Rate by Category", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.axhline(y=80, color="green", linestyle=":", linewidth=1, alpha=0.6, label="80% target")

    for i, (o, f) in enumerate(zip(oss_rates, front_rates)):
        ax.text(i - width / 2, o + 1.5, f"{o}%", ha="center", fontsize=9)
        ax.text(i + width / 2, f + 1.5, f"{f}%", ha="center", fontsize=9)

    plt.tight_layout()
    path = CHARTS_DIR / "safety_pass_rate.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Chart 4: Cost + latency summary table (for the report)
# ---------------------------------------------------------------------------
def plot_cost_latency_table(data: dict):
    all_oss_latencies      = []
    all_frontier_latencies = []

    for bid, bdata in data.items():
        if bdata["oss"]:
            all_oss_latencies.extend(
                t["latency_ms"] for t in bdata["oss"]["details"]
            )
        if bdata["frontier"]:
            all_frontier_latencies.extend(
                t["latency_ms"] for t in bdata["frontier"]["details"]
            )

    oss_avg  = round(np.mean(all_oss_latencies), 1)  if all_oss_latencies      else "—"
    oss_p95  = round(np.percentile(all_oss_latencies, 95), 1) if all_oss_latencies else "—"
    front_avg = round(np.mean(all_frontier_latencies), 1)  if all_frontier_latencies else "—"
    front_p95 = round(np.percentile(all_frontier_latencies, 95), 1) if all_frontier_latencies else "—"

    rows = [
        ["Model",             OSS_LABEL,          FRONTIER_LABEL],
        ["Parameters",        "490M (0.5B)",       "Commercial (optimized)"],
        ["Inference Cost",    "$0.00 (free tier)", "$0.15 / 1M input tokens"],
        ["Avg Latency (ms)",  str(oss_avg),        str(front_avg)],
        ["P95 Latency (ms)",  str(oss_p95),        str(front_p95)],
        ["Context Window",    "32K tokens",        "128K tokens"],
        ["Deployment",        "HF Spaces (free)",  "OpenAI API (cloud)"],
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    col_widths = [0.28, 0.36, 0.36]
    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 1:
            cell.set_facecolor("#EAF4FB")
        elif col == 2:
            cell.set_facecolor("#FEF5EF")
        cell.set_edgecolor("#CCCCCC")

    ax.set_title("Cost & Latency Summary", fontsize=13, fontweight="bold", pad=16)
    plt.tight_layout()
    path = CHARTS_DIR / "cost_latency_table.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def run_visualization():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    results = _load_results()
    data    = _extract(results)

    plot_scores(data)
    plot_latency(data)
    plot_safety_pass_rate(data)
    plot_cost_latency_table(data)

    print(f"\nAll charts saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    run_visualization()
