"""Cross-backend P2G NCU analysis figures.

Reads a ``nsight_results.json`` produced by a ``profile_nsight.py`` sweep with
``metric_preset=full derive_metric=full`` over the custom P2G backends, and
renders four figures: a hierarchical fp32 roofline, the atomic-scatter story,
occupancy + its limiter, and the warp-issue-stall breakdown.

    # 1. collect (single NCU sweep over the custom kernels)
    pixi run python profile_nsight.py -cn nsight_profile nsight.target=scatter \
        nsight.analyze.metric_preset=full nsight.analyze.derive_metric=full \
        'nsight.sweep.kernels=[cuda_v1,cuda_v2,cuda_v3,cuda_v4,cutile_v1,cutile_v3]'
    # 2. plot
    pixi run python postprocessing/nsight_plots.py <run_dir>/nsight_results.json -o figs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Canonical left->right order tracing the optimization arc; cuTile in a cool hue.
BACKEND_ORDER = ["cuda_v1", "cuda_v2", "cuda_v3", "cuda_v4", "cutile_v1", "cutile_v3"]
COLOR = {
    "cuda_v1": "#fdae6b",
    "cuda_v2": "#fd8d3c",
    "cuda_v3": "#e6550d",
    "cuda_v4": "#a63603",
    "cutile_v1": "#6baed6",
    "cutile_v3": "#2171b5",
}
LIMITER_NAME = {0: "registers", 1: "shared_mem", 2: "warps", 3: "blocks"}
# Stall reasons grouped global-mem -> shared-mem -> compute -> hidden(good).
STALLS = [
    ("long_scoreboard", "#d62728"),
    ("lg_throttle", "#ff7f0e"),
    ("short_scoreboard", "#1f77b4"),
    ("mio_throttle", "#17becf"),
    ("barrier", "#9467bd"),
    ("math_pipe_throttle", "#7f7f7f"),
    ("wait", "#bcbd22"),
    ("not_selected", "#2ca02c"),
]


def load(path):
    """Long-format JSON -> {backend: {metric: value}}."""
    rows = json.loads(Path(path).read_text())
    table = {}
    for r in rows:
        backend = r.get("variant_backend") or r.get("backend") or r.get("kernel")
        if backend is None:
            continue
        table.setdefault(backend, {})[r["Metric"]] = r["AvgValue"]
    return table


def _ordered(table):
    return [b for b in BACKEND_ORDER if b in table] or list(table)


def _first(table, backends, key):
    for b in backends:
        v = table[b].get(key)
        if v is not None:
            return v
    return None


def plot_roofline(table, out):
    import numpy as np

    backends = _ordered(table)
    peak_c = _first(table, backends, "peak_compute_gflops")
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ai = np.logspace(-2, 3, 300)
    for lvl, shade in (("l1", "#bbbbbb"), ("l2", "#888888"), ("hbm", "#222222")):
        bw = _first(table, backends, f"peak_{lvl}_gbps")
        if bw and peak_c:
            ax.plot(
                ai,
                np.minimum(bw * ai, peak_c),
                color=shade,
                lw=1.3,
                label=f"{lvl.upper()} roof ({bw / 1000:.1f} TB/s)",
            )
    if peak_c:
        ax.axhline(peak_c, color="black", ls="--", lw=0.8)
        ax.text(
            ai[-1], peak_c * 1.05, f"fp32 compute {peak_c / 1000:.1f} TFLOP/s",
            ha="right", va="bottom", fontsize=8,
        )
    markers = {"l1": "o", "l2": "s", "hbm": "^"}
    for b in backends:
        g = table[b].get("gflops_per_s")
        if not g:
            continue
        for lvl, mark in markers.items():
            x = table[b].get(f"ai_{lvl}_flop_per_byte")
            if x:
                ax.scatter(x, g, marker=mark, color=COLOR.get(b, "k"), s=45, zorder=3,
                           edgecolor="white", linewidth=0.4)
        x = table[b].get("ai_hbm_flop_per_byte")
        if x:
            ax.annotate(b, (x, g), fontsize=7, color=COLOR.get(b, "k"),
                        xytext=(4, 2), textcoords="offset points")
    handles = [plt.Line2D([], [], marker=m, ls="", color="gray", label=lvl.upper())
               for lvl, m in markers.items()]
    leg = ax.legend(loc="lower right", fontsize=7)
    ax.add_artist(leg)
    ax.legend(handles=handles, title="level", loc="upper left", fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity [FLOP/byte]")
    ax.set_ylabel("Performance [GFLOP/s]")
    ax.set_title("Hierarchical fp32 roofline — P2G scatter")
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _bar(ax, backends, values, colors, ylabel, title, ref=None, ref_label=None):
    xs = range(len(backends))
    ax.bar(xs, values, color=colors)
    if ref is not None:
        ax.axhline(ref, ls="--", color="0.4", lw=0.8)
        if ref_label:
            ax.text(0, ref, ref_label, fontsize=7, va="bottom", color="0.3")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(backends, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)


def plot_atomics(table, out):
    backends = _ordered(table)
    colors = [COLOR.get(b, "k") for b in backends]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _bar(
        axes[0], backends,
        [table[b].get("global_scatter_requests_per_particle", 0) for b in backends],
        colors, "global scatter requests / particle",
        "Atomic-scatter volume", ref=108.0, ref_label="108 (uncoalesced)",
    )
    lg = [table[b].get("stallr_lg_throttle", 0) for b in backends]
    mio = [table[b].get("stallr_mio_throttle", 0) for b in backends]
    import numpy as np

    x = np.arange(len(backends))
    axes[1].bar(x - 0.2, lg, 0.4, label="lg_throttle (global)", color="#ff7f0e")
    axes[1].bar(x + 0.2, mio, 0.4, label="mio_throttle (shared)", color="#17becf")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(backends, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("warps stalled / issue-active cycle")
    axes[1].set_title("Contention fingerprint: global -> shared", fontsize=10)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_occupancy(table, out):
    backends = _ordered(table)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = range(len(backends))
    theo = [table[b].get("theoretical_occ_pct", 0) for b in backends]
    ach = [table[b].get("active_warps_pct", 0) for b in backends]
    ax.bar(xs, theo, color="none", edgecolor="0.5", lw=1.2, label="theoretical")
    ax.bar(xs, ach, color=[COLOR.get(b, "k") for b in backends], label="achieved")
    for i, b in enumerate(backends):
        code = table[b].get("occ_limiter_code")
        regs = table[b].get("regs_per_thread")
        if code is not None:
            lbl = LIMITER_NAME.get(int(code), "?")
            note = f"[{lbl}]" + (f"\nR={int(regs)}" if regs else "")
            ax.annotate(note, (i, max(theo[i], ach[i])), fontsize=7, ha="center",
                        va="bottom", xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(backends, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("occupancy (% of 64 warps/SM)")
    ax.set_ylim(0, 105)
    ax.set_title("Occupancy (achieved vs theoretical; limiter annotated)", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_scheduler(table, out):
    import numpy as np

    backends = _ordered(table)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(backends))
    left = np.zeros(len(backends))
    for reason, color in STALLS:
        vals = np.array([table[b].get(f"stallr_{reason}", 0.0) for b in backends])
        ax.barh(y, vals, left=left, color=color, label=reason)
        left += vals
    for i, b in enumerate(backends):
        issue = table[b].get("issue_active_pct")
        if issue is not None:
            ax.text(left[i], i, f"  issue {issue:.0f}%", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(backends, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("warps stalled / issue-active cycle (sum of reasons)")
    ax.set_title("Warp-issue-stall breakdown", fontsize=10)
    ax.legend(fontsize=6, ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_json", type=Path)
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("nsight_figs"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = load(args.results_json)
    plot_roofline(table, args.out_dir / "roofline.png")
    plot_atomics(table, args.out_dir / "atomics.png")
    plot_occupancy(table, args.out_dir / "occupancy.png")
    plot_scheduler(table, args.out_dir / "scheduler.png")
    print(f"wrote roofline/atomics/occupancy/scheduler .png to {args.out_dir}")


if __name__ == "__main__":
    main()
