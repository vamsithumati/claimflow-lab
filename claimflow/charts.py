"""Charts for the post, drawn from lab results (not illustrative numbers)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from .data import CATEGORIES  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
FOOT = "Source: ClaimFlow reference lab (synthetic insurance-claims data, seed-reproducible)"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "text.color": INK, "axes.edgecolor": BASE,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def _title(fig, title: str, sub: str) -> None:
    fig.text(0.02, 0.93, title, fontsize=19, fontweight="bold")
    fig.text(0.02, 0.87, sub, fontsize=13, color=INK2)
    fig.text(0.02, 0.02, FOOT, color=MUTED, fontsize=10.5)


def funnel(e1: dict, out: Path) -> None:
    p = e1["base"]["pick_to_accept"]
    rows = [("AI suggestions", p["drafts"], SERIES[0]), ("Overridden by a human", p["overrides"], SERIES[0]),
            ("  ↳ Real corrections", p["true_corrections"], SERIES[1]),
            ("  ↳ Re-picked the AI's answer", p["confirm_as_is"], BASE)]
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=150)
    fig.subplots_adjust(left=0.30, right=0.95, top=0.80, bottom=0.14)
    for i, (_, v, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, height=0.62, color=c, edgecolor=SURFACE, linewidth=2)
        ax.text(v + p["drafts"] * 0.012, y, f"{v:,}", va="center", fontweight="bold")
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_xlim(0, p["drafts"] * 1.12)
    ax.xaxis.set_visible(False)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    _title(fig, f"A {p['naive_override_rate']:.0%} override rate hides a {p['true_correction_rate']:.0%} correction rate",
           f"{p['confirm_as_is_share']:.0%} of overrides re-selected the AI's own answer (pick-to-accept interface)")
    fig.savefig(out / "01-override-funnel.png", facecolor=SURFACE)
    plt.close(fig)


def bounds(e2: dict, out: Path) -> None:
    b, a = e2["bounds"], e2["audit"]["adjudicated"]
    lo, hi, tr = 100 * b["lower_bound"], 100 * b["reviewed_subset_error"], 100 * b["oracle_true_error"]
    est, hw = 100 * a["estimate"]["mean"], a["ci_half_width_pp"]
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=150)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.70, bottom=0.24)
    ax.set_xlim(0, max(60, hi + 8))
    ax.set_ylim(-1, 1)
    ax.yaxis.set_visible(False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, 10), [f"{t}%" for t in range(0, int(ax.get_xlim()[1]) + 1, 10)])
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.plot([lo, hi], [0, 0], color=BASE, lw=10, solid_capstyle="round", zorder=1)
    ax.errorbar(est, 0, xerr=[[hw], [hw]], fmt="none", ecolor=SERIES[0], elinewidth=3, capsize=8, zorder=2)
    ax.axvline(tr, color=INK, lw=1.2, ls=(0, (3, 3)), zorder=1)
    ax.text(tr, 0.88, f"true error {tr:.0f}% (lab only)", ha="center", fontsize=12, color=INK)
    for x, c, lab, dy in [(lo, INK2, f"Lower bound {lo:.0f}%\ncorrections ÷ all suggestions", -1),
                          (hi, INK2, f"Reviewed-subset error {hi:.0f}%\npeople open what looks wrong", -1),
                          (est, SERIES[0], f"Blind audit {est:.0f}% ± {hw:.0f}pp", 1)]:
        ax.scatter([x], [0], s=170, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        if dy > 0:  # audit label sits right of its interval, clear of the truth line
            ax.text(est + hw + 1.2, 0.30, lab, ha="left", va="bottom", fontsize=12.5, color=INK, fontweight="bold")
            continue
        ax.text(x, -0.36, lab, ha="center", va="top", fontsize=12.5, color=INK2)
    _title(fig, "Accuracy is a range until you audit what nobody reviewed",
           "AI error rate: the dashboard's two candidate numbers vs. a 4-week blind stratified audit")
    fig.savefig(out / "02-error-bounds.png", facecolor=SURFACE)
    plt.close(fig)


def confusion(e3: dict, out: Path) -> None:
    conf = e3["keyword_router"]["confusion"]
    M = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype=int)
    for p in conf["pairs"]:
        M[CATEGORIES.index(p["ai"]), CATEGORIES.index(p["human"])] = p["n"]
    labs = [c.replace("_", " ").capitalize() for c in CATEGORIES]
    ramp = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
    vmax = max(int(M.max()), 61)
    norm = BoundaryNorm([0, 1, 3, 6, 12, 25, 60, vmax + 1], len(ramp))
    fig, ax = plt.subplots(figsize=(11.5, 8.8), dpi=150)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.80, bottom=0.20)
    ax.imshow(M, cmap=ListedColormap(ramp), norm=norm)
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            v = M[i, j]
            txt = "—" if i == j else str(v)
            ax.text(j, i, txt, ha="center", va="center", fontsize=14,
                    color="#ffffff" if v >= 25 else (MUTED if i == j else INK2),
                    fontweight="bold" if v >= 25 else "normal")
    ax.set_xticks(range(len(labs)), labs, rotation=30, ha="right", fontsize=13)
    ax.set_yticks(range(len(labs)), labs, fontsize=13)
    ax.set_xticks(np.arange(-.5, len(labs)), minor=True)
    ax.set_yticks(np.arange(-.5, len(labs)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel("AI picked", fontsize=13, labelpad=10)
    ax.set_xlabel("Human corrected to", fontsize=13, labelpad=10)
    _title(fig, "Errors cluster in look-alike pairs",
           f"Real corrections only — two twin pairs account for {conf['twin_pair_share']:.0%} of {conf['total']} corrections")
    fig.savefig(out / "03-twin-confusion.png", facecolor=SURFACE)
    plt.close(fig)


def frontier(e3: dict, out: Path) -> None:
    from .claims import STRATEGIES
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.80, bottom=0.14)
    for (name, (clf, key)), color in zip(STRATEGIES.items(), SERIES):
        pts = sorted((p["hold_rate"] * 100, p["auto_passed_error_rate"] * 100) for p in e3[clf][key])
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, lw=2.2, marker="o", ms=8, mec=SURFACE, mew=2, label=name)
    ax.axvline(33, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(33.8, ax.get_ylim()[1] * 0.95, "33% review budget", color=MUTED, fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.set_xlabel("Share of documents held for a human (workload)")
    ax.set_ylabel("Errors that pass automatically")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.legend(loc="upper right", frameon=False, fontsize=12.5)
    _title(fig, "Fix the confidence before you move the threshold",
           "Same review workload, very different error rates — lower-left is better")
    fig.savefig(out / "04-workload-vs-errors.png", facecolor=SURFACE)
    plt.close(fig)


def all_charts(results: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    funnel(results["e1"], out)
    bounds(results["e2"], out)
    confusion(results["e3"], out)
    frontier(results["e3"], out)
