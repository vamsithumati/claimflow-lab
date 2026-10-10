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
    p = e1["representative"]["pick_to_accept"]
    rows = [("AI suggestions", p["drafts"], SERIES[0]), ("Overridden by a human", p["overrides"], SERIES[0]),
            ("  ↳ Real corrections", p["true_corrections"], SERIES[1]),
            ("  ↳ Re-picked the AI's answer", p["confirm_as_is"], BASE),
            ("  ↳ Bulk-set to the AI's answer", p["bulk_set"], BASE)]
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=150)
    fig.subplots_adjust(left=0.31, right=0.95, top=0.80, bottom=0.12)
    for i, (_, v, c) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, v, height=0.62, color=c, edgecolor=SURFACE, linewidth=2)
        ax.text(v + p["drafts"] * 0.012, y, f"{v:,}", va="center", fontweight="bold")
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_xlim(0, p["drafts"] * 1.12)
    ax.xaxis.set_visible(False)
    for s_ in ("top", "right", "bottom"):
        ax.spines[s_].set_visible(False)
    _title(fig, f"A {p['naive_override_rate']:.0%} override rate, a {p['true_correction_rate']:.0%} correction rate",
           f"{p['not_a_correction_share']:.0%} of overrides changed nothing (median of 20 runs) — illustrates the mechanism, not a measured rate")
    fig.savefig(out / "01-override-funnel.png", facecolor=SURFACE)
    plt.close(fig)


def bounds(e2: dict, out: Path) -> None:
    b, a = e2["bounds"], e2["headline_audit"]
    lo, hi, tr = 100 * b["lower_bound"], 100 * b["reviewed_subset_error"], 100 * b["oracle_true_error"]
    est, hw = 100 * a["estimate"], a["ci_half_width_pp"]
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
                          (hi, INK2, f"Reviewed-subset error {hi:.0f}%\nnot a bound — depends on why items get opened", -1),
                          (est, SERIES[0], f"Blind audit {est:.0f}% ± {hw:.0f}pp", 1)]:
        ax.scatter([x], [0], s=170, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        if dy > 0:  # audit label sits right of its interval, clear of the truth line
            ax.text(est + hw + 1.2, 0.30, lab, ha="left", va="bottom", fontsize=12.5, color=INK, fontweight="bold")
            continue
        ax.text(x, -0.36, lab, ha="center", va="top", fontsize=12.5, color=INK2)
    _title(fig, "Neither dashboard number is your error rate",
           "A lower bound, a selection-biased rate, and a 4-week blind stratified audit (double-labelled, reviewed items re-audited)")
    fig.savefig(out / "02-error-bounds.png", facecolor=SURFACE)
    plt.close(fig)


def confusion(e1: dict, out: Path) -> None:
    conf = e1["confusion"]
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
           f"Real corrections — {conf['twin_pair_share']:.0%} fall in two look-alike pairs (the lab generates twins, so this is illustrative)")
    fig.savefig(out / "03-twin-confusion.png", facecolor=SURFACE)
    plt.close(fig)


def frontier(e3: dict, out: Path) -> None:
    shown = ["match-strength, one global threshold", "match-strength, threshold on look-alike families only",
             "match-strength + look-alike evidence check", "classifier, 3,000 labels"]
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.80, bottom=0.14)
    for name, color in zip(shown, SERIES):
        pts = sorted((p["hold_rate"] * 100, p["auto_passed_error_rate"] * 100) for p in e3["curves"][name])
        xs, ys = zip(*pts)
        ax.step(xs, ys, where="post", color=color, lw=2.2, label=name)  # deterministic operating points
    ax.axvline(100 / 3, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(34.2, 19.5, "⅓ review budget", color=MUTED, fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 21)
    ax.set_xlabel("Share of documents held for a human (workload)")
    ax.set_ylabel("Errors that pass automatically")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=12.5)
    _title(fig, "Same workload, different errors — treat look-alikes differently",
           f"Seed {e3['curves_seed']}, every threshold; lower-left is better.\n"
           "Classifier: 3,000 labels. Rules: no labels, but built from the generator's own vocabulary (an upper bound).")
    fig.savefig(out / "04-workload-vs-errors.png", facecolor=SURFACE)
    plt.close(fig)


def all_charts(results: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    funnel(results["e1"], out)
    bounds(results["e2"], out)
    confusion(results["e1"], out)
    frontier(results["e3"], out)
