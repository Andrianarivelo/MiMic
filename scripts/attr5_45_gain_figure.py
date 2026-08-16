"""Figure 5: what "gain" means, and whether one loud animal carries the result.

Panel (a) states the two competing hypotheses geometrically, because on a log
axis they are told apart by SHAPE rather than by level: a uniform volume change
keeps the two genotype curves PARALLEL, whereas a lost gain leaves HET flat so
the gap widens with engagement. Panel (b) shows which one the data pick.
Panels (c) and (d) answer the separate question of whether animal 31078, which
supplies 39% of all WT calls, is doing the work.
"""
from __future__ import annotations

import json
import sys
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import attr_common as A
from attr5_40_figures import (WT, HET, INK, INK2, INK3, GRID, NULLC, CRIT,
                              tidy, save)

BIG = "31078"                       # the heavy caller under scrutiny
STATE_LAB = {"none": "nothing\nsocial",
             "solitary": "solitary\n(rear, passive)",
             "withdraw": "withdrawal,\nescape",
             "investigate": "investigating\nthe partner",
             "contact": "social\ncontact"}
ORDER = ["none", "solitary", "withdraw", "investigate", "contact"]


def main():
    out = A.ensure_out()
    rob = json.load(open(out / "v5_robustness.json"))
    why = json.load(open(out / "v5_why_het.json"))
    sup = json.load(open(out / "v5_supplement.json"))
    per = pd.read_csv(out / "v5_gain_per_session.csv",
                      dtype={"animal_id": str})

    fig = plt.figure(figsize=(12.6, 7.6))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.30,
                          height_ratios=[1, 1.05])

    # --- (a) the two hypotheses, side by side -----------------------------
    sub = gs[0, 0].subgridspec(1, 2, wspace=0.22)
    x = np.linspace(0.06, 0.94, 40)
    wt_line = 0.75 + 1.75 * x
    panels = [(wt_line - 1.05, "if it were a volume knob",
               "gap CONSTANT\nat every level", INK3),
              (np.full_like(x, 0.60), "what the data show",
               "gap WIDENS with\nengagement", HET)]
    for j, (het_line, ttl, note, col) in enumerate(panels):
        a2 = fig.add_subplot(sub[0, j])
        a2.plot(x, wt_line, color=WT, lw=2.4)
        a2.plot(x, het_line, color=HET, lw=2.4)
        a2.annotate("WT", (x[-1], wt_line[-1]), xytext=(-2, 5),
                    textcoords="offset points", color=WT, fontweight="bold",
                    fontsize=8.5, ha="right")
        a2.annotate("HET", (x[-1], het_line[-1]), xytext=(-2, -13),
                    textcoords="offset points", color=HET, fontweight="bold",
                    fontsize=8.5, ha="right")
        # vertical double arrows at both ends make the gap comparison explicit
        for xv in (0.13, 0.87):
            i = int(np.argmin(np.abs(x - xv)))
            a2.annotate("", xy=(x[i], wt_line[i]), xytext=(x[i], het_line[i]),
                        arrowprops=dict(arrowstyle="<->", color=col, lw=1.1))
        a2.annotate(note, (0.5, 0.035), xycoords="axes fraction", ha="center",
                    fontsize=7.3, color=col, fontweight="bold")
        a2.set_xticks([0.06, 0.94])
        a2.set_xticklabels(["low", "high"], fontsize=7.5)
        a2.set_yticks([])
        a2.set_ylim(-0.35, 2.75)
        a2.set_xlabel("social engagement", fontsize=7.8)
        if j == 0:
            a2.set_ylabel("log call rate", fontsize=8)
        a2.set_title(ttl, loc="left", fontsize=8.4)
        tidy(a2, None)
    fig.text(gs[0, 0].get_position(fig).x0,
             gs[0, 0].get_position(fig).y1 + 0.045,
             "a   Two hypotheses, told apart by SHAPE not level",
             fontsize=9.5, fontweight="bold", color=INK, ha="left")

    # --- (b) which hypothesis the data pick -------------------------------
    ax = fig.add_subplot(gs[0, 1])
    rows = {r["state"]: r for r in rob["state_ratios"]["all 12 WT"]}
    rows_no = {r["state"]: r for r in rob["state_ratios"][f"WT minus {BIG}"]}
    xs = np.arange(len(ORDER))
    v_all = [rows[s]["ratio"] for s in ORDER]
    v_no = [rows_no[s]["ratio"] for s in ORDER]
    ax.bar(xs - 0.19, v_all, width=0.36, color=WT, lw=0, label="all 12 WT")
    ax.bar(xs + 0.19, v_no, width=0.36, color=NULLC, lw=0,
           label=f"WT without {BIG}")
    ax.axhline(why["kitagawa"]["R_WT"] / why["kitagawa"]["R_HET"],
               color=CRIT, lw=1.1, ls=(0, (4, 3)))
    ax.annotate("overall 11.9x", (-0.45, 12.8), ha="left", va="bottom",
                fontsize=7.4, color=CRIT)
    for i, (a_, b_) in enumerate(zip(v_all, v_no)):
        ax.annotate(f"{a_:.0f}x", (i - 0.19, a_), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.2,
                    color=INK2)
        ax.annotate(f"{b_:.0f}x", (i + 0.19, b_), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.2,
                    color=INK3)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([STATE_LAB[s] for s in ORDER], fontsize=7.4)
    ax.set_ylabel("WT / HET call rate in that state")
    ax.set_ylim(1.6, 320)
    ax.set_title("b   The deficit grows with how social the state is",
                 loc="left")
    ax.legend(loc="upper left", fontsize=7.6)
    ax.annotate("not a volume knob: the gap widens ~14x from left to right",
                (0.5, -0.24), xycoords="axes fraction", ha="center", va="top",
                fontsize=7.5, color=INK2)
    tidy(ax)

    # --- (c) per-session gain: one value per animal -----------------------
    ax = fig.add_subplot(gs[1, 0])
    rng = np.random.default_rng(5)
    for j, (gt, col) in enumerate((("WT", WT), ("HET", HET))):
        d = per[per.genotype == gt]
        xx = np.full(len(d), j) + rng.normal(0, 0.055, len(d))
        ax.scatter(xx, d["gain"], s=46, color=col, alpha=0.8, lw=0, zorder=3)
        ax.plot([j - 0.26, j + 0.26], [d["gain"].median()] * 2, color=INK,
                lw=2.2, zorder=4)
        ax.annotate(f"median {d['gain'].median():.2f}x",
                    (j, d["gain"].median()), xytext=(0, -18),
                    textcoords="offset points", ha="center", fontsize=7.8,
                    color=INK, fontweight="bold")
        if gt == "WT" and BIG in set(d.animal_id):
            i = list(d.animal_id).index(BIG)
            gval = d[d.animal_id == BIG]["gain"].iloc[0]
            ax.scatter([xx[i]], [gval], s=150, facecolors="none",
                       edgecolors=INK, lw=1.6, zorder=5)
            ax.annotate(f"{BIG}: 39% of all WT calls,\nbut only the 3rd "
                        "LOWEST gain\nof the 12 WT animals",
                        (xx[i], gval), xytext=(30, -6),
                        textcoords="offset points", fontsize=7.4, color=INK,
                        va="top",
                        arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8))
    ax.axhline(1, color=CRIT, lw=1.1, ls=(0, (4, 3)))
    ax.annotate("dashed line = 1x: calls just as often, social or not",
                (0.985, 0.025), xycoords="axes fraction", ha="right",
                fontsize=7.0, color=CRIT)
    g = rob["per_session_gain"]
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["WT", "HET"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("social gain of that animal\n(call rate social / non-social)")
    ax.set_title("c   Every animal contributes exactly one value", loc="left")
    ax.annotate(f"{g['WT_n_above_1']}/12 WT animals above 1x; "
                f"{g['HET_n_above_1']}/12 HET\n"
                f"Mann-Whitney p = {g['p']:.4f}\n"
                f"dropping {BIG}: median {g['WT_median_drop31078']:.2f}x, "
                f"p = {g['p_drop31078']:.4f}",
                (0.97, 0.96), xycoords="axes fraction", ha="right", va="top",
                fontsize=7.5, color=INK2)
    tidy(ax)

    # --- (d) the drop-31078 ledger ----------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    items = [
        ("WT social gain", f"{rob['pooled_gain']['WT all 12']['gain']:.2f}x",
         f"{rob['pooled_gain'][f'WT without {BIG}']['gain']:.2f}x"),
        ("gain ratio WT / HET", f"{why['social_gain']['gain_ratio']:.2f}x",
         f"{rob['gain_ratio_drop31078']['ratio']:.2f}x"),
        ("propensity share of the gap",
         f"{why['kitagawa']['pct_propensity']:.0f}%",
         f"{rob['kitagawa_drop31078']['pct_propensity']:.0f}%"),
        ("engagement-ladder rho (WT)", f"{sup['ladder']['WT']['rho']:+.2f}",
         f"{rob['ladder'][f'WT minus {BIG}']['rho']:+.2f}"),
        ("distance slope (WT)", f"{why['gain']['slope_WT']:+.2f}",
         f"{rob['distance_slope_drop31078']:+.2f}"),
        ("session call rate, WT vs HET",
         f"p = {why['escalation']['p_rate']:.4f}",
         f"p = {rob['rate_drop31078']['p']:.4f}"),
    ]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y = 0.87
    ax.text(0.0, 0.98, f"d   Every headline number, recomputed without {BIG}",
            fontsize=9.5, fontweight="bold", color=INK,
            transform=ax.transAxes)
    ax.text(0.575, y + 0.055, "all 12 WT", fontsize=7.4, color=INK3,
            ha="center", fontweight="bold")
    ax.text(0.86, y + 0.055, f"without {BIG}", fontsize=7.4, color=INK3,
            ha="center", fontweight="bold")
    ax.plot([0, 1], [y + 0.035, y + 0.035], color=INK, lw=1.1)
    for lab, a_, b_ in items:
        ax.text(0.0, y - 0.035, lab, fontsize=8.2, color=INK, va="center")
        ax.text(0.575, y - 0.035, a_, fontsize=8.6, color=INK2, va="center",
                ha="center", family="monospace")
        ax.text(0.86, y - 0.035, b_, fontsize=8.6, color=WT, va="center",
                ha="center", family="monospace", fontweight="bold")
        y -= 0.148
        ax.plot([0, 1], [y + 0.048, y + 0.048], color=GRID, lw=0.8)
    ax.text(0.0, y - 0.005,
            "The gain is a WITHIN-animal ratio, so an animal that simply "
            "calls a lot cannot\ninflate it. Removing 31078 makes every one "
            "of these numbers stronger or\nleaves it unchanged.",
            fontsize=7.6, color=INK2, va="top")

    fig.suptitle("What “gain” means, and whether one loud animal is "
                 "carrying it",
                 x=0.011, y=1.005, ha="left", fontsize=12.5, fontweight="bold")
    save(fig, "v5_fig5_gain_robustness")


if __name__ == "__main__":
    main()
