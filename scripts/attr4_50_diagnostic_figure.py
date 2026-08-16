"""Figure: the post-call 'partner approaches' effect is regression to the mean.

v4_fig5_consequence_retracted
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A
from attr4_20_figures import PRETTY, pretty, savefig

C_CALL, C_CTRL = "#1baf7a", "#898781"
C_POS, C_NEG = "#2a78d6", "#e34948"
C_INK, C_SEC, C_MUT = "#0b0b0b", "#52514e", "#898781"
C_GRID, C_SURF, C_BASE = "#e1e0d9", "#fcfcfb", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": C_SURF, "axes.facecolor": C_SURF,
    "savefig.facecolor": C_SURF, "axes.edgecolor": C_BASE,
    "axes.labelcolor": C_SEC, "text.color": C_INK,
    "xtick.color": C_MUT, "ytick.color": C_MUT,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "figure.titlesize": 14, "figure.titleweight": "bold", "svg.fonttype": "none",
})
OUT = A.ensure_out()


def main():
    traj = pd.read_csv(OUT / "beh_diag_traj.csv")
    bal = pd.read_csv(OUT / "beh_diag_balance.csv")
    ctx = pd.read_csv(OUT / "beh_diag_context.csv")
    rt = pd.read_csv(OUT / "beh_diag_retest.csv")
    S = json.load(open(OUT / "beh_diag_summary.json"))

    fig = plt.figure(figsize=(15.0, 8.8))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.36)

    # (a) absolute trajectories - the decisive panel
    ax = fig.add_subplot(gs[0, 0])
    t = traj[traj["metric"] == "app_m2"].sort_values("lag_s")
    for col, c, lab in (("call", C_CALL, "calls"),
                        ("ctrl", C_CTRL, "matched no-call moments")):
        ax.plot(t["lag_s"], t[f"{col}_mean"], color=c, lw=2, label=lab)
        ax.fill_between(t["lag_s"], t[f"{col}_mean"] - t[f"{col}_sem"],
                        t[f"{col}_mean"] + t[f"{col}_sem"], color=c, alpha=0.22,
                        edgecolor="none")
    ax.axvline(0, color=C_INK, lw=1.2, ls="--")
    ax.axhline(0, color=C_BASE, lw=1)
    ax.set_xlabel("time from event (s)")
    ax.set_ylabel("partner approach velocity (px/s)\n+ = moving toward resident")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("(a)  The call trace starts far LOWER and\nmerely converges — "
                 "regression to the mean", loc="left")

    # (b) balance
    ax = fig.add_subplot(gs[0, 1])
    b = bal.sort_values("smd")
    y = np.arange(len(b))
    ax.barh(y, b["smd"], color=[C_NEG if abs(v) > 0.1 else C_MUT
                                for v in b["smd"]])
    for x in (-0.1, 0.1):
        ax.axvline(x, color=C_INK, lw=1, ls=":")
    ax.axvline(0, color=C_INK, lw=1)
    ax.text(0.115, len(b) - 0.4, "|SMD| > 0.1\n= imbalanced", fontsize=7.5,
            color=C_NEG, va="top")
    ax.set_yticks(y, [pretty(m) for m in b["metric"]], fontsize=8)
    ax.set_xlabel("standardised mean difference, pre-call window")
    ax.set_title("(b)  The matching FAILED on the outcome:\ncalls start from a "
                 "more extreme state", loc="left")

    # (c) context: partner approach velocity per behaviour
    ax = fig.add_subplot(gs[0, 2])
    c = ctx[ctx["calls"] >= 15].sort_values("app_m2_in")
    y = np.arange(len(c))
    ax.barh(y, c["app_m2_in"],
            color=[C_NEG if v < 0 else C_POS for v in c["app_m2_in"]])
    ax.axvline(0, color=C_INK, lw=1)
    ax.set_yticks(y, [f"{PRETTY.get(b_, b_)} (n={int(n)})"
                      for b_, n in zip(c["behavior"], c["calls"])], fontsize=7.5)
    ax.set_xlabel("partner approach velocity while resident is in state (px/s)")
    ax.set_title("(c)  Calls happen where the partner is\nRETREATING fastest",
                 loc="left")

    # (d,e) re-test forests
    for k, (metric, title) in enumerate((
            ("app_m2", "(d)  Partner approach velocity"),
            ("speed_m1", "(e)  Caller speed"))):
        ax = fig.add_subplot(gs[1, k])
        r = rt[rt["metric"] == metric].reset_index(drop=True)
        labels = [f"{m.replace('state+speed+dist', 'base')}\n{s}"
                  for m, s in zip(r["matching"], r["subset"])]
        y = np.arange(len(r))
        for i, row in r.iterrows():
            ax.plot([row["boot_lo"], row["boot_hi"]], [i, i], color=C_MUT, lw=1.8)
            ax.scatter([row["pooled_diff"]], [i], s=38, zorder=3,
                       c=C_POS if row["sig"] else C_MUT,
                       edgecolors="white", linewidths=0.5)
            if row["sig"]:
                ax.text(row["boot_hi"], i, "  *", va="center", fontsize=12)
        ax.axvline(0, color=C_INK, lw=1.2)
        ax.set_yticks(y, labels, fontsize=7)
        ax.set_xlabel("call − matched control (px/s)")
        ax.set_title(f"{title}\nsurvives only with the loose matching", loc="left")

    # (f) verdict
    ax = fig.add_subplot(gs[1, 2])
    ax.set_axis_off()
    txt = (
        "RETRACTION\n\n"
        "Earlier claim: 'after a call the partner\n"
        "approaches and the caller slows.'\n\n"
        "That was REGRESSION TO THE MEAN.\n\n"
        f"  pre-call partner approach\n"
        f"    at calls    {S['balance']['app_m2']:+.2f} SMD vs controls\n"
        f"  during anogenital sniffing the partner\n"
        f"    retreats at {S['app_m2_during_anogenital']:.0f} px/s\n"
        f"    (vs {S['app_m2_elsewhere']:.0f} px/s elsewhere)\n\n"
        "Calls are emitted at kinematic extremes,\n"
        "which must relax afterwards whatever the\n"
        "call does. Matching on the baseline, or\n"
        "using isolated calls, removes the effect\n"
        "entirely (see d, e).\n\n"
        "HONEST CONCLUSION: this dataset shows\n"
        "WHERE calls occur, but does NOT show a\n"
        "measurable consequence of calling."
    )
    ax.text(0.02, 0.98, txt, va="top", ha="left", fontsize=8.6,
            family="monospace", color=C_INK,
            bbox=dict(boxstyle="round,pad=0.6", fc="#faeceb", ec=C_NEG))
    ax.set_title("(f)  Verdict", loc="left")

    fig.suptitle("Does calling change behaviour? Testing the post-call effect "
                 "against regression to the mean", y=1.0)
    savefig(fig, "v4_fig5_consequence_retracted")


if __name__ == "__main__":
    main()
