"""Presentation-grade v4 figure with a per-session count view in panel A.

This is a redesign, not a re-run: every number comes from the same analysis as
`attr4_20_figures.py::fig1` (`attr4_10_analysis.py` outputs), but the figure is
rebuilt around a single declarative claim and stripped of the low-information
bars that dominated the original panel (a).

What changes, and why
---------------------
(a) CALL COUNTS    seven atomic behaviours are shown as horizontal WT versus HET
                   box-and-strip distributions. One animal is one independent
                   observation (12 WT and 12 HET). Inference uses a two-way mixed
                   ANOVA on log1p counts, with genotype between animals and
                   behaviour repeated within animals, followed by Bonferroni-
                   corrected simple effects and within-genotype comparisons.

(b) ACTION x REACTION  old panels (a) and (b) are merged into one 2 x 2 contingency
                   of *raw* call rate: resident investigating (anogenital sniff or
                   following) x partner retreating (withdrawal / escape /
                   withdrawal-after-contact). This answers the two-sided question
                   in one object instead of two facing bar charts.

(c) ACTOR vs TARGET  the same six directional behaviours as the original panel (c),
                   redrawn as dumbbells on the multiplier axis.

(d) EXEMPLAR       new: a real 4.5 s excerpt - three video frames from the arena
                   above the matching 384 kHz spectrogram, with the detected calls
                   and the behaviour state ribbon drawn on the same time axis. The
                   quantitative claim is grounded in one visible episode.

The validation panels of the original (LOSO AUC, split-half reliability) are
demoted to a compact stat strip: they are reassurance, not the message.

Run
---
    python scripts/attr4_22_hero_figure.py

Outputs `v4_fig1_calls_by_genotype.{png,svg,pdf}`, a visual benchmark against
the pooled-count version, and the mixed-ANOVA/post-hoc statistical tables.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch, Patch,
                                Rectangle)
from scipy import signal, stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import attr_common as A

BIN_S = 0.1

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
PARTNER_T0 = 305.0          # first 5 s after partner introduction are dropped
N_BOOT = 4000               # cluster-bootstrap replicates (over sessions)
N_STRATA = 9                # 3 speed terciles x 3 distance terciles
SEED = A.RNG_SEED

# Inclusion rule for the hero panel, fixed before looking at the estimates:
# a behaviour must occupy at least MIN_OCC_PCT of the interaction window and
# host at least MIN_CALLS calls, otherwise its rate ratio is dominated by noise.
MIN_OCC_PCT = 3.0
MIN_CALLS = 100

# The two "star" behaviours pinned to the top of the hero panel.
STARS = ("nose2anogenital", "following")
# The composite that contains them; shown in a lighter accent, never as a star.
COMPOSITE = "ACTIVEsocial"

# 2 x 2 action-reaction definition
RES_INVESTIGATE = ("m1_nose2anogenital", "m1_following")
PART_RETREAT = ("m2_withdrawal_from_partner", "m2_escape",
                "m2_withdrawal_after_contact")

# Six directional behaviours for the actor-vs-target dumbbells
DIRECTIONAL = ("nose2anogenital", "nose2body", "oriented_toward",
               "following", "chasing", "approach")

# Representative episode (chosen for: sustained investigation -> following, a
# dense call bout, and both animals well tracked and visually separable).
EXEMPLAR = {
    "animal": "31078",
    "t0": 360.0,
    "t1": 364.5,
    "frames": (360.30, 362.15, 363.55),
    "frame_labels": ("anogenital sniff", "following", "following"),
}

# --------------------------------------------------------------------------- #
# Palette                                                                     #
# --------------------------------------------------------------------------- #
C_ACCENT = "#1f6fd0"        # the two star behaviours
C_ACCENT_LT = "#7fb0e8"     # the composite that contains them
C_MUTE = "#b6b4ab"          # every other behaviour
C_MUTE_D = "#8d8b83"        # muted, below-baseline
C_RES = "#1baf7a"           # resident is the actor
C_PART = "#4a3aa7"          # partner is the actor
C_INK, C_SEC, C_MUT = "#111110", "#4f4e4a", "#87857f"
C_GRID, C_SURF, C_BASE = "#e4e3dc", "#fcfcfb", "#c9c8bd"
C_HOT = "#ff4d3d"           # exemplar call marks
C_PARTNER_MARK = "#f2b134"  # partner ring on the video stills
FRAME_AR = 4 / 3            # aspect ratio of the cropped video stills

plt.rcParams.update({
    "figure.facecolor": C_SURF, "axes.facecolor": C_SURF,
    "savefig.facecolor": C_SURF, "axes.edgecolor": C_BASE,
    "axes.labelcolor": C_SEC, "text.color": C_INK,
    "xtick.color": C_MUT, "ytick.color": C_MUT,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "figure.titlesize": 15, "figure.titleweight": "bold", "svg.fonttype": "none",
})

PRETTY = {
    "nose2anogenital": "anogenital sniff", "nose2body": "body sniff",
    "nose2nose": "nose-to-nose", "sidebyside": "side-by-side",
    "sidereside": "side-to-side", "oriented_toward": "oriented toward",
    "following": "following", "chasing": "chasing", "approach": "approach",
    "withdrawal_from_partner": "withdrawal", "escape": "escape",
    "withdrawal_after_contact": "withdrawal after contact",
    "fighting": "fighting", "rearing": "rearing", "passive": "passive",
    "ACTIVEsocial": "any active engagement", "CONTACTmutual": "mutual contact",
}
OUT = A.ensure_out()


def load_bins() -> pd.DataFrame:
    """Load the cached 100 ms behaviour table without importing model code."""
    df = pd.read_csv(OUT / "beh_bins.csv.gz", compression="gzip", low_memory=False)
    df["animal_id"] = df["animal_id"].astype(str)
    return df


def mouse_glyph(ax, cx: float, cy: float, length: float, color: str,
                alpha: float = 1.0) -> tuple[float, float]:
    """Minimal top-down mouse pictogram, nose pointing +x.

    Drawn from three primitives - body ellipse, head circle, tail arc - so it
    stays legible at thumbnail size and vectorises cleanly into the SVG/PDF.
    Returns the nose and tail-base x positions so glyphs can be chained.
    """
    body_w, body_h = length * 0.62, length * 0.30
    ax.add_patch(Ellipse((cx, cy), body_w, body_h, facecolor=color,
                         edgecolor="none", alpha=alpha, zorder=6))
    head_r = length * 0.145
    nose_c = cx + body_w / 2 * 0.86
    ax.add_patch(Circle((nose_c, cy), head_r, facecolor=color, edgecolor="none",
                        alpha=alpha, zorder=6))
    # two small ears and a tail: enough to read as "mouse", nothing more
    for sgn in (-1, 1):
        ax.add_patch(Circle((nose_c - head_r * 0.45, cy + sgn * head_r * 0.95),
                            head_r * 0.42, facecolor=color, edgecolor="none",
                            alpha=alpha, zorder=6))
    tail_x = cx - body_w / 2
    tx = np.linspace(0, 1, 24)
    # tail arcs backwards and upwards, so a trailing animal never sits on it
    ax.plot(tail_x - tx * length * 0.26,
            cy + (1 - np.cos(tx * np.pi / 2)) * length * 0.20,
            color=color, lw=1.1, alpha=alpha, zorder=6, solid_capstyle="round")
    return nose_c + head_r, tail_x


def savefig(fig: plt.Figure, name: str) -> None:
    """Write png / svg / pdf side by side, as the lab convention requires."""
    for ext in ("png", "svg", "pdf"):
        target = OUT / f"{name}.{ext}"
        temporary = OUT / f".{name}.tmp.{ext}"
        fig.savefig(temporary, dpi=300 if ext == "png" else None,
                    bbox_inches="tight", format=ext)
        temporary.replace(target)
    plt.close(fig)
    print("saved", name)


# --------------------------------------------------------------------------- #
# Statistics                                                                  #
# --------------------------------------------------------------------------- #
def session_cells(df: pd.DataFrame) -> dict:
    """Per session: call counts, behaviour flags and the speed x distance stratum.

    Mirrors `attr4_10_analysis.session_arrays` exactly (same composites, same
    stratification, same partner-phase cut) so that the point estimates here
    reproduce `beh_enrichment.csv` to the last digit.
    """
    beh = [c[3:] for c in df.columns if c.startswith("m1_")]
    flag_names = [f"m1_{b}" for b in beh] + [f"m2_{b}" for b in beh]
    active = ("nose2anogenital", "nose2body", "following", "chasing", "approach")
    contact = ("nose2nose", "sidebyside", "sidereside")

    out = {}
    for animal, g in df[df["t"] >= PARTNER_T0].groupby("animal_id"):
        g = g.sort_values("bin").reset_index(drop=True)
        counts = g["n_calls"].to_numpy(float)
        if counts.sum() < 1:
            continue
        flags = g[flag_names].to_numpy() > 0

        def terc(v):
            """Tercile index of a kinematic trace, NaNs pulled to the median."""
            v = np.where(np.isfinite(v), v, np.nanmedian(v))
            return np.digitize(v, np.nanquantile(v, [1 / 3, 2 / 3]))

        strat = terc(g["speed_m1"].to_numpy()) * 3 + terc(g["dist"].to_numpy())

        comp, comp_names = [], []
        for tag in ("m1", "m2"):
            acc = np.zeros(len(counts), bool)
            for b in active:
                acc |= g[f"{tag}_{b}"].to_numpy() > 0
            comp.append(acc)
            comp_names.append(f"{tag}_ACTIVEsocial")
        acc = np.zeros(len(counts), bool)
        for b in contact:
            acc |= g[f"m1_{b}"].to_numpy() > 0
        comp.append(acc)
        comp_names.append("m1_CONTACTmutual")

        out[animal] = {
            "counts": counts,
            "flags": np.hstack([flags, np.stack(comp, axis=1)]),
            "strat": strat,
            "g": g,
            "names": flag_names + comp_names,
        }
    return out


def mh_terms(S: dict, j: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per-session Mantel-Haenszel numerator and denominator for flag `j`.

    The MH person-time rate ratio pools strata as
        RR = sum_k (x1_k * t0_k / T_k) / sum_k (x0_k * t1_k / T_k),
    which is a ratio of two sums over sessions. Because both sums are additive
    over sessions, a cluster bootstrap only needs one number per session per
    side - that is what this returns, making 4000 replicates essentially free.
    """
    animals, num, den = [], [], []
    for a, s in S.items():
        f = s["flags"][:, j]
        # same session-inclusion rule as the original enrichment analysis
        if f.sum() < 20 or (~f).sum() < 20:
            continue
        n_a = d_a = 0.0
        for k in range(N_STRATA):
            m = s["strat"] == k
            i1, i0 = m & f, m & ~f
            t1, t0 = i1.sum() * BIN_S, i0.sum() * BIN_S
            if t1 <= 0 or t0 <= 0:
                continue
            T = t1 + t0
            n_a += s["counts"][i1].sum() * t0 / T
            d_a += s["counts"][i0].sum() * t1 / T
        animals.append(a)
        num.append(n_a)
        den.append(d_a)
    return animals, np.asarray(num), np.asarray(den)


def fold_with_ci(S: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Movement-adjusted call-rate multiplier per behaviour, with a 95% cluster
    bootstrap CI over sessions (sessions are the independent unit, calls are not).
    """
    names = S[next(iter(S))]["names"]
    rows = []
    for j, nm in enumerate(names):
        animals, num, den = mh_terms(S, j)
        if len(animals) < 6 or den.sum() <= 0:
            continue
        fold = num.sum() / den.sum()
        idx = rng.integers(0, len(animals), size=(N_BOOT, len(animals)))
        bn, bd = num[idx].sum(1), den[idx].sum(1)
        boot = np.where(bd > 0, bn / np.maximum(bd, 1e-12), np.nan)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        occ = float(np.mean([S[a]["flags"][:, j].mean() for a in animals]) * 100)
        n_in = int(sum(S[a]["counts"][S[a]["flags"][:, j]].sum() for a in animals))
        rows.append({"flag": nm, "actor": nm[:2], "behavior": nm[3:],
                     "n_sessions": len(animals), "occupancy_pct": occ,
                     "calls_in_state": n_in, "fold_adj": float(fold),
                     "fold_lo": float(lo), "fold_hi": float(hi)})
    return pd.DataFrame(rows)


def hero_behaviour_order(fold: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return the pre-specified panel-A behaviours and the omitted behaviours."""
    f = fold[fold["actor"] == "m1"].set_index("behavior")
    keep = [b for b in f.index
            if f.loc[b, "occupancy_pct"] >= MIN_OCC_PCT
            and f.loc[b, "calls_in_state"] >= MIN_CALLS]
    rest = sorted((b for b in keep if b not in STARS and b != COMPOSITE),
                  key=lambda b: -f.loc[b, "fold_adj"])
    order = [b for b in STARS if b in keep]
    if COMPOSITE in keep:
        order.append(COMPOSITE)
    order += rest
    return order, [b for b in f.index if b not in keep]


def behaviour_call_count_statistics(
        S: dict, fold: pd.DataFrame
) -> tuple[pd.DataFrame, dict, pd.DataFrame, list[str], list[str]]:
    """Animal-level counts and a genotype x behaviour mixed ANOVA.

    One animal is one independent observation. Genotype is between animals and
    behaviour is repeated within animal. Counts are analysed after log1p
    transformation because their raw distribution is extremely right-skewed and
    contains zeros. Composite categories are excluded from inference because
    they contain the atomic behaviours and would duplicate the same calls.
    """
    selected, dropped = hero_behaviour_order(fold)
    order = [b for b in selected if b not in ("ACTIVEsocial", "CONTACTmutual")]
    rows = []
    for animal in sorted(S):
        s = S[animal]
        genotype = A.GENOTYPE_MAP[animal]
        for behaviour in order:
            j = s["names"].index(f"m1_{behaviour}")
            rows.append({
                "animal_id": animal,
                "genotype": genotype,
                "behavior": behaviour,
                "call_count": int(s["counts"][s["flags"][:, j]].sum()),
            })
    long = pd.DataFrame(rows)
    wide = long.pivot(index=["animal_id", "genotype"], columns="behavior",
                      values="call_count").reindex(columns=order)
    animals = wide.index.get_level_values("animal_id").to_numpy()
    groups = wide.index.get_level_values("genotype").to_numpy()
    Y_raw = wide.to_numpy(float)
    Y = np.log1p(Y_raw)
    n, k = Y.shape
    levels = ("WT", "HET")
    if set(groups) != set(levels):
        raise ValueError(f"Expected WT and HET animals, found {sorted(set(groups))}")

    grand = float(Y.mean())
    behavior_mean = Y.mean(axis=0)
    animal_mean = Y.mean(axis=1)
    group_mean = {g: float(Y[groups == g].mean()) for g in levels}
    group_behavior_mean = {g: Y[groups == g].mean(axis=0) for g in levels}
    group_n = {g: int(np.sum(groups == g)) for g in levels}

    ss_genotype = k * sum(group_n[g] * (group_mean[g] - grand) ** 2 for g in levels)
    ss_animal_group = k * sum(
        (animal_mean[i] - group_mean[groups[i]]) ** 2 for i in range(n))
    ss_behavior = n * np.square(behavior_mean - grand).sum()
    ss_interaction = sum(
        group_n[g] * np.square(group_behavior_mean[g] - group_mean[g]
                               - behavior_mean + grand).sum() for g in levels)
    ss_error = sum(
        np.square(Y[i] - animal_mean[i] - group_behavior_mean[groups[i]]
                  + group_mean[groups[i]]).sum() for i in range(n))

    df_genotype, df_animal_group = 1, n - len(levels)
    df_behavior = k - 1
    df_error = df_animal_group * df_behavior
    ms_error = ss_error / df_error
    f_genotype = (ss_genotype / df_genotype) / (ss_animal_group / df_animal_group)
    f_behavior = (ss_behavior / df_behavior) / ms_error
    f_interaction = (ss_interaction / df_behavior) / ms_error

    # Greenhouse-Geisser correction for the two within-animal effects.
    residual = np.vstack([
        Y[i] - animal_mean[i] - group_behavior_mean[groups[i]] + group_mean[groups[i]]
        for i in range(n)
    ])
    covariance = np.cov(residual, rowvar=False, ddof=1)
    centering = np.eye(k) - np.ones((k, k)) / k
    centered_cov = centering @ covariance @ centering
    epsilon = float(np.trace(centered_cov) ** 2 /
                    (df_behavior * np.trace(centered_cov @ centered_cov)))
    epsilon = float(np.clip(epsilon, 1 / df_behavior, 1.0))
    df_behavior_gg = epsilon * df_behavior
    df_error_gg = epsilon * df_error

    omnibus = {
        "test": "two-way mixed ANOVA on log1p animal-level call counts",
        "experimental_unit": "animal",
        "n_animals": n,
        "n_WT": group_n["WT"],
        "n_HET": group_n["HET"],
        "n_behaviors": k,
        "greenhouse_geisser_epsilon": epsilon,
        "effects": {
            "genotype": {
                "F": float(f_genotype), "df1": 1.0,
                "df2": float(df_animal_group),
                "p": float(stats.f.sf(f_genotype, 1, df_animal_group)),
                "partial_eta_squared": float(ss_genotype /
                                             (ss_genotype + ss_animal_group)),
            },
            "behavior_GG": {
                "F": float(f_behavior), "df1": df_behavior_gg,
                "df2": df_error_gg,
                "p": float(stats.f.sf(f_behavior, df_behavior_gg, df_error_gg)),
                "partial_eta_squared": float(ss_behavior / (ss_behavior + ss_error)),
            },
            "genotype_x_behavior_GG": {
                "F": float(f_interaction), "df1": df_behavior_gg,
                "df2": df_error_gg,
                "p": float(stats.f.sf(f_interaction, df_behavior_gg, df_error_gg)),
                "partial_eta_squared": float(ss_interaction /
                                             (ss_interaction + ss_error)),
            },
        },
    }

    posthoc = []
    for j, behavior in enumerate(order):
        wt, het = Y[groups == "WT", j], Y[groups == "HET", j]
        result = stats.ttest_ind(wt, het, equal_var=False, nan_policy="omit")
        posthoc.append({
            "family": "WT_vs_HET_within_behavior", "genotype": "WT_vs_HET",
            "behavior_1": behavior, "behavior_2": behavior,
            "t": float(result.statistic), "df": float(result.df),
            "p_raw": float(result.pvalue),
            "p_bonferroni": min(float(result.pvalue) * k, 1.0),
        })
    n_pairs = k * (k - 1) // 2
    for genotype in levels:
        Z = Y[groups == genotype]
        for i in range(k):
            for j in range(i + 1, k):
                result = stats.ttest_rel(Z[:, i], Z[:, j], nan_policy="omit")
                posthoc.append({
                    "family": "behavior_pairs_within_genotype",
                    "genotype": genotype, "behavior_1": order[i],
                    "behavior_2": order[j], "t": float(result.statistic),
                    "df": float(result.df), "p_raw": float(result.pvalue),
                    "p_bonferroni": min(float(result.pvalue) * n_pairs, 1.0),
                })
    posthoc = pd.DataFrame(posthoc)
    posthoc["significant_0.05"] = posthoc["p_bonferroni"] < 0.05
    return long, omnibus, posthoc, dropped, order


def action_reaction(S: dict, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """2 x 2 raw call rate: resident investigating x partner retreating.

    Deliberately *unadjusted*: this panel is about where the calls physically
    fall, and adding stratification here would make the four cells no longer
    comparable to a single overall baseline.
    """
    per = []
    for a, s in S.items():
        g, c = s["g"], s["counts"]
        inv = np.zeros(len(c), bool)
        for f in RES_INVESTIGATE:
            inv |= g[f].to_numpy() > 0
        ret = np.zeros(len(c), bool)
        for f in PART_RETREAT:
            ret |= g[f].to_numpy() > 0
        for lab, m in (("inv_ret", inv & ret), ("inv_only", inv & ~ret),
                       ("ret_only", ~inv & ret), ("neither", ~inv & ~ret)):
            per.append({"animal": a, "cell": lab, "calls": c[m].sum(),
                        "time_s": m.sum() * BIN_S})
    per = pd.DataFrame(per)
    cells = ["inv_ret", "inv_only", "ret_only", "neither"]
    piv_c = per.pivot_table(index="animal", columns="cell", values="calls",
                            aggfunc="sum").reindex(columns=cells).fillna(0)
    piv_t = per.pivot_table(index="animal", columns="cell", values="time_s",
                            aggfunc="sum").reindex(columns=cells).fillna(0)
    C, T = piv_c.to_numpy(), piv_t.to_numpy()

    rate = C.sum(0) / T.sum(0) * 60.0
    base = C.sum() / T.sum() * 60.0
    occ = T.sum(0) / T.sum() * 100.0

    idx = rng.integers(0, C.shape[0], size=(N_BOOT, C.shape[0]))
    bc, bt = C[idx].sum(1), T[idx].sum(1)
    brate = np.where(bt > 0, bc / np.maximum(bt, 1e-12) * 60.0, np.nan)
    bbase = bc.sum(1) / bt.sum(1) * 60.0
    bfold = brate / bbase[:, None]

    lo, hi = np.nanpercentile(bfold, [2.5, 97.5], axis=0)
    tab = pd.DataFrame({"cell": cells, "calls": C.sum(0), "time_s": T.sum(0),
                        "occupancy_pct": occ, "rate_per_min": rate,
                        "fold_vs_overall": rate / base,
                        "fold_lo": lo, "fold_hi": hi}).set_index("cell")

    def contrast(a_, b_):
        """Ratio of two cell rates, with its own bootstrap CI (same replicates)."""
        r = brate[:, cells.index(a_)] / brate[:, cells.index(b_)]
        return {"ratio": float(rate[cells.index(a_)] / rate[cells.index(b_)]),
                "lo": float(np.nanpercentile(r, 2.5)),
                "hi": float(np.nanpercentile(r, 97.5))}

    stats = {
        "baseline_rate_per_min": float(base),
        # does the partner's retreat add anything on top of the resident's action?
        "partner_effect_given_investigation": contrast("inv_ret", "inv_only"),
        # ... and does it do anything on its own?
        "partner_effect_alone": contrast("ret_only", "neither"),
        # the resident's action, with the partner held still
        "resident_effect_alone": contrast("inv_only", "neither"),
    }
    return tab, stats


# --------------------------------------------------------------------------- #
# Panels                                                                      #
# --------------------------------------------------------------------------- #
def panel_hero(ax, fold: pd.DataFrame, enr: pd.DataFrame) -> dict:
    """(a) Filtered, high-contrast multiplier bars, stars pinned to the top."""
    f = fold[fold["actor"] == "m1"].set_index("behavior")
    e = enr[enr["actor"] == "m1"].set_index("behavior")
    keep = [b for b in f.index
            if f.loc[b, "occupancy_pct"] >= MIN_OCC_PCT
            and f.loc[b, "calls_in_state"] >= MIN_CALLS]
    rest = sorted((b for b in keep if b not in STARS and b != COMPOSITE),
                  key=lambda b: -f.loc[b, "fold_adj"])
    order = [b for b in STARS if b in keep]
    if COMPOSITE in keep:
        order.append(COMPOSITE)
    order += rest
    dropped = [b for b in f.index if b not in keep]

    # top of the chart = first element of `order`
    y = np.arange(len(order))[::-1]
    for yi, b in zip(y, order):
        v = f.loc[b, "fold_adj"]
        col = (C_ACCENT if b in STARS else
               C_ACCENT_LT if b == COMPOSITE else
               C_MUTE if v >= 1 else C_MUTE_D)
        ax.barh(yi, v - 1.0, left=1.0, height=0.62, color=col,
                edgecolor=C_SURF, linewidth=1.2, zorder=3)
        ax.plot([f.loc[b, "fold_lo"], f.loc[b, "fold_hi"]], [yi, yi],
                color=C_INK if b in STARS else C_SEC, lw=1.4, alpha=0.85, zorder=4)
        # raw, movement-unadjusted multiplier: shown so the adjustment is auditable
        raw = 2.0 ** e.loc[b, "log2_RR_crude"]
        ax.plot([raw], [yi], marker="D", ms=4.2, mfc="none", mew=1.1,
                mec=C_SEC, zorder=5)
        q = e.loc[b, "q_adj"]
        tag = f"{v:.2f}×   q = {q:.3f}" if q < 0.05 else f"{v:.2f}×   q = {q:.2f}"
        ax.text(max(v, f.loc[b, "fold_hi"], raw) * 1.05, yi, tag, va="center",
                ha="left", fontsize=8.4, color=C_INK if b in STARS else C_SEC,
                fontweight="bold" if b in STARS else "normal")

    ax.axvline(1.0, color=C_INK, lw=1.3, zorder=2)
    ax.set_xscale("log")
    ticks = [0.5, 0.75, 1, 1.5, 2, 3]
    ax.set_xticks(ticks, [f"{t:g}×" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(0.42, 5.6)
    ax.set_yticks(y, [f"{PRETTY.get(b, b)}   ({f.loc[b, 'occupancy_pct']:.0f}% of time, "
                      f"{int(f.loc[b, 'calls_in_state'])} calls)" for b in order],
                  fontsize=9)
    for lab, b in zip(ax.get_yticklabels(), order):
        if b in STARS:
            lab.set_fontweight("bold")
            lab.set_color(C_INK)
    ax.set_ylim(-0.75, len(order) - 0.25)
    ax.set_xlabel("call rate while doing it  ÷  call rate while not doing it\n"
                  "(movement-adjusted; 1× = no change)")
    ax.grid(axis="y", visible=False)

    # pictograms for the two star behaviours, in the empty left margin of their
    # own rows: the reader sees what the behaviour *is* before reading the label
    for b in (bb for bb in STARS if bb in order):
        yi = y[order.index(b)]
        box = ax.inset_axes([0.015, (yi - ax.get_ylim()[0] - 0.34)
                             / np.ptp(ax.get_ylim()), 0.275, 0.075])
        box.set_axis_off()
        box.set_xlim(0, 100)
        box.set_ylim(-14, 14)
        box.patch.set_alpha(0)
        if b == "nose2anogenital":
            # nose of the resident right at the partner's tail base
            mouse_glyph(box, 76, -1, 46, C_ACCENT_LT)          # partner, ahead
            mouse_glyph(box, 33, -1, 46, C_ACCENT)             # resident, behind
        else:
            mouse_glyph(box, 79, -1, 42, C_ACCENT_LT)
            mouse_glyph(box, 25, -1, 42, C_ACCENT)
            box.annotate("", xy=(63, -1), xytext=(46, -1),
                         arrowprops=dict(arrowstyle="-|>", color=C_INK, lw=1.2,
                                         ls=(0, (2.2, 1.4))))

    ax.legend(handles=[
        Patch(color=C_ACCENT, label="investigation of the partner"),
        Patch(color=C_ACCENT_LT, label="composite containing both"),
        Patch(color=C_MUTE, label="every other behaviour"),
        plt.Line2D([], [], color=C_SEC, lw=1.4, label="95% CI (bootstrap over sessions)"),
        plt.Line2D([], [], marker="D", ls="none", mfc="none", mec=C_SEC,
                   label="before movement adjustment"),
    ], fontsize=7.6, frameon=False, loc="lower right", handlelength=1.4)

    ax.set_title("a   Calls are 2× more likely while the resident investigates\n"
                 "     the partner - and only while it does", loc="left",
                 color=C_INK)
    return {"order": order, "dropped": dropped}


def panel_call_counts(ax, counts: pd.DataFrame, omnibus: dict,
                      pairwise: pd.DataFrame, order: list[str]) -> None:
    """(a) WT versus HET animal-level call-count box-and-strip plot."""
    y = np.arange(len(order))[::-1]
    genotype_style = {"WT": (C_ACCENT, 0.18), "HET": ("#dc6b35", -0.18)}
    rng = np.random.default_rng(SEED + 17)
    max_count = 0.0
    for genotype, (color, offset) in genotype_style.items():
        values = [counts.loc[(counts["behavior"] == b)
                             & (counts["genotype"] == genotype),
                             "call_count"].to_numpy(float) for b in order]
        max_count = max(max_count, max(float(np.max(v)) for v in values))
        bp = ax.boxplot(values, positions=y + offset, orientation="horizontal",
                        widths=0.28, patch_artist=True, showfliers=False, whis=1.5,
                        medianprops={"color": C_INK, "linewidth": 1.15},
                        whiskerprops={"color": color, "linewidth": 0.9},
                        capprops={"color": color, "linewidth": 0.9})
        for box in bp["boxes"]:
            box.set_facecolor(matplotlib.colors.to_rgba(color, 0.27))
            box.set_edgecolor(color)
            box.set_linewidth(1.05)
        for yi, vals in zip(y + offset, values):
            jitter = rng.uniform(-0.075, 0.075, len(vals))
            ax.scatter(vals, yi + jitter, s=27, color=color, edgecolor=C_SURF,
                       linewidth=0.5, alpha=0.88, zorder=4)

    ax.set_xscale("symlog", linthresh=1.0, linscale=0.55, base=10)
    ticks = [0, 1, 3, 10, 30, 100, 300]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(-0.15, max(430.0, max_count * 1.55))
    ax.set_yticks(y, [PRETTY.get(b, b) for b in order], fontsize=9)
    for lab, b in zip(ax.get_yticklabels(), order):
        if b in STARS:
            lab.set_fontweight("bold")
            lab.set_color(C_INK)
    # Extra headroom keeps the ANOVA summary separate from the first data row.
    ax.set_ylim(-0.85, len(order) + 0.65)
    ax.set_xlabel("calls per 10-minute animal session (each dot is one animal; log-like axis)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="none", color=C_ACCENT, label="WT, n = 12"),
        plt.Line2D([], [], marker="o", ls="none", color="#dc6b35", label="HET, n = 12"),
    ], frameon=False, loc="upper left", ncol=2, fontsize=8.0,
       bbox_to_anchor=(0.00, 1.005), borderaxespad=0)

    genotype_tests = pairwise[
        pairwise["family"] == "WT_vs_HET_within_behavior"
    ].set_index("behavior_1")
    star_x = max_count * 1.18
    for yi, behavior in zip(y, order):
        p_adj = float(genotype_tests.loc[behavior, "p_bonferroni"])
        stars = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else (
            "*" if p_adj < 0.05 else "ns")
        ax.text(star_x, yi, stars, ha="center", va="center", fontsize=8.0,
                color=C_INK if stars != "ns" else C_MUT,
                fontweight="bold" if stars != "ns" else "normal")
    ax.text(star_x, len(order) - 0.42, "WT vs HET", ha="center", va="bottom",
            fontsize=6.8, color=C_SEC)

    effects = omnibus["effects"]
    g = effects["genotype"]
    b = effects["behavior_GG"]
    interaction = effects["genotype_x_behavior_GG"]
    ax.text(0.99, 0.985,
            "two-way mixed ANOVA on log(1 + count), N = 24 animals\n"
            f"genotype: F({g['df1']:.0f}, {g['df2']:.0f}) = {g['F']:.2f}, "
            f"p = {g['p']:.4f}; behavior: F({b['df1']:.2f}, {b['df2']:.1f}) "
            f"= {b['F']:.2f}, p = {b['p']:.2e}\n"
            f"genotype x behavior: F({interaction['df1']:.2f}, "
            f"{interaction['df2']:.1f}) = {interaction['F']:.2f}, "
            f"p = {interaction['p']:.4f} (GG corrected)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.4,
            color=C_INK,
            bbox={"boxstyle": "round,pad=0.35", "fc": C_SURF,
                  "ec": C_BASE, "alpha": 0.94})

    within = pairwise[(pairwise["family"] == "behavior_pairs_within_genotype")
                      & pairwise["significant_0.05"]]
    summaries = []
    for genotype in ("WT", "HET"):
        q = within[within["genotype"] == genotype]
        terms = []
        for r in q.itertuples():
            high = r.behavior_1 if r.t > 0 else r.behavior_2
            low = r.behavior_2 if r.t > 0 else r.behavior_1
            terms.append(f"{PRETTY.get(high, high)} > {PRETTY.get(low, low)}")
        summaries.append(f"{genotype}: " + (", ".join(terms) if terms else "none"))
    ax.text(0.01, 0.012,
            "Within-genotype behavior contrasts (Bonferroni):\n"
            + "; ".join(summaries), transform=ax.transAxes, ha="left",
            va="bottom", fontsize=6.4, color=C_SEC, style="italic")
    ax.set_title("a   WT animals call more across behaviours; profile interaction p = 0.050",
                 loc="left", color=C_INK)


def panel_matrix(ax, tab: pd.DataFrame, st: dict) -> None:
    """(b) Action x reaction: the resident's move and the partner's answer."""
    ax.set_axis_off()
    # rows: partner retreating yes / no ; cols: resident investigating yes / no
    grid = [["inv_ret", "ret_only"], ["inv_only", "neither"]]
    vmax = tab["rate_per_min"].max()
    for r in range(2):
        for c in range(2):
            cell = grid[r][c]
            v = tab.loc[cell, "rate_per_min"]
            shade = 0.10 + 0.80 * (v / vmax)
            face = matplotlib.colors.to_rgba(C_ACCENT, shade)
            ax.add_patch(Rectangle((c, 1 - r), 0.92, 0.92, facecolor=face,
                                   edgecolor=C_SURF, lw=2))
            dark = shade > 0.5
            ink = "white" if dark else C_INK
            ax.text(c + 0.46, 1 - r + 0.60, f"{v:.1f}", ha="center", va="center",
                    fontsize=18, fontweight="bold", color=ink)
            ax.text(c + 0.46, 1 - r + 0.42, "calls / min", ha="center",
                    va="center", fontsize=7.4, color=ink, alpha=0.85)
            ax.text(c + 0.46, 1 - r + 0.24,
                    f"{tab.loc[cell, 'fold_vs_overall']:.1f}× overall",
                    ha="center", va="center", fontsize=7.0, color=ink, alpha=0.85)
            ax.text(c + 0.46, 1 - r + 0.10,
                    f"{tab.loc[cell, 'occupancy_pct']:.0f}% of the window",
                    ha="center", va="center", fontsize=6.6, color=ink, alpha=0.75)

    ax.set_xlim(-0.78, 2.02)
    ax.set_ylim(-1.42, 2.44)
    ax.text(0.46, 2.14, "resident\ninvestigates", ha="center", va="center",
            fontsize=8.8, fontweight="bold", color=C_ACCENT)
    ax.text(1.46, 2.14, "resident\ndoes not", ha="center", va="center",
            fontsize=8.8, color=C_SEC)
    ax.text(-0.08, 1.46, "partner\nretreats", ha="right", va="center",
            fontsize=8.8, color=C_SEC)
    ax.text(-0.08, 0.46, "partner\nstays", ha="right", va="center",
            fontsize=8.8, color=C_SEC)

    a = st["resident_effect_alone"]
    b = st["partner_effect_given_investigation"]
    c_ = st["partner_effect_alone"]
    ax.annotate("", xy=(0.30, -0.10), xytext=(1.62, -0.10),
                arrowprops=dict(arrowstyle="-|>", color=C_ACCENT, lw=2.2))
    ax.text(-0.78, -0.46, f"resident starts investigating   ×{a['ratio']:.1f} "
                          f"[{a['lo']:.1f}, {a['hi']:.1f}]", ha="left", va="top",
            fontsize=8.4, color=C_ACCENT, fontweight="bold")
    ax.text(-0.78, -0.78,
            f"partner starts retreating   ×{b['ratio']:.2f} "
            f"[{b['lo']:.2f}, {b['hi']:.2f}] while investigated,\n"
            f"                                        ×{c_['ratio']:.2f} "
            f"[{c_['lo']:.2f}, {c_['hi']:.2f}] when it is not",
            ha="left", va="top", fontsize=8.0, color=C_SEC)
    ax.text(-0.78, -1.32, "raw rates; ratios are cluster bootstraps over the "
                          "24 animals (one session each)", ha="left", va="bottom", fontsize=6.8,
            color=C_MUT, style="italic")
    ax.set_title("b   The resident's action sets the rate; the partner's\n"
                 "     retreat adds nothing on top of it", loc="left",
                 color=C_INK)


def panel_actor(ax, fold: pd.DataFrame, val: dict) -> None:
    """(c) Same behaviour, resident as actor vs partner as actor."""
    F = fold.set_index(["behavior", "actor"])
    f = F["fold_adj"]
    beh = [b for b in DIRECTIONAL if (b, "m1") in f.index and (b, "m2") in f.index]
    beh = sorted(beh, key=lambda b: f[(b, "m1")] / f[(b, "m2")])
    y = np.arange(len(beh))
    for i, b in enumerate(beh):
        v1, v2 = f[(b, "m1")], f[(b, "m2")]
        ax.plot([v2, v1], [i, i], color=C_BASE, lw=2.6, zorder=2,
                solid_capstyle="round")
        # a hollow dot flags an estimate resting on fewer than 20 calls
        for v, col, actor in ((v2, C_PART, "m2"), (v1, C_RES, "m1")):
            thin = F.loc[(b, actor), "calls_in_state"] < 20
            ax.scatter([v], [i], s=56, zorder=3, linewidths=1.4,
                       facecolors=C_SURF if thin else col,
                       edgecolors=col if thin else C_SURF)
        if i == len(beh) - 1:                      # direct labels, no legend box
            ax.annotate("resident acts", (v1, i), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=7.8,
                        color=C_RES, fontweight="bold")
            ax.annotate("partner acts", (v2, i), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=7.8,
                        color=C_PART, fontweight="bold")
    ax.axvline(1.0, color=C_INK, lw=1.2, zorder=1)
    ax.set_xscale("log")
    ticks = [0.25, 0.5, 1, 2, 4]
    ax.set_xticks(ticks, [f"{t:g}×" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(0.2, 5.0)
    ax.set_yticks(y, [PRETTY.get(b, b) for b in beh], fontsize=8.4)
    ax.set_ylim(-0.6, len(beh) - 0.15)
    ax.set_xlabel("call-rate multiplier")
    ax.grid(axis="y", visible=False)
    ax.text(0.02, 0.02, "hollow = < 20 calls", transform=ax.transAxes,
            fontsize=6.8, color=C_MUT, style="italic")
    asym = val.get("actor_asymmetry", {})
    ax.set_title(f"c   Enrichment follows the actor, not the target\n"
                 f"     gap {2 ** asym.get('stat_log2', np.nan):.1f}×, "
                 f"p = {asym.get('p', np.nan):.2f} (suggestive)",
                 loc="left", color=C_INK)


def panel_exemplar(ax_frames, ax_spec, df: pd.DataFrame) -> None:
    """(d) One real episode: arena frames over the matching spectrogram."""
    import cv2

    a = EXEMPLAR["animal"]
    t0, t1 = EXEMPLAR["t0"], EXEMPLAR["t1"]

    # ---- video frames, cropped around the pair --------------------------- #
    track = A.load_tracking(a)
    cap = cv2.VideoCapture(str(A.session_dir(a) / f"{a}_1_baseline.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    for ax, tv, lab in zip(ax_frames, EXEMPLAR["frames"], EXEMPLAR["frame_labels"]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(tv * fps)))
        ok, frame = cap.read()
        ax.set_axis_off()
        if not ok:
            continue
        i = (track["t"] - tv).abs().idxmin()
        q = track.loc[i]
        cx, cy = (q["m1_x"] + q["m2_x"]) / 2, (q["m1_y"] + q["m2_y"]) / 2
        # crop just wide enough for both animals, then padded out to a fixed
        # 4:3 frame so the three stills tile the strip without gaps
        wx = 0.60 * abs(q["m1_x"] - q["m2_x"]) + 145
        wy = 0.60 * abs(q["m1_y"] - q["m2_y"]) + 145
        wx, wy = max(wx, FRAME_AR * wy), max(wy, wx / FRAME_AR)
        wx, wy = int(max(wx, FRAME_AR * wy)), int(wy)
        H, W = frame.shape[:2]
        x0 = int(np.clip(cx - wx, 0, max(W - 2 * wx, 0)))
        y0 = int(np.clip(cy - wy, 0, max(H - 2 * wy, 0)))
        crop = frame[y0:y0 + 2 * wy, x0:x0 + 2 * wx]
        # gentle lift: the animals are black on a bright substrate, so the
        # posture is invisible at the printed size without it
        crop = cv2.convertScaleAbs(crop, alpha=1.45, beta=18)
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.scatter([q["m1_x"] - x0], [q["m1_y"] - y0], s=110, facecolors="none",
                   edgecolors=C_RES, linewidths=2.2)
        ax.scatter([q["m2_x"] - x0], [q["m2_y"] - y0], s=110, facecolors="none",
                   edgecolors=C_PARTNER_MARK, linewidths=2.2)
        # arrow from resident to partner: who is doing it to whom
        ax.add_patch(FancyArrowPatch(
            (q["m1_x"] - x0, q["m1_y"] - y0), (q["m2_x"] - x0, q["m2_y"] - y0),
            arrowstyle="-|>", mutation_scale=13, color="white", lw=1.8,
            shrinkA=11, shrinkB=11, alpha=0.95))
        # caption inside the still: keeps the strip flush against panel c above
        ax.text(0.03, 0.965, f"{lab}   ·   t = {tv:.2f} s",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.0,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.30", fc="#00000099", ec="none"))
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(C_BASE)
    cap.release()

    # ---- spectrogram ----------------------------------------------------- #
    import soundfile as sf
    wav = str(A.wav_path(a))
    info = sf.info(wav)
    x, sr = sf.read(wav, start=int(t0 * info.samplerate),
                    frames=int((t1 - t0) * info.samplerate), dtype="float32")
    freqs, times, Sxx = signal.spectrogram(x, sr, nperseg=1024, noverlap=768,
                                           window="hann")
    band = (freqs >= 35e3) & (freqs <= 120e3)
    Sdb = 10 * np.log10(Sxx[band] + 1e-14)
    ax_spec.pcolormesh(times + t0, freqs[band] / 1e3, Sdb,
                       vmin=np.percentile(Sdb, 88),
                       vmax=np.percentile(Sdb, 99.85),
                       cmap="magma", shading="auto", rasterized=True)
    ax_spec.set_ylim(35, 120)
    ax_spec.set_ylabel("frequency (kHz)")
    ax_spec.set_xlabel(f"session time (s)   ·   animal {a}, "
                       "384 kHz single microphone")
    ax_spec.grid(False)
    for s in ax_spec.spines.values():
        s.set_visible(True)
        s.set_color(C_BASE)

    # detected calls as ticks under the spectrogram
    calls = A.load_calls()
    c = calls[(calls["animal_id"] == a) & (calls["end_s"] > t0)
              & (calls["start_s"] < t1)]
    for _, r in c.iterrows():
        ax_spec.add_patch(Rectangle((r["start_s"], 36.0),
                                    max(r["end_s"] - r["start_s"], 0.004), 3.0,
                                    facecolor=C_HOT, edgecolor="none", zorder=4))
    ax_spec.text(t0 + 0.04, 44.0, f"{len(c)} detected calls in {t1 - t0:.1f} s "
                                  f"= {len(c) / (t1 - t0) * 60:.0f} / min",
                 fontsize=8.0, color="white", va="bottom", zorder=5,
                 fontweight="bold")

    # behaviour ribbon on top of the spectrogram
    g = df[(df["animal_id"] == a) & (df["t"] >= t0 - BIN_S) & (df["t"] <= t1)]
    ribbon = [("m1_nose2anogenital", 117.0, C_ACCENT, "anogenital sniff"),
              ("m1_following", 112.0, "#79b3f0", "following")]
    for col, yy, cc, lab in ribbon:
        for tt in g[g[col] > 0]["t"].to_numpy():
            ax_spec.add_patch(Rectangle((tt, yy - 1.5), BIN_S, 3.0,
                                        facecolor=cc, edgecolor="none", zorder=4))
        ax_spec.text(t1 - 0.04, yy, lab, ha="right", va="center", fontsize=7.4,
                     color="white", zorder=5, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.22", fc="#000000b0",
                               ec="none"))

    # tie each frame to its instant on the spectrogram
    for tv in EXEMPLAR["frames"]:
        ax_spec.axvline(tv, color="white", lw=1.0, ls=":", alpha=0.85, zorder=5)
    ax_spec.set_xlim(t0, t1)
    ax_spec.set_title("d   The same claim in one 4.5 s episode: the resident sniffs, then follows, and calls throughout",
                      loc="left", color=C_INK)


def panel_stats(ax, pred: dict, val: dict, hero: dict) -> None:
    """Compact reassurance strip: held-out prediction and cohort replication."""
    ax.set_axis_off()
    tiles = [
        (f"{val['n_calls_partner']:,}", "attributed calls", "24 animals"),
        (f"{pred['both']['loso_auc_mean']:.2f}",
         "held-out AUC", "leave-one-session-out"),
        (f"ρ = {val['splithalf_rho_mean']:.2f}", "split-half replication",
         f"95% CI {val['splithalf_rho_lo']:.2f}–{val['splithalf_rho_hi']:.2f}"),
    ]
    w = 1.0 / len(tiles)
    for i, (big, lab, sub) in enumerate(tiles):
        x = i * w
        ax.add_patch(Rectangle((x + 0.006, 0.06), w - 0.012, 0.88,
                               transform=ax.transAxes, facecolor="#f4f3ee",
                               edgecolor=C_BASE, lw=0.8))
        ax.text(x + w / 2, 0.68, big, transform=ax.transAxes, ha="center",
                va="center", fontsize=15, fontweight="bold", color=C_ACCENT)
        ax.text(x + w / 2, 0.40, lab, transform=ax.transAxes, ha="center",
                va="center", fontsize=8.4, color=C_INK)
        ax.text(x + w / 2, 0.21, sub, transform=ax.transAxes, ha="center",
                va="center", fontsize=7.2, color=C_MUT)


# --------------------------------------------------------------------------- #
# Figure                                                                      #
# --------------------------------------------------------------------------- #
def build(df, fold, enr, tab, st, pred, val, counts, omnibus, pairwise, dropped,
          order, panel_a: str = "multiplier"):
    """Assemble the figure.

    `panel_a` selects the left-hand hero panel and nothing else, so the two
    variants differ in exactly one object and the benchmark compares like with
    like: "multiplier" is the movement-adjusted rate ratio with bootstrap CIs,
    "counts" is the per-session box-and-strip of raw call counts.
    """
    fig = plt.figure(figsize=(17.2, 9.6))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.00, 1.20], wspace=0.15,
                             left=0.052, right=0.986, top=0.876, bottom=0.062)
    left = outer[0, 0].subgridspec(2, 1, height_ratios=[7.6, 1.0], hspace=0.20)
    right = outer[0, 1].subgridspec(3, 1, height_ratios=[2.55, 1.95, 2.10],
                                    hspace=0.32)

    ax_a = fig.add_subplot(left[0, 0])
    if panel_a == "counts":
        panel_call_counts(ax_a, counts, omnibus, pairwise, order)
        hero = {"order": order, "dropped": dropped}
        note = ("shows animal-level WT and HET counts for atomic behaviours "
                f"occupying ≥ {MIN_OCC_PCT:.0f}% of the window with ≥ "
                f"{MIN_CALLS} calls. Composite and rarer states are omitted.")
    else:
        hero = panel_hero(ax_a, fold, enr)
        note = ("shows only behaviours occupying ≥ "
                f"{MIN_OCC_PCT:.0f}% of the window with ≥ {MIN_CALLS} calls "
                f"({len(dropped)} rarer states omitted, incl. "
                "withdrawal-after-contact, the one significant suppression).")
    panel_stats(fig.add_subplot(left[1, 0]), pred, val, hero)

    gs_top = right[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.88], wspace=0.38)
    panel_matrix(fig.add_subplot(gs_top[0, 0]), tab, st)
    panel_actor(fig.add_subplot(gs_top[0, 1]), fold, val)

    gs_fr = right[1, 0].subgridspec(1, 3, wspace=0.045)
    ax_frames = [fig.add_subplot(gs_fr[0, i]) for i in range(3)]
    ax_spec = fig.add_subplot(right[2, 0])
    panel_exemplar(ax_frames, ax_spec, df)

    fig.suptitle("Vocalisations track active pursuit: calling peaks while the "
                 "resident is sniffing and following its partner", y=0.965)
    fig.text(0.05, 0.925,
             f"{val['n_calls_partner']:,} resident-attributed calls, "
             "N = 24 animals (12 WT, 12 HET), one session per animal, 100 ms bins.  Panel a "
             + note, fontsize=8.4, color=C_SEC, ha="left", va="center")
    return fig


def save_benchmark(old_name: str, new_name: str) -> None:
    """Save a side-by-side visual benchmark of the original and revised figures."""
    old_path, new_path = OUT / f"{old_name}.png", OUT / f"{new_name}.png"
    if not old_path.exists() or not new_path.exists():
        print("benchmark skipped: source PNG missing")
        return
    old, new = plt.imread(old_path), plt.imread(new_path)
    fig, axes = plt.subplots(2, 1, figsize=(17.2, 19.2))
    for ax, image, title in zip(
            axes, (old, new),
            ("Previous version: pooled animal-level call counts",
             "New version: WT versus HET animal-level call counts")):
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=14, pad=10)
    fig.tight_layout()
    savefig(fig, "v4_fig1_calls_boxstrip_vs_genotype")


def main():
    rng = np.random.default_rng(SEED)
    print("loading bins ...", flush=True)
    df = load_bins()
    S = session_cells(df)
    print(f"{len(S)} animals in the interaction window")

    print("bootstrapping call-rate multipliers ...", flush=True)
    fold = fold_with_ci(S, rng)
    fold.to_csv(OUT / "hero_fold_ci.csv", index=False)
    print(fold[fold["actor"] == "m1"]
          .sort_values("fold_adj", ascending=False)
          [["behavior", "occupancy_pct", "calls_in_state", "fold_adj",
            "fold_lo", "fold_hi"]].round(3).to_string(index=False))

    print("\nanimal-level genotype x behaviour call counts ...", flush=True)
    counts, omnibus, pairwise, dropped, order = behaviour_call_count_statistics(S, fold)
    counts.to_csv(OUT / "hero_behavior_call_counts_by_genotype.csv", index=False)
    pairwise.to_csv(OUT / "hero_behavior_call_counts_by_genotype_posthoc.csv",
                    index=False)
    with open(OUT / "hero_behavior_call_counts_two_way_anova.json", "w",
              encoding="utf-8") as f:
        json.dump(omnibus, f, indent=2)
    print(json.dumps(omnibus, indent=2))
    print(pairwise.sort_values("p_bonferroni").head(10)
          .round(5).to_string(index=False))

    print("\naction x reaction ...", flush=True)
    tab, st = action_reaction(S, rng)
    tab.to_csv(OUT / "hero_action_reaction.csv")
    print(tab.round(3).to_string())
    print(json.dumps(st, indent=2))

    enr = pd.read_csv(OUT / "beh_enrichment.csv")
    pred = json.load(open(OUT / "beh_prediction.json"))
    val = json.load(open(OUT / "beh_validation.json"))

    # Both variants are rendered from the same run so the benchmark below never
    # compares a fresh figure against a stale PNG left over from an earlier one.
    args = (df, fold, enr, tab, st, pred, val, counts, omnibus, pairwise,
            dropped, order)
    savefig(build(*args, panel_a="multiplier"), "v4_fig1_hero")
    savefig(build(*args, panel_a="counts"), "v4_fig1_calls_by_genotype")
    save_benchmark("v4_fig1_calls_boxstrip", "v4_fig1_calls_by_genotype")


if __name__ == "__main__":
    main()
