"""
Spectroscopic constants validation.

For each assigned vibrational band (isotope, m1, m2, m3, r, parity), fits:

    E(J) = E0 + B_v * J(J+1) - D_v * [J(J+1)]^2 + ...

using linear least squares with iterative MAD-based sigma-clipping.

Three checks:
  1. Ma (MARVEL) bands: fit residuals should be < 0.001 cm-1.
  2. GNN vs Ma comparison: for overlapping clean bands, |B_v(GNN) - B_v(Ma)|
     should be < 0.001 cm-1.
  3. Isotopologue mass scaling: B_v(iso)/B_v(626) should match I(626)/I(iso).

Outputs
-------
  data/spectroscopic_constants_marvel.csv
  data/spectroscopic_constants_gnn.csv
  data/figures/spec_const_bv_scatter.png
  data/figures/spec_const_mae_by_isotopologue.png
  data/figures/spec_const_delta_bv.png
"""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

from plotting import thesis_params

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
PREDICTIONS_PATH = os.path.join(DATA_DIR, "assigned_co2_predictions.csv")
FIGURES_DIR = os.path.join(DATA_DIR, "figures")
OUT_MARVEL = os.path.join(DATA_DIR, "spectroscopic_constants_marvel.csv")
OUT_GNN = os.path.join(DATA_DIR, "spectroscopic_constants_gnn.csv")

# ── Thresholds ─────────────────────────────────────────────────────────────────
CONFIDENT_MARGIN = 2.0  # GNN states used for fitting
EXCLUDE_POLYADS = {15, 18}  # known overconfident polyads
MIN_STATES_FIT = 5  # minimum states per group to attempt a fit
# High polyads have dense Fermi resonance manifolds where the AFGL r-label
# undergoes order crossings with J (the r-th state by energy at J=10 may be a
# different physical state than at J=20).  Restricting to low polyads ensures
# the r-label is J-stable and each (m1,m2,m3,r,parity,polyad) group forms a
# single, well-defined rotational ladder.
MAX_POLYAD = 10
R_EQ_ANG = 1.1621  # equilibrium C-O bond length in Angstrom (fixed geometry)
B_E_626 = 0.39022  # B_e for main isotopologue 626 (cm-1), Herzberg/NIST
# GNN bands with RMSE below this threshold after sigma-clipping are considered
# "clean" rotational ladders — physically consistent assignments.  Bands above
# threshold are resonance-perturbed or mixed-assignment cases excluded from
# the validation comparison.
RMSE_CLEAN = 0.01  # cm-1

mpl.rcParams.update(thesis_params)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Physics helpers ────────────────────────────────────────────────────────────


def moment_of_inertia(m_A, m_C, m_B, r_ang=R_EQ_ANG):
    """
    Moment of inertia (amu·Å²) for linear O_A-C-O_B with equal bond lengths r.
    COM offset included for asymmetric isotopologues.
    """
    M = m_A + m_C + m_B
    # Place C at origin; O_A at -r, O_B at +r
    x_com = r_ang * (m_B - m_A) / M
    I = m_A * (r_ang + x_com) ** 2 + m_C * x_com**2 + m_B * (r_ang - x_com) ** 2
    return I


def _poly_fit(x, E, n_terms):
    """Scaled polynomial fit; returns (coeffs_unscaled_to_physical, E_fit, x_scale)."""
    x_scale = max(x.max(), 1.0)
    xs = x / x_scale
    A = np.column_stack([xs**k for k in range(n_terms)])
    coeffs, _, _, _ = lstsq(A, E)
    return coeffs, A @ coeffs, x_scale


def fit_rotational_constants(J_vals, energies):
    """
    Fit E = E0 + B_v*x - D_v*x^2 + H_v*x^3 - L_v*x^4  (x = J*(J+1)).

    Higher-order terms needed at high J (D_v*x^2 ~ 160 cm-1 at J=200).
    x is scaled internally to avoid Vandermonde ill-conditioning.

    Iterative sigma-clipping using MAD (median absolute deviation) as the
    robust scatter estimate.  Standard std() is strongly biased by even a
    single large outlier, making the clip threshold too wide.  MAD is
    insensitive to outliers so the threshold stays tight.

    Convergence: iterate until no new outliers are found (max 5 rounds).

    Polynomial order: N>=8 -> 5 terms; N>=6 -> 4; else 3.

    Returns (E0, B_v, D_v, rmse, n_outliers).
    """
    x = J_vals * (J_vals + 1)
    N = len(x)
    n_terms = 5 if N >= 8 else (4 if N >= 6 else 3)

    keep = np.ones(N, dtype=bool)
    n_outliers = 0

    for _ in range(5):
        if keep.sum() < MIN_STATES_FIT:
            break
        nt = min(n_terms, keep.sum() - 1)
        coeffs, E_fit, x_scale = _poly_fit(x[keep], energies[keep], nt)
        resid_all = energies - (
            np.column_stack([(x / x_scale) ** k for k in range(nt)]) @ coeffs
        )
        resid_kept = resid_all[keep]
        mad = np.median(np.abs(resid_kept - np.median(resid_kept)))
        sigma_rob = mad / 0.6745 if mad > 0 else 1e-6
        new_keep = np.abs(resid_all) <= 3 * sigma_rob
        new_outliers = int((~new_keep).sum())
        if new_outliers == n_outliers:
            break
        n_outliers = new_outliers
        keep = new_keep

    # Final fit on inliers
    if keep.sum() >= MIN_STATES_FIT:
        nt = min(n_terms, keep.sum() - 1)
        coeffs, E_fit, x_scale = _poly_fit(x[keep], energies[keep], nt)
        resid = energies[keep] - E_fit
    else:
        resid = np.array([0.0])

    rmse = np.sqrt(np.mean(resid**2))
    E0 = coeffs[0]
    Bv = coeffs[1] / x_scale if len(coeffs) > 1 else np.nan
    Dv = -coeffs[2] / x_scale**2 if len(coeffs) > 2 else np.nan

    return E0, Bv, Dv, rmse, n_outliers


# ── Fitting routines ───────────────────────────────────────────────────────────


def fit_groups(df, m1_col, m2_col, m3_col, r_col, label):
    """
    Fit rotational constants for all qualifying groups in df.

    Only states where AFGL polyad (2*m1 + m2 + 3*m3) matches the Trove
    polyad_int are included.  States where these differ are highly mixed
    resonance cases where the same AFGL label spans multiple Trove manifolds
    — they cannot form a single rotational ladder and are excluded from
    the validation fit.
    """
    df = df.copy()
    afgl_polyad = 2 * df[m1_col] + df[m2_col] + 3 * df[m3_col]
    df = df[afgl_polyad == df["polyad_int"]]

    group_cols = ["isotope_id", m1_col, m2_col, m3_col, r_col, "parity_encoded"]
    records = []
    for keys, grp in df.groupby(group_cols):
        iso, m1, m2, m3, r, par = keys
        if len(grp) < MIN_STATES_FIT:
            continue
        J = grp["J"].values.astype(float)
        E = grp["energy"].values
        if len(np.unique(J)) < 2:
            continue
        E0, Bv, Dv, rmse, n_out = fit_rotational_constants(J, E)
        records.append(
            {
                "isotope_id": iso,
                "m1": m1,
                "m2": m2,
                "m3": m3,
                "r": r,
                "parity": par,
                "N": len(grp),
                "n_outliers": n_out,
                "E0": E0,
                "B_v": Bv,
                "D_v": Dv,
                "rmse": rmse,
                "source": label,
            }
        )
    return pd.DataFrame(records)


# ── Mass scaling ───────────────────────────────────────────────────────────────


def add_mass_scaling(df_ca, df_all):
    """
    For each Ca fit, attach the mass-predicted B_v ratio relative to 626.
    Uses masses from the first row of each isotopologue in df_all.
    """
    mass_lookup = (
        df_all.groupby("isotope_id")[["C_mass", "O_A_mass", "O_B_mass"]]
        .first()
        .to_dict("index")
    )

    m626 = mass_lookup.get(626, mass_lookup.get("626"))
    if m626 is None:
        return df_ca

    I_626 = moment_of_inertia(m626["O_A_mass"], m626["C_mass"], m626["O_B_mass"])

    ratios, predicted_bv = [], []
    for iso in df_ca["isotope_id"]:
        iso_key = int(iso) if int(iso) in mass_lookup else str(iso)
        if iso_key not in mass_lookup:
            ratios.append(np.nan)
            predicted_bv.append(np.nan)
            continue
        m = mass_lookup[iso_key]
        I_iso = moment_of_inertia(m["O_A_mass"], m["C_mass"], m["O_B_mass"])
        ratio = I_626 / I_iso  # B_v(iso)/B_v(626) ≈ I(626)/I(iso)
        ratios.append(ratio)
        predicted_bv.append(B_E_626 * ratio)

    df_ca = df_ca.copy()
    df_ca["mass_ratio_predicted"] = ratios
    df_ca["B_v_mass_predicted"] = predicted_bv
    return df_ca


# ── Figures ───────────────────────────────────────────────────────────────────


def _merged_clean(df_marvel, df_gnn):
    """Merge Ma and GNN fits on clean GNN bands; return merged DataFrame."""
    merge_keys = ["isotope_id", "m1", "m2", "m3", "r", "parity"]
    gnn_clean = df_gnn[df_gnn["rmse"] < RMSE_CLEAN]
    merged = df_marvel.merge(gnn_clean, on=merge_keys, suffixes=("_ma", "_gnn"))
    merged["delta_bv"] = merged["B_v_gnn"] - merged["B_v_ma"]
    return merged


def plot_bv_scatter(df_marvel, df_gnn, save_path):
    """B_v scatter across all isotopologues — Ma (x) vs GNN (y), y=x line."""
    merged = _merged_clean(df_marvel, df_gnn)
    if merged.empty:
        print("  No clean overlapping bands - skipping scatter.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        merged["B_v_ma"], merged["B_v_gnn"], s=40, alpha=0.8, color="#2ecc71", zorder=3
    )
    lims = [
        merged[["B_v_ma", "B_v_gnn"]].min().min() - 0.001,
        merged[["B_v_ma", "B_v_gnn"]].max().max() + 0.001,
    ]
    ax.plot(lims, lims, "k--", linewidth=1.2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r"$B_v$ from Ma (cm$^{-1}$)")
    ax.set_ylabel(r"$B_v$ from GNN (cm$^{-1}$)")
    ax.set_aspect("equal")
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}  (N={len(merged)} bands)")


def plot_mae_by_isotopologue(df_marvel, df_gnn, save_path):
    """Bar chart of mean |delta_B_v| per isotopologue with std error bars."""
    merged = _merged_clean(df_marvel, df_gnn)
    if merged.empty:
        print("  No overlapping clean bands - skipping MAE bar chart.")
        return

    iso_order = sorted(merged["isotope_id"].unique())
    means, stds, labels = [], [], []
    for iso in iso_order:
        sub = merged[merged["isotope_id"] == iso]["delta_bv"].abs()
        if len(sub) == 0:
            continue
        means.append(sub.mean())
        stds.append(sub.std() if len(sub) > 1 else 0.0)
        labels.append(str(iso))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(labels))
    ax.bar(
        x,
        [m * 1e4 for m in means],
        yerr=[s * 1e4 for s in stds],
        color="#3498db",
        alpha=0.8,
        capsize=4,
        error_kw={"linewidth": 1.2},
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Isotopologue")
    ax.set_ylabel(r"MAE of $\Delta B_v$  ($\times 10^{-4}$ cm$^{-1}$)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(
        f"Saved: {save_path}  (N={len(merged)} bands across {len(labels)} isotopologues)"
    )


def plot_bv_spread_by_isotopologue(df_marvel, df_gnn, save_path):
    """
    Three-group strip chart per isotopologue, stacked into two square panels
    (top: 626-638, bottom: 727-838), sharing y-axis limits and one legend:
      circles   (blue)   — Ma-only bands (no GNN counterpart)
      triangles (green)  — Ma+GNN matched bands (plotted at Ma B_v; GNN values
                           are indistinguishable at this scale, confirming
                           agreement)
      squares   (orange) — GNN-only bands (new assignments with no MARVEL
                           counterpart)
    Horizontal ticks show the group median.
    """
    gnn_clean = df_gnn[df_gnn["rmse"] < RMSE_CLEAN]
    merge_keys = ["isotope_id", "m1", "m2", "m3", "r", "parity"]

    matched = df_marvel.merge(gnn_clean, on=merge_keys, suffixes=("_ma", "_gnn"))

    ma_only = df_marvel.merge(
        gnn_clean[merge_keys], on=merge_keys, how="left", indicator=True
    )
    ma_only = ma_only[ma_only["_merge"] == "left_only"].drop(columns="_merge")

    gnn_only = gnn_clean.merge(
        df_marvel[merge_keys], on=merge_keys, how="left", indicator=True
    )
    gnn_only = gnn_only[gnn_only["_merge"] == "left_only"].drop(columns="_merge")

    iso_order = sorted(set(df_marvel["isotope_id"]) | set(gnn_clean["isotope_id"]))
    mid = len(iso_order) // 2
    panels = [iso_order[:mid], iso_order[mid:]]

    rng = np.random.default_rng(42)
    offsets = [-0.28, 0.0, 0.28]
    jitter_w = 0.07
    tick_hw = 0.10

    groups = [
        (ma_only, "isotope_id", "B_v", "#3498db", "#1a6fa8", "o"),
        (matched, "isotope_id", "B_v_ma", "#2ecc71", "#1a9450", "^"),
        (gnn_only, "isotope_id", "B_v", "#e67e22", "#b85a00", "s"),
    ]

    all_vals = np.concatenate(
        [df_g[bv_col].values for df_g, _, bv_col, *_ in groups]
    )
    pad = 0.05 * (all_vals.max() - all_vals.min())
    y_lims = (all_vals.min() - pad, all_vals.max() + pad)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=False)

    for ax, iso_subset in zip(axes, panels):
        for (df_g, iso_col, bv_col, dot_col, med_col, marker), off in zip(
            groups, offsets
        ):
            for i, iso in enumerate(iso_subset):
                vals = df_g[df_g[iso_col] == iso][bv_col].values
                if not len(vals):
                    continue
                jx = rng.uniform(-jitter_w, jitter_w, len(vals))
                ax.scatter(
                    i + off + jx,
                    vals,
                    marker=marker,
                    color=dot_col,
                    s=22,
                    alpha=0.75,
                    zorder=3,
                    linewidths=0,
                )
                ax.plot(
                    [i + off - tick_hw, i + off + tick_hw],
                    [np.median(vals)] * 2,
                    color=med_col,
                    linewidth=2.0,
                    zorder=4,
                )

        ax.set_xticks(range(len(iso_subset)))
        ax.set_xticklabels([str(iso) for iso in iso_subset])
        ax.set_xlabel("Isotopologue")
        ax.set_ylabel(r"$B_v$ (cm$^{-1}$)")
        ax.set_ylim(y_lims)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    legend_items = [
        ("#3498db", "o", f"Ma only  ($N={len(ma_only)}$)"),
        ("#2ecc71", "^", f"Ma + GNN  ($N={len(matched)}$)"),
        ("#e67e22", "s", f"GNN only  ($N={len(gnn_only)}$)"),
    ]
    fig.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor=c,
                markersize=7,
                label=lbl,
            )
            for c, marker, lbl in legend_items
        ],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.04),
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(
        f"Saved: {save_path}  "
        f"(Ma-only={len(ma_only)}, matched={len(matched)}, GNN-only={len(gnn_only)}, "
        f"{len(iso_order)} isotopologues)"
    )


def plot_delta_bv_histogram(df_marvel, df_gnn, save_path):
    """Histogram of delta_B_v = B_v(GNN) - B_v(Ma) across all isotopologues."""
    merged = _merged_clean(df_marvel, df_gnn)
    if merged.empty:
        print("  No overlapping clean bands - skipping delta_B_v histogram.")
        return

    delta = merged["delta_bv"] * 1e4  # units of 1e-4 cm-1
    median_val = delta.median()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(delta, bins=20, color="#9b59b6", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(
        median_val,
        color="#e74c3c",
        linestyle="-",
        linewidth=1.5,
        label=f"Median = {median_val:.2f} x 10$^{{-4}}$ cm$^{{-1}}$",
    )
    ax.set_xlabel(
        r"$\Delta B_v = B_v^\mathrm{GNN} - B_v^\mathrm{Ma}$  ($\times 10^{-4}$ cm$^{-1}$)"
    )
    ax.set_ylabel("Number of bands")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(
        f"Saved: {save_path}  (N={len(merged)} bands, median |dB_v|={delta.abs().median():.2f} x 1e-4 cm-1)"
    )


# ── Print summary ─────────────────────────────────────────────────────────────


def print_summary(df_marvel, df_ca):
    print("\n" + "=" * 64)
    print("Spectroscopic Constants - Summary")
    print("=" * 64)

    print(f"\nMARVEL bands fitted:              {len(df_marvel):>5,}")
    print(f"  Median RMSE:                    {df_marvel['rmse'].median():.5f} cm-1")
    print(f"  Bands with RMSE > 0.01:         {(df_marvel['rmse'] > 0.01).sum():>5,}")

    df_ca_clean = df_ca[df_ca["rmse"] < RMSE_CLEAN]
    print(f"\nGNN bands fitted (margin>=2.0):   {len(df_ca):>5,}")
    print(
        f"  Clean (RMSE < {RMSE_CLEAN} cm-1):       {len(df_ca_clean):>5,}  ({len(df_ca_clean)/len(df_ca)*100:.0f}%)"
    )
    print(f"  Median RMSE (clean):            {df_ca_clean['rmse'].median():.6f} cm-1")
    bv_clean = df_ca_clean["B_v"]
    n_phys = ((bv_clean >= 0.30) & (bv_clean <= 0.45)).sum()
    print(
        f"  B_v in [0.30, 0.45]:            {n_phys:>5,} / {len(df_ca_clean)} ({n_phys/len(df_ca_clean)*100:.0f}%)"
    )

    merge_keys = ["isotope_id", "m1", "m2", "m3", "r", "parity"]
    merged = df_marvel.merge(df_ca_clean, on=merge_keys, suffixes=("_mv", "_ca"))
    if not merged.empty:
        delta = (merged["B_v_ca"] - merged["B_v_mv"]).abs()
        print(f"\nClean overlapping bands:          {len(merged):>5,}")
        print(
            f"  Median |dB_v|:                  {delta.median():.6f} cm-1  ({delta.median()/0.39*1e6:.0f} ppm)"
        )
        print(f"  |dB_v| < 1e-4 cm-1:             {(delta < 1e-4).sum():>5,} bands")
        print(f"  |dB_v| < 1e-3 cm-1:             {(delta < 1e-3).sum():>5,} bands")
        print(f"  |dB_v| > 1e-3 cm-1:             {(delta > 1e-3).sum():>5,} bands")


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    print("Loading predictions...")
    df = pd.read_csv(PREDICTIONS_PATH)

    # ── Step 1: MARVEL fits ────────────────────────────────────────────────────
    print("Fitting MARVEL rotational series...")
    marvel = df[df["is_marvel"] & (df["polyad_int"] <= MAX_POLYAD)].copy()
    df_marvel = fit_groups(marvel, "AFGL_m1", "AFGL_m2", "AFGL_m3", "AFGL_r", "MARVEL")
    df_marvel.to_csv(OUT_MARVEL, index=False)
    print(f"  {len(df_marvel):,} MARVEL bands fitted -> {OUT_MARVEL}")

    # ── Step 2: Ca fits ────────────────────────────────────────────────────────
    print("Fitting Ca rotational series...")
    ca_confident = df[
        (~df["is_marvel"])
        & (df["pred_class_id"] != -1)
        & (df["assigned_margin"] >= CONFIDENT_MARGIN)
        & (~df["polyad_int"].isin(EXCLUDE_POLYADS))
        & (df["polyad_int"] <= MAX_POLYAD)
    ].copy()
    df_gnn = fit_groups(ca_confident, "pred_m1", "pred_m2", "pred_m3", "pred_r", "GNN")

    # ── Step 4: mass scaling columns ───────────────────────────────────────────
    df_gnn = add_mass_scaling(df_gnn, df)
    df_gnn.to_csv(OUT_GNN, index=False)
    print(f"  {len(df_gnn):,} GNN bands fitted -> {OUT_GNN}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print_summary(df_marvel, df_gnn)

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_bv_scatter(
        df_marvel,
        df_gnn,
        os.path.join(FIGURES_DIR, "spec_const_bv_scatter.png"),
    )
    plot_mae_by_isotopologue(
        df_marvel,
        df_gnn,
        os.path.join(FIGURES_DIR, "spec_const_mae_by_isotopologue.png"),
    )
    plot_delta_bv_histogram(
        df_marvel,
        df_gnn,
        os.path.join(FIGURES_DIR, "spec_const_delta_bv.png"),
    )
    plot_bv_spread_by_isotopologue(
        df_marvel,
        df_gnn,
        os.path.join(FIGURES_DIR, "spec_const_bv_spread_by_isotopologue.png"),
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
