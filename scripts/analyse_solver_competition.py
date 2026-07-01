"""
Solver competition asymmetry analysis — answers Open Q3 from uncertainty_analysis_log.md.

Investigates:
  1. Who wins block conflicts over MARVEL test states? (Ca Gen N vs MARVEL train/val)
  2. Do later bootstrap generations cause more conflicts? (bootstrap-inflation hypothesis)
  3. Which test states were raw-correct but degraded by the solver? (polyad breakdown)

Outputs
-------
- Console tables: winner type counts, per-state conflict rates, margin deltas
- data/figures/solver_competition.png: winner breakdown + logit-margin box plots
- data/figures/solver_degradation.png: degraded states by polyad + margin comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

DATA_DIR = "data"
PREDICTIONS_PATH = os.path.join(DATA_DIR, "assigned_co2_predictions.csv")
CLASS_MAPPING_PATH = os.path.join(DATA_DIR, "class_mapping.csv")
FIGURES_DIR = os.path.join(DATA_DIR, "figures")

mpl.rcParams.update({"font.size": 11, "legend.frameon": False})

BLOCK_KEYS = ["isotope_id", "J", "parity_encoded", "polyad_int"]

GEN_LABELS = {
    "marvel_train": "MARVEL train",
    "marvel_val": "MARVEL val",
    "ca_unharvested": "Ca (unharvested)",
    "ca_gen_1": "Ca Gen 1",
    "ca_gen_2": "Ca Gen 2",
    "ca_gen_3": "Ca Gen 3",
    "ca_gen_4": "Ca Gen 4",
    "ca_gen_5": "Ca Gen 5",
}

GEN_COLORS = {
    "marvel_train": "#2ecc71",
    "marvel_val": "#27ae60",
    "ca_unharvested": "#bdc3c7",
    "ca_gen_1": "#f39c12",
    "ca_gen_2": "#e67e22",
    "ca_gen_3": "#d35400",
    "ca_gen_4": "#c0392b",
    "ca_gen_5": "#96281b",
}


def _load_data():
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(f"{PREDICTIONS_PATH} not found. Run train.py first.")
    df = pd.read_csv(PREDICTIONS_PATH)

    # Build AFGL_class_id by joining class_mapping on (m1, m2, m3, r)
    mapping = pd.read_csv(CLASS_MAPPING_PATH)
    mapping_lookup = mapping.set_index(["m1", "m2", "m3", "r"])["class_id"]
    marvel_mask = df["is_marvel"]
    afgl_keys = df.loc[marvel_mask, ["AFGL_m1", "AFGL_m2", "AFGL_m3", "AFGL_r"]]
    df.loc[marvel_mask, "AFGL_class_id"] = afgl_keys.apply(
        lambda r: mapping_lookup.get((r.iloc[0], r.iloc[1], r.iloc[2], r.iloc[3]), -1),
        axis=1,
    ).values
    df["AFGL_class_id"] = df["AFGL_class_id"].fillna(-1).astype(int)

    # Correctness flags (meaningful only for MARVEL test states)
    df["is_correct_raw"] = marvel_mask & (df["raw_class_id"] == df["AFGL_class_id"])
    df["is_correct_post"] = (
        marvel_mask
        & (df["AFGL_m1"] == df["pred_m1"])
        & (df["AFGL_m2"] == df["pred_m2"])
        & (df["AFGL_m3"] == df["pred_m3"])
        & (df["AFGL_r"] == df["pred_r"])
    )

    # State type label (vectorized)
    if "assignment_generation" not in df.columns:
        df["assignment_generation"] = 0
    gen = df["assignment_generation"].fillna(0).astype(int)
    ca_gen_labels = "ca_gen_" + gen.astype(str)
    df["state_type"] = np.select(
        [
            df["is_marvel"] & df["train_mask"],
            df["is_marvel"] & df["val_mask"],
            df["is_marvel"],
            (~df["is_marvel"]) & (gen == 0),
        ],
        ["marvel_train", "marvel_val", "marvel_test", "ca_unharvested"],
        default=ca_gen_labels,
    )

    return df


def _find_conflict_winners(df):
    """
    For each conflicted MARVEL test state, find which state in the same block
    was assigned the class the test state wanted (raw_class_id).

    Returns a DataFrame with one row per conflict event:
      test_node_id, winner_node_id, winner_type, winner_gen,
      loser_logit_margin, winner_logit_margin, polyad_int,
      test_raw_correct, matched (bool)
    """
    # Index for fast winner lookup: (block_key, pred_class_id) → row
    pred_lookup = df[df["pred_class_id"] >= 0].set_index(BLOCK_KEYS + ["pred_class_id"])

    test_conflicts = df[df["test_mask"] & df["hungarian_conflict"]].copy()

    records = []
    for _, row in test_conflicts.iterrows():
        block_key = tuple(row[k] for k in BLOCK_KEYS)
        wanted_class = int(row["raw_class_id"])
        lookup_key = block_key + (wanted_class,)

        matched = lookup_key in pred_lookup.index
        if matched:
            winner = pred_lookup.loc[lookup_key]
            # loc can return Series (1 match) or DataFrame (multiple); take first
            if isinstance(winner, pd.DataFrame):
                winner = winner.iloc[0]
            records.append(
                {
                    "test_node_id": row["node_id"],
                    "winner_node_id": winner["node_id"],
                    "winner_type": winner["state_type"],
                    "winner_gen": int(winner.get("assignment_generation", 0)),
                    "loser_logit_margin": float(row["logit_margin"]),
                    "winner_logit_margin": float(winner["logit_margin"]),
                    "polyad_int": int(row["polyad_int"]),
                    "test_raw_correct": bool(row["is_correct_raw"]),
                    "matched": True,
                }
            )
        else:
            records.append(
                {
                    "test_node_id": row["node_id"],
                    "winner_node_id": -1,
                    "winner_type": "unmatched",
                    "winner_gen": -1,
                    "loser_logit_margin": float(row["logit_margin"]),
                    "winner_logit_margin": float("nan"),
                    "polyad_int": int(row["polyad_int"]),
                    "test_raw_correct": bool(row["is_correct_raw"]),
                    "matched": False,
                }
            )

    return pd.DataFrame(records)


def analyse_competition(df, winners):
    n_test = df["test_mask"].sum()
    n_conflict = winners["matched"].count()  # total conflicts
    n_matched = winners["matched"].sum()
    n_unmatched = (~winners["matched"]).sum()

    print(f"\n{'='*70}")
    print("=== SOLVER COMPETITION ASYMMETRY ANALYSIS ===")
    print(f"{'='*70}")
    print(f"MARVEL test states:   {n_test:,}")
    print(f"Conflicted (total):   {n_conflict:,} ({n_conflict/n_test*100:.1f}%)")
    print(f"  Winner identified:  {n_matched:,}")
    print(f"  Unmatched:          {n_unmatched:,}")

    matched = winners[winners["matched"]].copy()

    # ── Table A: Winner type breakdown ───────────────────────────────────────
    # Count of each state type in the full dataset (for per-state conflict rate)
    type_counts_full = df["state_type"].value_counts()

    winner_counts = matched["winner_type"].value_counts()
    print(f"\n--- Winner Type Breakdown (N={n_matched} matched conflicts) ---")
    print(
        f"{'Winner Type':<22} {'N wins':>8} {'% of wins':>10} {'N states (total)':>18} {'Wins per 1k states':>20}"
    )
    for wtype, n_wins in winner_counts.items():
        n_total_that_type = type_counts_full.get(wtype, 0)
        per_1k = (
            (n_wins / n_total_that_type * 1000)
            if n_total_that_type > 0
            else float("nan")
        )
        print(
            f"  {wtype:<20} {n_wins:>8,} {n_wins/n_matched*100:>9.1f}% {n_total_that_type:>18,} {per_1k:>19.2f}"
        )

    # ── Table B: Logit margin of winners by generation ───────────────────────
    ca_winners = matched[matched["winner_type"].str.startswith("ca_gen")].copy()
    if len(ca_winners) > 0:
        print(f"\n--- Ca Winner Logit Margins by Generation ---")
        print(
            f"{'Gen':<10} {'N wins':>8} {'Mean margin':>13} {'Median':>8} {'P25':>6} {'P75':>6}"
        )
        for gen_type in sorted(ca_winners["winner_type"].unique()):
            sub = ca_winners[ca_winners["winner_type"] == gen_type]
            m = sub["winner_logit_margin"]
            print(
                f"  {gen_type:<8} {len(sub):>8,} {m.mean():>13.3f} {m.median():>8.3f} {m.quantile(0.25):>6.3f} {m.quantile(0.75):>6.3f}"
            )

    # ── Margin delta: winner - loser ─────────────────────────────────────────
    matched["margin_delta"] = (
        matched["winner_logit_margin"] - matched["loser_logit_margin"]
    )
    print(f"\n--- Margin Delta: Winner minus Loser logit_margin ---")
    print(f"  Mean delta:   {matched['margin_delta'].mean():.3f}")
    print(f"  Median delta: {matched['margin_delta'].median():.3f}")
    print(
        f"  Delta > 0 (winner more confident): {(matched['margin_delta'] > 0).mean()*100:.1f}%"
    )
    print(
        f"  Delta < 0 (loser was more confident): {(matched['margin_delta'] < 0).mean()*100:.1f}%"
    )

    # Stratify delta by winner generation
    print(f"\n  Margin delta by winner generation:")
    for gen_type in sorted(matched["winner_type"].unique()):
        sub = matched[matched["winner_type"] == gen_type]
        d = sub["margin_delta"]
        pct_pos = (d > 0).mean() * 100
        print(
            f"    {gen_type:<22} mean={d.mean():+.3f}  pct_pos={pct_pos:.1f}%  n={len(sub)}"
        )

    return matched


def analyse_degradation(df, winners):
    """Analyse raw-correct → post-wrong degradation cases."""
    test_df = df[df["test_mask"]].copy()
    degraded = test_df[test_df["is_correct_raw"] & ~test_df["is_correct_post"]]
    total_test = len(test_df)
    total_raw_correct = test_df["is_correct_raw"].sum()

    print(f"\n{'='*70}")
    print("=== SOLVER DEGRADATION DIAGNOSIS ===")
    print(f"{'='*70}")
    print(
        f"Raw-correct test states:    {total_raw_correct:,} / {total_test:,} ({total_raw_correct/total_test*100:.2f}%)"
    )
    print(
        f"Degraded (raw-ok, post-no): {len(degraded):,} ({len(degraded)/total_test*100:.2f}% of test set)"
    )
    print(
        f"  = {len(degraded)/total_raw_correct*100:.2f}% of raw-correct states were degraded by the solver"
    )

    # By polyad
    print(f"\n--- Degraded States by Polyad (top 10) ---")
    poly_counts = degraded["polyad_int"].value_counts().head(10)
    poly_total = test_df.groupby("polyad_int")["is_correct_raw"].sum()
    print(
        f"{'Polyad':>8} {'Degraded':>10} {'Raw-correct in polyad':>22} {'Degradation rate':>18}"
    )
    for polyad, n_deg in poly_counts.items():
        n_raw_corr = poly_total.get(polyad, 0)
        rate = n_deg / n_raw_corr * 100 if n_raw_corr > 0 else float("nan")
        print(f"  {polyad:>6} {n_deg:>10,} {n_raw_corr:>22,} {rate:>17.1f}%")

    # Winner type for degraded cases
    degraded_node_ids = set(degraded["node_id"].values)
    degraded_winners = winners[
        winners["matched"] & winners["test_node_id"].isin(degraded_node_ids)
    ].copy()

    if len(degraded_winners) > 0:
        print(f"\n--- Winner Types for Degraded Cases (n={len(degraded_winners)}) ---")
        for wtype, cnt in degraded_winners["winner_type"].value_counts().items():
            print(f"  {wtype:<25} {cnt:>5,} ({cnt/len(degraded_winners)*100:.1f}%)")

        deg_delta = (
            degraded_winners["winner_logit_margin"]
            - degraded_winners["loser_logit_margin"]
        )
        print(f"\n  Margin delta (winner minus degraded loser):")
        print(f"    Mean:   {deg_delta.mean():+.3f}")
        print(f"    Median: {deg_delta.median():+.3f}")
        print(f"    Winner > Loser: {(deg_delta > 0).mean()*100:.1f}%")

    return degraded, degraded_winners


def plot_competition(df, matched):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel A: Winner type breakdown ───────────────────────────────────────
    winner_counts = matched["winner_type"].value_counts()
    labels = [GEN_LABELS.get(wt, wt) for wt in winner_counts.index]
    colors = [GEN_COLORS.get(wt, "#95a5a6") for wt in winner_counts.index]
    fracs = winner_counts.values / winner_counts.sum() * 100

    bars = ax1.barh(labels, fracs, color=colors, alpha=0.85, edgecolor="white")
    for bar, pct in zip(bars, fracs):
        ax1.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=9,
        )
    ax1.set_xlabel("% of matched conflicts")
    ax1.set_title(
        "Conflict Winner Breakdown\n(who displaced MARVEL test state)",
        fontweight="bold",
    )
    ax1.set_xlim(0, max(fracs) + 12)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.invert_yaxis()

    # ── Panel B: Logit margin box plots by winner type ────────────────────────
    types_present = [
        t
        for t in [
            "marvel_train",
            "marvel_val",
            "ca_gen_1",
            "ca_gen_2",
            "ca_gen_3",
            "ca_gen_4",
            "ca_gen_5",
        ]
        if t in matched["winner_type"].values
    ]

    data_by_type = [
        matched.loc[matched["winner_type"] == t, "winner_logit_margin"].values
        for t in types_present
    ]
    box_labels = [GEN_LABELS.get(t, t) for t in types_present]
    box_colors = [GEN_COLORS.get(t, "#95a5a6") for t in types_present]

    bp = ax2.boxplot(
        data_by_type,
        patch_artist=True,
        vert=True,
        medianprops={"color": "black", "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax2.set_xticks(range(1, len(box_labels) + 1))
    ax2.set_xticklabels(box_labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("logit_margin (pre-solver confidence)")
    ax2.set_title(
        "Winner Pre-Solver Confidence by Type\n(higher = more dominant in block)",
        fontweight="bold",
    )
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Add reference line: median loser margin
    loser_median = matched["loser_logit_margin"].median()
    ax2.axhline(
        loser_median,
        color="steelblue",
        linestyle="--",
        linewidth=1.5,
        label=f"Loser median ({loser_median:.2f})",
    )
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Solver Competition Asymmetry — MARVEL Test State Conflict Winners",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_path = os.path.join(FIGURES_DIR, "solver_competition.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {save_path}")


def plot_degradation(df, degraded, degraded_winners):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel A: Degraded states by polyad ───────────────────────────────────
    test_df = df[df["test_mask"]].copy()
    poly_deg = degraded.groupby("polyad_int").size()
    poly_raw = test_df.groupby("polyad_int")["is_correct_raw"].sum()
    poly_all = poly_deg.reindex(poly_raw.index, fill_value=0)
    polyads = sorted(poly_raw.index)

    x = np.arange(len(polyads))
    w = 0.4
    raw_vals = [int(poly_raw.get(p, 0)) for p in polyads]
    deg_vals = [int(poly_all.get(p, 0)) for p in polyads]

    ax1.bar(x - w / 2, raw_vals, w, label="Raw-correct", color="#2ecc71", alpha=0.7)
    ax1.bar(
        x + w / 2, deg_vals, w, label="Degraded by solver", color="#e74c3c", alpha=0.7
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(p) for p in polyads])
    ax1.set_xlabel("Polyad")
    ax1.set_ylabel("Count")
    ax1.set_title(
        "Raw-correct vs Solver-degraded States\nby Polyad (MARVEL test set)",
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Panel B: Paired margin comparison for degraded cases ─────────────────
    if len(degraded_winners) > 0:
        loser_margins = degraded_winners["loser_logit_margin"].values
        winner_margins = degraded_winners["winner_logit_margin"].dropna().values
        # Violin plots side by side
        parts = ax2.violinplot(
            [loser_margins, winner_margins],
            positions=[1, 2],
            showmedians=True,
            showextrema=True,
        )
        colors_v = ["#3498db", "#e74c3c"]
        for pc, col in zip(parts["bodies"], colors_v):
            pc.set_facecolor(col)
            pc.set_alpha(0.7)
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(
            ["Degraded test state\n(loser)", "Winning state\n(displaced it)"]
        )
        ax2.set_ylabel("logit_margin (pre-solver)")
        ax2.set_title(
            f"Pre-solver Confidence: Degraded vs Winner\n(n={len(degraded_winners)} cases)",
            fontweight="bold",
        )
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

        # Annotate medians
        for pos, vals, col in [
            (1, loser_margins, "#3498db"),
            (2, winner_margins, "#e74c3c"),
        ]:
            med = np.median(vals)
            ax2.text(
                pos, med + 0.05, f"med={med:.2f}", ha="center", fontsize=9, color=col
            )
    else:
        ax2.text(
            0.5,
            0.5,
            "No degraded winners found",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    fig.suptitle(
        "Solver Degradation Diagnosis — Raw-correct → Post-solver-wrong",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "solver_degradation.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    df = _load_data()
    winners = _find_conflict_winners(df)
    matched = analyse_competition(df, winners)
    degraded, degraded_winners = analyse_degradation(df, winners)
    plot_competition(df, matched)
    plot_degradation(df, degraded, degraded_winners)
    print("\nDone.")


if __name__ == "__main__":
    main()
