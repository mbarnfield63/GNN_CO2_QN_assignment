"""
Decision tree: AFGL + TROVE local-mode quanta → Herzberg labels for asymmetric CO2 isotopologues.

Scrapes raw .states.cut files for Ma states, trains a DecisionTreeClassifier on a
70/20/10 split, and reports test accuracy. Expected result: 100% (the mapping is a
deterministic rule set that the tree captures exactly).

Run from GNN_CO2_QN_assignment/:
    uv run scripts/train_asymmetric_dt.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from config import ISOTOPES, STATES_DIR
from dataset import EXOMOL_COLUMNS

AFGL_COLS = ["AFGL_m1", "AFGL_m2", "AFGL_l2", "AFGL_m3", "AFGL_r"]
TROVE_COLS = ["Trove_v1", "Trove_v2", "Trove_v3"]  # t1, t2, t3 in paper notation
HZB_COLS = ["hzb_v1", "hzb_v2", "hzb_l2", "hzb_v3"]
FEATURE_COLS = AFGL_COLS + TROVE_COLS


def load_asymmetric_marvel() -> pd.DataFrame:
    dfs = []
    for iso in ISOTOPES:
        if iso["is_symmetric"]:
            continue
        path = os.path.join(STATES_DIR, iso["file"])
        if not os.path.exists(path):
            print(f"  Warning: missing {path}")
            continue
        df = pd.read_csv(path, header=None, sep=r"\s+", names=EXOMOL_COLUMNS)
        df = df[df["Source"].isin(["Ma", "MA"])].copy()
        df["isotope_id"] = iso["id"]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


df = load_asymmetric_marvel()

# Keep only rows with complete AFGL and Herzberg labels
df = df[FEATURE_COLS + HZB_COLS].dropna()

# Encode the 4-QN Herzberg target as a single joint label
df["hzb_label"] = df[HZB_COLS].astype(int).astype(str).agg(",".join, axis=1)

X = df[FEATURE_COLS].astype(int).values
y = df["hzb_label"].values

print(f"Ma asymmetric states with complete labels: {len(X)}")
print(f"Unique Herzberg (v1,v2,l2,v3) combinations: {len(set(y))}")

# 70 / 20 / 10 split
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.2222, random_state=42
)

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

val_acc = accuracy_score(y_val, dt.predict(X_val))
test_acc = accuracy_score(y_test, dt.predict(X_test))

print(f"Val accuracy:  {val_acc * 100:.2f}%")
print(f"Test accuracy: {test_acc * 100:.2f}%")

y_pred = dt.predict(X_test)
mask = y_pred != y_test
if mask.any():
    wrong = pd.DataFrame(X_test[mask], columns=FEATURE_COLS)
    wrong["true"] = y_test[mask]
    wrong["pred"] = y_pred[mask]
    print(f"\nMisclassified ({mask.sum()} state(s)):")
    print(wrong.to_string(index=False))
    print(
        "Note: inspect true label for l2 > v2 physical inconsistency in MARVEL source data."
    )
else:
    print("Self-check passed: 100% test accuracy confirmed.")
