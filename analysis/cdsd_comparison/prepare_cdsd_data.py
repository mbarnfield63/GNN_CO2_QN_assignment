"""
Patches the 626 (12C-16O2) raw states file so that the CDSD-2024-PI-derived
(`Source == "EH"`) states are treated as ordinary unlabeled Ca states by the
pipeline, and stashes their true CDSD-assigned AFGL quantum numbers separately
for scoring after the isolated side-run (run_side_pipeline.py) completes.

Only 626 uses CDSD-2024-PI as its EH source; every other isotopologue's
EH/HI tag comes from HITRAN2020, so this script is 626-only by design.

The patched Ca rows use E_Ca (the original TROVE-calculated energy), not E
(the CDSD-corrected energy) -- otherwise these states would carry more
precise energies than genuine Ca states, leaking CDSD information through
the energy feature itself and making the comparison less honest.
"""

import os
import pandas as pd

RAW_STATES_PATH = r"C:\Code\_raw_data_store\Triatomics\CO2\12C-16O2__Dozen.states.cut"

HERE = os.path.dirname(os.path.abspath(__file__))
PATCHED_STATES_PATH = os.path.join(
    HERE, "states_patched", "12C-16O2__Dozen_cdsd_patched.states.cut"
)
TRUE_LABELS_PATH = os.path.join(HERE, "cdsd_true_labels.csv")

# Same column layout as src/dataset.py's EXOMOL_COLUMNS
EXOMOL_COLUMNS = [
    "ID", "E", "gtot", "J", "unc", "tau", "tot_sym", "e_f",
    "hzb_v1", "hzb_v2", "hzb_l2", "hzb_v3", "Trove_coeff",
    "AFGL_m1", "AFGL_m2", "AFGL_l2", "AFGL_m3", "AFGL_r",
    "Trove_v1", "Trove_v2", "Trove_v3", "Source", "E_Ca",
]


def main():
    df = pd.read_csv(RAW_STATES_PATH, header=None, sep=r"\s+", names=EXOMOL_COLUMNS)

    eh_mask = df["Source"] == "EH"
    print(f"Found {eh_mask.sum():,} CDSD-2024-PI (EH) states in 626.")

    true_labels = df.loc[
        eh_mask, ["ID", "E_Ca", "J", "e_f", "AFGL_m1", "AFGL_m2", "AFGL_m3", "AFGL_r"]
    ].rename(
        columns={
            "ID": "state_id",
            "E_Ca": "energy",
            "e_f": "parity",
            "AFGL_m1": "m1_true",
            "AFGL_m2": "m2_true",
            "AFGL_m3": "m3_true",
            "AFGL_r": "r_true",
        }
    )
    true_labels.to_csv(TRUE_LABELS_PATH, index=False)
    print(f"Stashed {len(true_labels):,} true CDSD labels -> {TRUE_LABELS_PATH}")

    patched = df.copy()
    patched.loc[eh_mask, "E"] = patched.loc[eh_mask, "E_Ca"]
    patched.loc[eh_mask, "Source"] = "Ca"
    patched.loc[eh_mask, ["AFGL_m1", "AFGL_m2", "AFGL_l2", "AFGL_m3", "AFGL_r"]] = -1

    os.makedirs(os.path.dirname(PATCHED_STATES_PATH), exist_ok=True)
    # na_rep="NaN": the raw file uses the literal string "NaN" as a placeholder
    # (e.g. the ground state's tau field). pandas parses that into a real NaN on
    # read; writing NaN back out as an empty field would then vanish entirely
    # under dataset.py's sep=r"\s+" re-read, shifting every later column left.
    patched.to_csv(PATCHED_STATES_PATH, sep=" ", header=False, index=False, na_rep="NaN")
    print(f"Patched states file written -> {PATCHED_STATES_PATH}")


if __name__ == "__main__":
    main()
