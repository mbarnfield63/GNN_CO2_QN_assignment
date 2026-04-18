import pandas as pd
import numpy as np
import os
import concurrent.futures
from sklearn.model_selection import train_test_split

# Import from your config file
from config import STATES_DIR, ISOTOPES

# The original ExoMol .states columns
EXOMOL_COLUMNS = [
    "ID",
    "E",
    "gtot",
    "J",
    "unc",
    "tau",
    "tot_sym",
    "e_f",
    "hzb_v1",
    "hzb_v2",
    "hzb_l2",
    "hzb_v3",
    "Trove_coeff",
    "AFGL_m1",
    "AFGL_m2",
    "AFGL_l2",
    "AFGL_m3",
    "AFGL_r",
    "Trove_v1",
    "Trove_v2",
    "Trove_v3",
    "Source",
    "E_Ca",
]

UNIFIED_DATASET_PATH = "data/unified_co2_graph_data.csv"
CLASS_MAPPING_PATH = "data/class_mapping.csv"
ENERGY_CUTOFF = 15000.0


def process_single_isotope(iso_config):
    """Parses a single .states file, applies the energy cutoff, filters sources, and formats features."""
    filepath = os.path.join(STATES_DIR, iso_config["file"])

    if not os.path.exists(filepath):
        print(f"Warning: File not found {filepath}")
        return pd.DataFrame()

    print(f"Processing {iso_config['id']}...")

    # Read the text file (Allowing NaNs to exist on the first pass)
    df = pd.read_csv(filepath, header=None, sep=r"\s+", names=EXOMOL_COLUMNS)

    # Filter out 'EH' and 'HI' - keep only valid MARVEL and Calculated sources
    valid_sources = ["Ma", "MA", "Ca", "CA"]
    df = df[df["Source"].isin(valid_sources)].copy()

    # Apply Energy Cutoff to save memory
    df = df[df["E"] <= ENERGY_CUTOFF].copy()

    # Map 'is_marvel' flag (True for Ma, False for Ca)
    df["is_marvel"] = df["Source"].isin(["Ma", "MA"])

    # Encode e/f Parity (e -> 0, f -> 1)
    df["parity_encoded"] = df["e_f"].map({"e": 0, "f": 1}).fillna(-1).astype(int)

    # Append the structural context features from config
    df["isotope_id"] = iso_config["id"]
    df["is_symmetric"] = iso_config["is_symmetric"]
    df["C_mass"] = iso_config["C_mass"]
    df["O_A_mass"] = iso_config["O_A_mass"]
    df["O_B_mass"] = iso_config["O_B_mass"]

    # Rename columns to match what graph_builder.py expects
    df = df.rename(
        columns={
            "E": "energy",
            "Trove_coeff": "dom_coeff",
            "Trove_v1": "t1",
            "Trove_v2": "t2",
            "Trove_v3": "t3",
        }
    )

    # Keep only necessary columns
    keep_cols = [
        "energy",
        "J",
        "parity_encoded",
        "dom_coeff",
        "t1",
        "t2",
        "t3",
        "AFGL_m1",
        "AFGL_m2",
        "AFGL_m3",
        "AFGL_r",
        "isotope_id",
        "is_symmetric",
        "C_mass",
        "O_A_mass",
        "O_B_mass",
        "is_marvel",
    ]
    df = df[keep_cols]
    df["polyad"] = (2 * df["t1"]) + df["t2"]

    # Assign AFGL labels for inference states (Ca) with -1
    afgl_cols = ["AFGL_m1", "AFGL_m2", "AFGL_m3", "AFGL_r"]
    df.loc[~df["is_marvel"], afgl_cols] = -1

    # Filter for NaN AFTER dropping unnecessary columns
    df = df.dropna()

    return df


def create_unified_dataset():
    """Compiles all isotopes, generates combinatorial classes from MARVEL states, and creates masks."""
    print(f"Aggregating states with E <= {ENERGY_CUTOFF} cm⁻¹...")

    # 1. Multi-process the file reading
    dfs = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_single_isotope, ISOTOPES)
        for res_df in results:
            if not res_df.empty:
                dfs.append(res_df)

    if not dfs:
        raise ValueError("No data found. Check your STATES_DIR and ISOTOPES config.")

    # 2. Concatenate into a single dataframe
    full_df = pd.concat(dfs, ignore_index=True)

    # 3. Create Combinatorial Classes based ONLY on MARVEL data
    marvel_df = full_df[full_df["is_marvel"]].copy()

    # Create a string representation of the 4 quantum numbers
    marvel_class_strings = (
        marvel_df["AFGL_m1"].astype(int).astype(str)
        + "_"
        + marvel_df["AFGL_m2"].astype(int).astype(str)
        + "_"
        + marvel_df["AFGL_m3"].astype(int).astype(str)
        + "_"
        + marvel_df["AFGL_r"].astype(int).astype(str)
    )

    # Get unique classes and map them to integers starting from 0
    unique_classes = marvel_class_strings.unique()
    class_to_id = {cls_str: i for i, cls_str in enumerate(unique_classes)}
    print(
        f"Mapped {len(unique_classes)} unique physical combinations (m1, m2, m3, r) from MARVEL states."
    )

    # Export the mapping dictionary so we can decode predictions later
    mapping_data = []
    for cls_str, cls_id in class_to_id.items():
        m1, m2, m3, r = map(int, cls_str.split("_"))
        mapping_data.append([cls_id, m1, m2, m3, r])

    mapping_df = pd.DataFrame(mapping_data, columns=["class_id", "m1", "m2", "m3", "r"])
    mapping_df.to_csv(CLASS_MAPPING_PATH, index=False)
    print(f"Saved class decoding map to {CLASS_MAPPING_PATH}")

    # 4. Apply mapping to the full dataset
    def assign_class(row):
        if not row["is_marvel"]:
            return -1  # Dummy label for inference nodes (Calculated data)

        cls_str = f"{int(row['AFGL_m1'])}_{int(row['AFGL_m2'])}_{int(row['AFGL_m3'])}_{int(row['AFGL_r'])}"
        return class_to_id.get(cls_str, -1)

    full_df["combinatorial_class_id"] = full_df.apply(assign_class, axis=1)

    # 5. Create Stratified Masks for Transductive Learning
    marvel_idx = full_df[full_df["is_marvel"]].index
    isotope_labels = full_df.loc[marvel_idx, "isotope_id"]

    # 80% Train, 20% Temp (Stratified by isotope to ensure minor isotopes are represented)
    train_idx, temp_idx = train_test_split(
        marvel_idx, test_size=0.2, stratify=isotope_labels, random_state=42
    )
    # Split Temp into 10% Val, 10% Test
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=full_df.loc[temp_idx, "isotope_id"],
        random_state=42,
    )

    # Initialize all masks to False
    full_df["train_mask"] = False
    full_df["val_mask"] = False
    full_df["test_mask"] = False

    # Apply True to the split indices (Note: Non-MARVEL data remains all False)
    full_df.loc[train_idx, "train_mask"] = True
    full_df.loc[val_idx, "val_mask"] = True
    full_df.loc[test_idx, "test_mask"] = True

    # 6. Generate continuous node_ids for PyG edges
    full_df = full_df.reset_index(drop=True)
    full_df["node_id"] = full_df.index

    # 7. Save to disk
    print(f"Saving unified dataset to {UNIFIED_DATASET_PATH}...")
    full_df.to_csv(UNIFIED_DATASET_PATH, index=False)
    print("Dataset generated successfully.")

    return full_df


def load_and_preprocess_states():
    """Entry point for train.py. Loads the unified dataset, generating it first if needed."""
    if os.path.exists(UNIFIED_DATASET_PATH):
        print("Loading pre-processed unified dataset...")
        return pd.read_csv(UNIFIED_DATASET_PATH)
    else:
        print("Unified dataset not found. Generating now...")
        return create_unified_dataset()


if __name__ == "__main__":
    create_unified_dataset()
