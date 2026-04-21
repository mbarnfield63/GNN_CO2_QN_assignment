import pandas as pd
import numpy as np
import torch
import time
from torch_geometric.data import Data


def build_pyg_graph(df: pd.DataFrame) -> Data:
    print("Building edges using memory-safe NumPy vectorization...")
    start_time = time.time()

    df_logic = df[
        ["node_id", "t1", "t2", "t3", "isotope_id", "J", "parity_encoded", "energy"]
    ].copy()
    src_list, dst_list = [], []

    # ------------------------------------------------------------------
    # 1. INTER-ISOTOPE & PERTURBATION EDGES (1D Energy Chains)
    # ------------------------------------------------------------------
    grouped_inter = df_logic.groupby(["t1", "t2", "t3", "J", "parity_encoded"])
    for _, group in grouped_inter:
        if len(group) > 1:
            group = group.sort_values("energy")
            n_ids = group["node_id"].values

            src, dst = n_ids[:-1], n_ids[1:]
            src_list.extend([src, dst])
            dst_list.extend([dst, src])

    # ------------------------------------------------------------------
    # 2. INTRA-ISOTOPE EDGES (Nearest Energy Rotational Ladders)
    # ------------------------------------------------------------------
    df_logic = df_logic.sort_values("energy")
    df_target = df_logic.copy()
    df_target["J_target"] = df_target["J"] - 1

    intra_merge = pd.merge_asof(
        df_logic,
        df_target,
        on="energy",
        by=["t1", "t2", "t3", "isotope_id"],
        direction="nearest",
        suffixes=("_src", "_dst"),
    )

    intra_merge = intra_merge[intra_merge["J_src"] == intra_merge["J_target"]]
    intra_merge = intra_merge.dropna(subset=["node_id_dst"])

    intra_src = intra_merge["node_id_src"].astype(int).values
    intra_dst = intra_merge["node_id_dst"].astype(int).values

    src_list.extend([intra_src, intra_dst])
    dst_list.extend([intra_dst, intra_src])

    # ------------------------------------------------------------------
    # 3. COMBINE AND FORMAT
    # ------------------------------------------------------------------
    if src_list:
        all_src = np.concatenate(src_list)
        all_dst = np.concatenate(dst_list)
        edges = np.unique(np.vstack((all_src, all_dst)), axis=1)
    else:
        edges = np.empty((2, 0), dtype=np.int64)

    edge_index = torch.tensor(edges, dtype=torch.long).contiguous()
    print(
        f"Graph constructed in {time.time() - start_time:.2f}s. Total edges: {edge_index.shape[1]:,}"
    )

    # ------------------------------------------------------------------
    # 4. NODE FEATURES (No AFGL labels allowed!)
    # ------------------------------------------------------------------
    feature_cols = [
        "energy",
        "J",
        "parity_encoded",
        "dom_coeff",
        "t1",
        "t2",
        "t3",
        "polyad",
        "is_symmetric",
        "C_mass",
        "O_A_mass",
        "O_B_mass",
    ]

    x = torch.tensor(df[feature_cols].values, dtype=torch.float).contiguous()
    y = torch.tensor(df["combinatorial_class_id"].values, dtype=torch.long).contiguous()
    iso_idx_tensor = torch.tensor(
        df["iso_idx_encoded"].values, dtype=torch.long
    ).contiguous()

    train_mask = torch.tensor(df["train_mask"].values, dtype=torch.bool).contiguous()
    val_mask = torch.tensor(df["val_mask"].values, dtype=torch.bool).contiguous()
    test_mask = torch.tensor(df["test_mask"].values, dtype=torch.bool).contiguous()
    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        iso_idx=iso_idx_tensor,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
