"""
src/dataset.py -- Phase 8 refactor of 06_dataset_class.ipynb.

Diabetic foot thermography project (Physics-Informed GAT, dual-branch architecture).

This module is a straight extraction of already-validated, already-working code from
06_dataset_class.ipynb (Phase 7) -- no logic has been rewritten, modified, or "improved":

- `PlantarThermographyDataset` is copied EXACTLY from that notebook's Section 3.
- The manifest-merging logic in `_build_combined_manifest()` is copied EXACTLY from that
  notebook's Section 1 (reading split_manifest.csv and graph_manifest.csv, merging on
  ['subject_id', 'foot_side'], and verifying group/gender/split agree across both files
  before trusting the merge).
- `load_normalization_stats()` reads data/processed/normalization_stats.json, which was
  computed ONCE from the train split by 06_dataset_class.ipynb Section 2 and is never
  recomputed here (recomputing it anywhere else would risk silently leaking val/test
  statistics into the model).
- `build_dataloaders()` wraps the merge + dataset-building + dataloader-building steps
  (Sections 1, 4, and 6 of the notebook) into a single function, doing the same
  split-filtering the notebook did manually.

Phase 8 onward should `from src.dataset import build_dataloaders, load_normalization_stats`
rather than re-pasting this code into every subsequent notebook.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------------------
# Copied EXACTLY from 06_dataset_class.ipynb, Section 3 ("Dataset Class"). Do not modify --
# this logic is already validated (batch sanity check + label cross-check both passed).
# ---------------------------------------------------------------------------------------
class PlantarThermographyDataset(Dataset):
    """One row per subject+foot. Returns both branches of the dual-branch architecture:
    the 4-node angiosome graph (Phase 6) and the pixel images (Phases 4-5), normalized
    using train-derived statistics supplied via `normalization_stats` (never refit here).
    """

    def __init__(self, manifest_df, normalization_stats):
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.stats = normalization_stats

        self._node_feature_mean = np.array(normalization_stats["node_feature_mean"], dtype=np.float32)
        self._node_feature_std = np.array(normalization_stats["node_feature_std"], dtype=np.float32)
        self._raw_tissue_mean = normalization_stats["raw_tissue_mean"]
        self._raw_tissue_std = normalization_stats["raw_tissue_std"]
        self._relative_tissue_mean = normalization_stats["relative_tissue_mean"]
        self._relative_tissue_std = normalization_stats["relative_tissue_std"]

    def __len__(self):
        return len(self.manifest_df)

    def _normalize_image(self, arr, mask, mean, std):
        """Z-score tissue pixels using train-derived mean/std; force background to exactly 0.0
        AFTER normalization, since (0 - mean) / std is not 0 in general."""
        out = np.zeros_like(arr, dtype=np.float32)
        out[mask] = (arr[mask] - mean) / std
        return out

    def __getitem__(self, idx):
        row = self.manifest_df.iloc[idx]

        # --- Graph branch (Phase 6 .npz) ---
        with np.load(row["npz_path"]) as npz:
            node_features_raw = npz["node_features"].astype(np.float32)  # (4, 6)
            edge_index_base = npz["edge_index_base"].astype(np.int64)    # (2, 8)
            edge_attr_base = npz["edge_attr_base"].astype(np.float32)    # (8, 2)
            edge_index_diag = npz["edge_index_diag"].astype(np.int64)    # (2, 12)
            edge_attr_diag = npz["edge_attr_diag"].astype(np.float32)    # (12, 2)

        node_features = (node_features_raw - self._node_feature_mean) / self._node_feature_std

        # --- Image branch (Phases 4-5) ---
        mask = np.load(row["mask_path"])                     # boolean, (208, 96)
        raw_arr = np.load(row["raw_array_path"])              # float32, (208, 96)
        rel_arr = np.load(row["relative_array_path"])          # float32, (208, 96)
        colormap_arr = np.array(Image.open(row["colormap_png_path"]))  # uint8, (208, 96, 3)

        raw_norm = self._normalize_image(raw_arr, mask, self._raw_tissue_mean, self._raw_tissue_std)
        rel_norm = self._normalize_image(rel_arr, mask, self._relative_tissue_mean, self._relative_tissue_std)
        colormap_norm = (colormap_arr.astype(np.float32) / 255.0).transpose(2, 0, 1)  # (3, 208, 96)

        sample = {
            "node_features": torch.from_numpy(node_features.astype(np.float32)),
            "edge_index_base": torch.from_numpy(edge_index_base),
            "edge_attr_base": torch.from_numpy(edge_attr_base),
            "edge_index_diag": torch.from_numpy(edge_index_diag),
            "edge_attr_diag": torch.from_numpy(edge_attr_diag),
            "raw_image": torch.from_numpy(raw_norm).unsqueeze(0),           # (1, 208, 96)
            "relative_image": torch.from_numpy(rel_norm).unsqueeze(0),      # (1, 208, 96)
            "colormap_image": torch.from_numpy(colormap_norm),              # (3, 208, 96)
            "tissue_mask": torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),  # (1, 208, 96)
            "label": torch.tensor(1 if row["group"] == "DM" else 0, dtype=torch.int64),
            "subject_id": str(row["subject_id"]),
            "foot_side": str(row["foot_side"]),
            "gender": str(row["gender"]),
            "group": str(row["group"]),
            "split": str(row["split"]),
        }
        return sample


# ---------------------------------------------------------------------------------------
# Copied EXACTLY from 06_dataset_class.ipynb, Section 1 ("Setup" + "Merge & Agreement
# Check"), wrapped in a function so it can be called with a `processed_dir` argument
# instead of the notebook's hardcoded PROCESSED_DIR global.
# ---------------------------------------------------------------------------------------
def _build_combined_manifest(processed_dir: Path) -> pd.DataFrame:
    SPLIT_MANIFEST_PATH = processed_dir / "split_manifest.csv"
    GRAPH_MANIFEST_PATH = processed_dir / "graph_manifest.csv"

    split_manifest_df = pd.read_csv(SPLIT_MANIFEST_PATH)
    graph_manifest_df = pd.read_csv(GRAPH_MANIFEST_PATH)

    if split_manifest_df.shape[0] != 334:
        raise ValueError(
            f"build_dataloaders: expected 334 rows in split_manifest.csv, found {split_manifest_df.shape[0]}"
        )
    if graph_manifest_df.shape[0] != 334:
        raise ValueError(
            f"build_dataloaders: expected 334 rows in graph_manifest.csv, found {graph_manifest_df.shape[0]}"
        )

    SHARED_COLS = ["group", "gender", "split"]

    # Merge on the join key only, then compare the shared columns from each side explicitly
    # (pandas would otherwise silently suffix them as _x/_y and hide any drift).
    merged = split_manifest_df.merge(
        graph_manifest_df,
        on=["subject_id", "foot_side"],
        suffixes=("_split", "_graph"),
        how="inner",
    )

    mismatch_mask = pd.Series(False, index=merged.index)
    for col in SHARED_COLS:
        col_mismatch = merged[f"{col}_split"] != merged[f"{col}_graph"]
        mismatch_mask |= col_mismatch

    n_total_mismatches = int(mismatch_mask.sum())
    if n_total_mismatches > 0:
        raise ValueError(
            f"build_dataloaders: {n_total_mismatches} rows have group/gender/split "
            f"disagreement between split_manifest.csv and graph_manifest.csv"
        )

    # Build the single combined_manifest_df: keep one copy of each shared column (drop the
    # _graph duplicates), plus every other column each source file uniquely contributes.
    combined_manifest_df = merged.drop(columns=[f"{c}_graph" for c in SHARED_COLS])
    combined_manifest_df = combined_manifest_df.rename(columns={f"{c}_split": c for c in SHARED_COLS})

    if combined_manifest_df.shape[0] != 334:
        raise ValueError(
            f"build_dataloaders: combined_manifest_df has {combined_manifest_df.shape[0]} rows, expected 334"
        )

    return combined_manifest_df


def load_normalization_stats(processed_dir: Path) -> dict:
    """Reads data/processed/normalization_stats.json -- produced once by
    06_dataset_class.ipynb Section 2 and never recomputed here. This file is the single
    source of truth for normalization in every phase from here on.
    """
    norm_stats_path = Path(processed_dir) / "normalization_stats.json"
    with open(norm_stats_path, "r") as f:
        return json.load(f)


def build_dataloaders(processed_dir: Path, normalization_stats: dict, batch_size: int = 16) -> dict:
    """Wraps the manifest-merging (Section 1), dataset-building (Section 4), and
    dataloader-building (Section 6) steps of 06_dataset_class.ipynb into one call.

    Internally does the same split-filtering that 06_dataset_class.ipynb did manually:
    filters the merged manifest by split == 'train' / 'val' / 'test' and builds one
    PlantarThermographyDataset per split, all sharing the SAME `normalization_stats`
    dict (fit once on train, passed in unchanged -- never refit per split).

    Returns a dict with keys 'train', 'val', 'test' (each a DataLoader) and
    'train_dataset', 'val_dataset', 'test_dataset' (the underlying Dataset objects).
    """
    processed_dir = Path(processed_dir)
    combined_manifest_df = _build_combined_manifest(processed_dir)

    train_df = combined_manifest_df.loc[combined_manifest_df["split"] == "train"].reset_index(drop=True)
    val_manifest_df = combined_manifest_df.loc[combined_manifest_df["split"] == "val"].reset_index(drop=True)
    test_manifest_df = combined_manifest_df.loc[combined_manifest_df["split"] == "test"].reset_index(drop=True)

    train_dataset = PlantarThermographyDataset(train_df, normalization_stats)
    val_dataset = PlantarThermographyDataset(val_manifest_df, normalization_stats)
    test_dataset = PlantarThermographyDataset(test_manifest_df, normalization_stats)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }
