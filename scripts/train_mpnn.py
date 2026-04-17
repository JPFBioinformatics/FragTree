"""
Training script for FragTreeMPNN.

Pipeline
--------
1. Load a cluster with load_cluster.
2. Build a FragmentationTree + featurize each molecule.
3. Hold out TEST_FRAC (10%) as a permanent test set (never seen during training).
4. Run K_FOLDS-fold CV on the remaining 90% (train / val per fold).
5. Save the best model across all folds and report cosine similarity.

Edit the CONFIG section to set paths, cluster ID, and hyperparameters.
"""

# region Imports

import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split

from load_training_data import load_cluster
from frag_tree import FragmentationTree
from mpnn import FragTreeMPNN, featurize_mol, build_bond_targets

# endregion

# region CONFIG — edit here

CLUSTER_ID   = 1
CLUSTER_JSON = Path(__file__).resolve().parent.parent / "databases/reports/cluster_assignments.json"
DB_PATH      = Path(__file__).resolve().parent.parent / "databases/reference/tms/metadata.db"

MAX_DEPTH    = 3
HIDDEN_DIM   = 128
N_LAYERS     = 3
DROPOUT      = 0.1
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 25
K_FOLDS      = 3
TEST_FRAC    = 0.10

SAVE_DIR     = Path(__file__).resolve().parent.parent / "databases/reports"

# endregion

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# region Dataset construction

def build_dataset(items) -> tuple[list, list[int], list[int], list[float]]:
    """
    Builds fragmentation trees and featurizes all TrainingEntry objects.

    Each element of the returned list is a dict with keys:
        x, edge_index, edge_attr  — PyG tensors from featurize_mol
        bond_targets              — from build_bond_targets
        spectrum                  — observed spectrum tensor
        n_atoms                   — used for mass-proxy stratification

    Entries with no valid bond targets or an all-zero spectrum are skipped.

    Returns (dataset, fragment_counts, bond_break_counts, mol_weights) where
    the count lists contain one entry per successfully built tree (including
    skipped entries) for histogram analysis.
    """
    dataset           = []
    n_skip            = 0
    fragment_counts   = []
    bond_break_counts = []
    mol_weights       = []

    for i, entry in enumerate(items):
        log.info(f"  [{i+1}/{len(items)}] Building tree for {entry.cas_no} ...")
        try:
            tree         = FragmentationTree(entry.mol, max_depth=MAX_DEPTH)
            fragment_counts.append(tree.n_fragments)
            bond_break_counts.append(tree.n_bond_breaks)
            mol_weights.append(tree.root.mass)
            log.info(f"    -> {tree.n_fragments} fragments, {tree.n_bond_breaks} bond breaks, MW={tree.root.mass:.1f}")
            bond_targets = build_bond_targets(tree)

            if not bond_targets["atom_i"]:
                n_skip += 1
                continue

            x, edge_index, edge_attr = featurize_mol(entry.mol)
            spectrum = torch.tensor(entry.spectrum, dtype=torch.float32)

            if spectrum.sum() == 0:
                n_skip += 1
                continue

            dataset.append({
                "x":            x,
                "edge_index":   edge_index,
                "edge_attr":    edge_attr,
                "bond_targets": bond_targets,
                "spectrum":     spectrum,
                "n_atoms":      x.shape[0],
            })

        except Exception as e:
            n_skip += 1
            log.debug(f"Skipped {entry.cas_no}: {e}")

    log.info(f"Dataset: {len(dataset):,} usable entries, {n_skip:,} skipped")
    return dataset, fragment_counts, bond_break_counts, mol_weights

# endregion

# region Training helpers

def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cosine_similarity, averaged over the batch (here always batch=1)."""
    return 1.0 - F.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0))

def run_epoch(model, optimizer, dataset, indices, train: bool) -> float:
    """
    Runs one epoch over the given indices.
    If train=True, computes gradients and updates weights.
    Returns mean cosine similarity (not loss) for monitoring.
    """
    model.train(train)
    sims = []

    perm = np.random.permutation(indices) if train else indices

    for i in perm:
        sample = dataset[i]
        x, ei     = sample["x"], sample["edge_index"]
        bt, spec  = sample["bond_targets"], sample["spectrum"]

        if train:
            optimizer.zero_grad()

        pred = model(x, ei, bt)
        loss = cosine_loss(pred, spec)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        sim = 1.0 - loss.item()
        sims.append(sim)

    return float(np.mean(sims)) if sims else 0.0

def train_fold(
    model, optimizer, dataset, train_idx, val_idx, fold: int
) -> tuple[float, dict]:
    """
    Trains for EPOCHS epochs on train_idx.

    Returns
    -------
    best_val_sim   best validation cosine similarity seen across all epochs
    best_state     model state_dict at that epoch
    """
    best_val_sim = -1.0
    best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(1, EPOCHS + 1):
        train_sim = run_epoch(model, optimizer, dataset, train_idx, train=True)
        val_sim   = run_epoch(model, optimizer, dataset, val_idx,   train=False)

        if val_sim > best_val_sim:
            best_val_sim = val_sim
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            log.info(
                f"  Fold {fold}  Epoch {epoch:3d}/{EPOCHS}  "
                f"train_sim={train_sim:.4f}  val_sim={val_sim:.4f}  "
                f"best={best_val_sim:.4f}"
            )

    return best_val_sim, best_state

# endregion

# region Main

def main():
    # 1. Load cluster
    log.info(f"Loading cluster {CLUSTER_ID}...")
    items = load_cluster(
        cluster_id   = CLUSTER_ID,
        cluster_json = CLUSTER_JSON,
        db_path      = DB_PATH,
    )

    # 2. Build dataset (load from cache if available)
    cache_path = SAVE_DIR / f"cluster_{CLUSTER_ID}_depth{MAX_DEPTH}_dataset.pkl"
    if cache_path.exists():
        log.info(f"Loading cached dataset from {cache_path}...")
        with open(cache_path, "rb") as f:
            dataset, fragment_counts, bond_break_counts, mol_weights = pickle.load(f)
        log.info(f"Loaded {len(dataset):,} entries from cache")
    else:
        log.info("Building fragmentation trees and featurizing...")
        dataset, fragment_counts, bond_break_counts, mol_weights = build_dataset(items)
        log.info(f"Saving dataset to cache at {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump((dataset, fragment_counts, bond_break_counts, mol_weights), f)
        log.info("Cache saved")

        # plot tree-size histograms once when the cache is first built
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        ax1.hist(fragment_counts, bins=20, edgecolor="black")
        ax1.set_xlabel("Fragment count")
        ax1.set_ylabel("Number of molecules")
        ax1.set_title(f"Cluster {CLUSTER_ID} — fragments per tree")
        ax2.hist(bond_break_counts, bins=20, edgecolor="black")
        ax2.set_xlabel("Bond break count")
        ax2.set_ylabel("Number of molecules")
        ax2.set_title(f"Cluster {CLUSTER_ID} — bond breaks per tree")
        ax3.hist(mol_weights, bins=20, edgecolor="black")
        ax3.set_xlabel("Molecular weight (Da)")
        ax3.set_ylabel("Number of molecules")
        ax3.set_title(f"Cluster {CLUSTER_ID} — molecular weight")
        fig.tight_layout()
        hist_path = SAVE_DIR / f"cluster_{CLUSTER_ID}_tree_sizes.png"
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        log.info(f"Tree-size histograms saved to {hist_path}")

    if len(dataset) < K_FOLDS * 2:
        raise RuntimeError(
            f"Dataset too small ({len(dataset)}) for {K_FOLDS}-fold CV"
        )

    # 3. Stratify by atom-count quartile (proxy for molecular mass)
    n_atoms_arr = np.array([s["n_atoms"] for s in dataset])
    quartiles   = np.percentile(n_atoms_arr, [25, 50, 75])
    strata      = np.digitize(n_atoms_arr, quartiles)   # labels 0–3

    # 4. Hold out permanent test set
    all_idx = np.arange(len(dataset))
    train_val_idx, test_idx = train_test_split(
        all_idx,
        test_size    = TEST_FRAC,
        stratify     = strata,
        random_state = 42,
    )
    strata_tv = strata[train_val_idx]
    log.info(f"Split — train+val: {len(train_val_idx):,}  test: {len(test_idx):,}")

    # 5. K-fold CV on train+val
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_val_sims      = []
    best_overall_state = None
    best_overall_sim   = -1.0

    for fold, (tr_rel, val_rel) in enumerate(
        skf.split(train_val_idx, strata_tv), start=1
    ):
        train_idx = train_val_idx[tr_rel]
        val_idx   = train_val_idx[val_rel]

        log.info(
            f"\nFold {fold}/{K_FOLDS}  "
            f"train={len(train_idx):,}  val={len(val_idx):,}"
        )

        model     = FragTreeMPNN(
            hidden_dim = HIDDEN_DIM,
            n_layers   = N_LAYERS,
            dropout    = DROPOUT,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )

        best_val_sim, best_state = train_fold(
            model, optimizer, dataset, train_idx, val_idx, fold
        )
        fold_val_sims.append(best_val_sim)
        log.info(f"Fold {fold} best val cosine sim: {best_val_sim:.4f}")

        if best_val_sim > best_overall_sim:
            best_overall_sim   = best_val_sim
            best_overall_state = best_state

    log.info(
        f"\nK-fold results — "
        f"mean: {np.mean(fold_val_sims):.4f}  "
        f"std: {np.std(fold_val_sims):.4f}"
    )

    # 6. Evaluate best model on held-out test set
    best_model = FragTreeMPNN(
        hidden_dim = HIDDEN_DIM,
        n_layers   = N_LAYERS,
        dropout    = DROPOUT,
    )
    if best_overall_state is None:
        raise RuntimeError("No model state was saved — all folds may have failed.")
    best_model.load_state_dict(best_overall_state)
    test_sim = run_epoch(best_model, None, dataset, test_idx, train=False)
    log.info(f"Test cosine similarity: {test_sim:.4f}")

    # 7. Save
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / f"mpnn_cluster{CLUSTER_ID}.pt"
    torch.save(
        {
            "model_state":   best_overall_state,
            "hidden_dim":    HIDDEN_DIM,
            "n_layers":      N_LAYERS,
            "dropout":       DROPOUT,
            "cluster_id":    CLUSTER_ID,
            "fold_val_sims": fold_val_sims,
            "test_sim":      test_sim,
            "test_idx":      test_idx.tolist(),
            "max_depth":     MAX_DEPTH,
        },
        save_path,
    )
    log.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()

# endregion
