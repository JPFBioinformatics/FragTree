"""
Evaluation script for a trained FragTreeMPNN.

Loads a saved .pt checkpoint and the corresponding dataset cache, runs the
model on the held-out test set, and produces:

  1. Cosine similarity distribution (histogram)
  2. Top-k peak recall table (k = 1, 3, 10)
  3. Mirror plots for the N_MIRROR best and worst predicted spectra
  4. Console summary (mean / median / std cosine similarity)

Edit the CONFIG section to match the paths used during training.
"""

# region Imports

import sys
import pickle
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from mpnn import FragTreeMPNN

# endregion

# region CONFIG

CLUSTER_ID   = 1
MAX_DEPTH    = 3
MODEL_PATH   = Path(__file__).resolve().parent.parent / "databases/reports" / f"mpnn_cluster{CLUSTER_ID}.pt"
CACHE_PATH   = Path(__file__).resolve().parent.parent / "databases/reports" / f"cluster_{CLUSTER_ID}_depth{MAX_DEPTH}_dataset.pkl"
OUT_DIR      = Path(__file__).resolve().parent.parent / "databases/reports"

TEST_FRAC    = 0.10   # must match train_mpnn.py
N_MIRROR     = 5      # number of best + worst spectra to plot
TOPK_VALUES  = [1, 3, 10]

MZ_MIN       = 100    # must match load_training_data.py

# endregion

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def cosine_sim(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(F.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0)).item())


def compute_topk_retrieval(results: list[dict], dataset: list, k: int) -> float:
    """
    Top-k retrieval accuracy.

    For each test molecule, the predicted spectrum is compared against every
    observed spectrum in the full dataset using cosine similarity.  A hit is
    scored if the correct molecule's observed spectrum ranks in the top-k.
    Returns the fraction of test molecules that were correctly retrieved.

    The library is the full dataset (train + test) — this mirrors real-world
    use where you'd search a predicted spectrum against an entire reference
    library.
    """
    # build library matrix: (N_dataset, MZ_GRID_SIZE)
    library = np.stack([s["spectrum"].numpy() for s in dataset])  # (N, D)
    lib_norms = np.linalg.norm(library, axis=1, keepdims=True)
    lib_norms[lib_norms == 0] = 1.0
    library_normed = library / lib_norms  # (N, D)

    hits = 0
    for r in results:
        pred      = r["pred"]
        true_idx  = r["idx"]
        pred_norm = np.linalg.norm(pred)
        if pred_norm == 0:
            continue
        pred_normed = pred / pred_norm

        sims    = library_normed @ pred_normed          # (N,)
        top_k   = np.argpartition(sims, -k)[-k:]        # indices of top-k
        if true_idx in top_k:
            hits += 1

    return hits / len(results) if results else 0.0


def get_test_indices(dataset, checkpoint: dict) -> list[int]:
    """
    Returns test indices either from the checkpoint (if saved there) or by
    reproducing the identical stratified split used during training.
    """
    if "test_idx" in checkpoint:
        log.info("Using test indices saved in checkpoint.")
        return checkpoint["test_idx"]

    log.info("test_idx not in checkpoint — reproducing split with random_state=42.")
    n_atoms_arr = np.array([s["n_atoms"] for s in dataset])
    quartiles   = np.percentile(n_atoms_arr, [25, 50, 75])
    strata      = np.digitize(n_atoms_arr, quartiles)
    all_idx     = np.arange(len(dataset))
    _, test_idx = train_test_split(
        all_idx, test_size=TEST_FRAC, stratify=strata, random_state=42
    )
    return test_idx.tolist()


def evaluate(model, dataset, test_idx) -> list[dict]:
    """Runs the model on every test sample and returns per-sample results."""
    model.eval()
    results = []
    with torch.no_grad():
        for i in test_idx:
            s    = dataset[i]
            pred = model(s["x"], s["edge_index"], s["bond_targets"])
            pred_np   = pred.numpy()
            target_np = s["spectrum"].numpy()
            sim  = cosine_sim(pred, s["spectrum"])
            results.append({
                "idx":    i,
                "sim":    sim,
                "pred":   pred_np,
                "target": target_np,
            })
    return results


def plot_similarity_histogram(sims: list[float], out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sims, bins=40, edgecolor="black", color="#0D7377")
    ax.axvline(float(np.mean(sims)),   color="red",    linestyle="--",
               label=f"Mean {np.mean(sims):.3f}")
    ax.axvline(float(np.median(sims)), color="orange", linestyle="--",
               label=f"Median {np.median(sims):.3f}")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Number of spectra")
    ax.set_title(f"Cluster {CLUSTER_ID} — test set cosine similarity (n={len(sims)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Histogram saved to {out_path}")


def plot_topk_table(results: list[dict], dataset: list, out_path: Path):
    """Saves top-k retrieval accuracy as a PNG table."""
    rows = [["k", "Retrieval Accuracy"]]
    for k in TOPK_VALUES:
        acc = compute_topk_retrieval(results, dataset, k)
        rows.append([f"Top-{k}", f"{acc:.4f}"])

    fig, ax = plt.subplots(figsize=(4, 1 + 0.4 * len(rows)))
    ax.axis("off")
    tbl = ax.table(
        cellText  = rows[1:],
        colLabels = rows[0],
        cellLoc   = "center",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.5, 1.8)
    ax.set_title(
        f"Cluster {CLUSTER_ID} — Top-k Retrieval Accuracy\n"
        f"(library = full dataset, n_test = {len(results)})",
        fontsize=11, pad=12
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Top-k table saved to {out_path}")


def plot_intensity_scatter(results: list[dict], out_path: Path):
    """
    Scatter plot of predicted vs observed intensity for every non-zero m/z bin
    across all test spectra.  Each point is one (molecule, m/z bin) pair where
    the observed intensity is > 0.
    """
    obs_vals  = []
    pred_vals = []
    for r in results:
        # normalise each spectrum to sum=1 so predicted and observed are comparable
        obs_sum  = r["target"].sum()
        pred_sum = r["pred"].sum()
        if obs_sum == 0 or pred_sum == 0:
            continue
        obs_norm  = r["target"] / obs_sum
        pred_norm = r["pred"]   / pred_sum
        mask = obs_norm > 0
        obs_vals.append(obs_norm[mask])
        pred_vals.append(pred_norm[mask])

    obs_all  = np.concatenate(obs_vals)
    pred_all = np.concatenate(pred_vals)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(obs_all, pred_all, s=8, alpha=0.5, color="#0D7377", linewidths=0)
    lim = max(float(obs_all.max()), float(pred_all.max())) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="y = x (perfect prediction)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Observed intensity (sum-normalised)")
    ax.set_ylabel("Predicted intensity (sum-normalised)")
    ax.set_title(
        f"Cluster {CLUSTER_ID} — predicted vs observed peak intensities\n"
        f"(n={len(obs_all):,} non-zero peaks across {len(results)} test spectra)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Intensity scatter saved to {out_path}")


def plot_sim_vs_molweight(results: list[dict], dataset: list, out_path: Path):
    """
    Scatter plot of cosine similarity vs molecular weight (approximated as
    atom count from the dataset cache) for each test molecule.
    """
    n_atoms = np.array([dataset[r["idx"]]["n_atoms"] for r in results])
    sims    = np.array([r["sim"] for r in results])

    # adaptive binning: use quartiles for small sets, deciles for large
    n_bins  = 4 if len(results) < 50 else 10
    label   = "Quartile mean" if n_bins == 4 else "Decile mean"
    edges   = np.percentile(n_atoms, np.linspace(0, 100, n_bins + 1))
    # deduplicate edges so digitize doesn't create empty bins
    edges   = np.unique(edges)
    bin_ids = np.digitize(n_atoms, edges[1:-1])
    bin_means   = []
    bin_centers = []
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if mask.any():
            bin_means.append(float(sims[mask].mean()))
            bin_centers.append(float((edges[b] + edges[b+1]) / 2))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(n_atoms, sims, s=20, alpha=0.6, color="#0D7377", linewidths=0,
               label="Individual molecules")
    if len(bin_centers) > 1:
        ax.plot(bin_centers, bin_means, "r-o", markersize=5, linewidth=1.5,
                label=label)
    elif len(bin_centers) == 1:
        ax.plot(bin_centers, bin_means, "ro", markersize=5, label=label)
    ax.set_xlabel("Atom count (proxy for molecular weight)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Cluster {CLUSTER_ID} — prediction quality vs molecular size")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Sim vs mol weight saved to {out_path}")


def plot_mirror(pred: np.ndarray, target: np.ndarray, sim: float,
                title: str, out_path: Path):
    """Mirror plot: observed spectrum above axis, predicted below."""
    mzs = np.arange(len(pred)) + MZ_MIN

    fig, ax = plt.subplots(figsize=(10, 4))

    for mz, intensity in zip(mzs, target):
        if intensity > 0:
            ax.vlines(mz,  0,  intensity, colors="#2196F3", linewidth=0.8)

    for mz, intensity in zip(mzs, pred):
        if intensity > 0:
            ax.vlines(mz, 0, -intensity, colors="#F44336", linewidth=0.8)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative intensity")
    ax.set_title(f"{title}  |  cosine sim = {sim:.4f}")

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="#2196F3", lw=2, label="Observed (above)"),
        Line2D([0], [0], color="#F44336", lw=2, label="Predicted (below)"),
    ])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    # 1. Load checkpoint
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    log.info(
        f"Loaded model — cluster {checkpoint.get('cluster_id')}  "
        f"fold_val_sims={[f'{v:.4f}' for v in checkpoint.get('fold_val_sims', [])]}  "
        f"saved test_sim={checkpoint.get('test_sim', 'N/A')}"
    )

    # 2. Load dataset cache
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Dataset cache not found: {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        dataset, *_ = pickle.load(f)
    log.info(f"Loaded {len(dataset):,} entries from cache")

    # 3. Reconstruct model
    model = FragTreeMPNN(
        hidden_dim = checkpoint["hidden_dim"],
        n_layers   = checkpoint["n_layers"],
        dropout    = checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])

    # 4. Get test indices
    test_idx = get_test_indices(dataset, checkpoint)
    log.info(f"Test set size: {len(test_idx)}")

    # 5. Evaluate
    log.info("Running evaluation...")
    results = evaluate(model, dataset, test_idx)
    sims    = [r["sim"] for r in results]

    log.info(
        f"Cosine similarity — "
        f"mean={np.mean(sims):.4f}  "
        f"median={np.median(sims):.4f}  "
        f"std={np.std(sims):.4f}  "
        f"min={np.min(sims):.4f}  "
        f"max={np.max(sims):.4f}"
    )

    plot_topk_table(
        results, dataset,
        OUT_DIR / f"cluster{CLUSTER_ID}_topk_retrieval.png"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 6. Histogram
    plot_similarity_histogram(
        sims,
        OUT_DIR / f"cluster{CLUSTER_ID}_test_similarity_hist.png"
    )

    # 7. Intensity scatter
    plot_intensity_scatter(
        results,
        OUT_DIR / f"cluster{CLUSTER_ID}_intensity_scatter.png"
    )

    # 8. Cosine similarity vs molecular weight
    plot_sim_vs_molweight(
        results, dataset,
        OUT_DIR / f"cluster{CLUSTER_ID}_sim_vs_molweight.png"
    )

    # 9. Mirror plots — best and worst N_MIRROR spectra
    results_sorted = sorted(results, key=lambda r: r["sim"], reverse=True)

    for rank, r in enumerate(results_sorted[:N_MIRROR], start=1):
        plot_mirror(
            r["pred"], r["target"], r["sim"],
            title=f"Best #{rank} (dataset idx {r['idx']})",
            out_path=OUT_DIR / f"cluster{CLUSTER_ID}_best{rank}.png",
        )

    for rank, r in enumerate(results_sorted[-N_MIRROR:][::-1], start=1):
        plot_mirror(
            r["pred"], r["target"], r["sim"],
            title=f"Worst #{rank} (dataset idx {r['idx']})",
            out_path=OUT_DIR / f"cluster{CLUSTER_ID}_worst{rank}.png",
        )

    log.info(f"All plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
