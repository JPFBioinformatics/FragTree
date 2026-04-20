"""
Computes per-cluster stats for a specific cluster from a saved
cluster_assignments.json.

Loads the molecule fingerprints from the database, runs UMAP to get a 2D
embedding, then uses the saved labels to compute SSE, silhouette,
noise_neighbor_frac, and DBCV for the target cluster.

Persistence requires a live HDBSCAN run so is computed separately using the
same min_cluster_size / min_samples — it is an approximation since the UMAP
embedding may differ slightly from the original run due to library version
changes.

Edit CONFIG to set paths and the cluster you want metrics for.
"""

import sys
import json
import sqlite3
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import h5py
import numpy as np
import umap
import hdbscan as hdbscan_lib
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import silhouette_samples

RDLogger.DisableLog("rdApp.*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# region CONFIG

CLUSTER_JSON  = Path(__file__).resolve().parent.parent / "databases/reports/first_clustering/cluster_assignments.json"
DB_PATH       = Path(__file__).resolve().parent.parent / "databases/reference/tms/metadata.db"
MOL_H5_PATH   = Path(__file__).resolve().parent.parent / "databases/reference/tms/structures.h5"

TARGET_CLUSTER = 1   # which cluster to report metrics for

# HDBSCAN params — must match original run for persistence estimate
FINAL_CS = 75
FINAL_MS = 53

# fingerprint settings — must match cluster_analysis.py
FP_RADIUS = 2
FP_NBITS  = 2048

# UMAP settings — must match cluster_analysis.py
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1
UMAP_METRIC      = "cosine"

# endregion


def load_fingerprints():
    log.info("Loading mol objects and computing fingerprints...")
    conn   = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT casNO, h5ID FROM molecule ORDER BY molID")
    rows = cursor.fetchall()
    conn.close()

    cas_list = []
    fps      = []

    with h5py.File(str(MOL_H5_PATH), "r") as h5f:
        for cas, h5_id in rows:
            if h5_id not in h5f:
                continue
            try:
                binary = bytes(np.array(h5f[h5_id]))
                mol    = Chem.Mol(binary)  # type: ignore[arg-type]
                fp     = rdMolDescriptors.GetMorganFingerprint(mol, FP_RADIUS)
                arr    = np.zeros(FP_NBITS, dtype=np.float32)
                for idx, count in fp.GetNonzeroElements().items():
                    arr[idx % FP_NBITS] += count
                cas_list.append(cas)
                fps.append(arr)
            except Exception:
                pass

    log.info(f"Loaded {len(cas_list):,} fingerprints")
    return cas_list, np.array(fps, dtype=np.float32)


def run_umap(fps):
    log.info("Running UMAP...")
    reducer = umap.UMAP(
        n_components = 2,
        n_neighbors  = UMAP_N_NEIGHBORS,
        min_dist     = UMAP_MIN_DIST,
        metric       = UMAP_METRIC,
        random_state = 42,
    )
    embed = np.array(reducer.fit_transform(fps))
    log.info("UMAP complete")
    return embed


def main():
    # 1. Load old cluster labels
    with open(CLUSTER_JSON) as f:
        assignments = json.load(f)
    log.info(f"Loaded {len(assignments):,} assignments from {CLUSTER_JSON}")

    # 2. Load fingerprints
    cas_list, fps = load_fingerprints()

    # 3. Align labels to fingerprint order
    labels = np.array([assignments.get(cas, -1) for cas in cas_list])

    n_clusters = int(labels.max()) + 1
    log.info(f"Clusters: {n_clusters}  Noise: {(labels == -1).sum():,}")

    # 4. UMAP embedding
    embed = run_umap(fps)

    # 5. Silhouette scores (non-noise only)
    non_noise = labels != -1
    if n_clusters >= 2 and non_noise.sum() > n_clusters:
        sil_samples_arr = np.full(len(labels), float("nan"))
        sil_samples_arr[non_noise] = silhouette_samples(
            embed[non_noise], labels[non_noise]
        )
    else:
        sil_samples_arr = np.full(len(labels), float("nan"))

    # 6. Run HDBSCAN for persistence estimate
    log.info(f"Running HDBSCAN (cs={FINAL_CS}, ms={FINAL_MS}) for persistence...")
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size = FINAL_CS,
        min_samples      = FINAL_MS,
        gen_min_span_tree = True,
    )
    clusterer.fit(embed)
    persistence = clusterer.cluster_persistence_

    # 7. DBCV per cluster
    try:
        from hdbscan.validity import validity_index
        result = validity_index(
            embed.astype(np.float64), labels, per_cluster_scores=True
        )
        dbcv_global, dbcv_per = result if isinstance(result, tuple) else (result, np.full(n_clusters, float("nan")))
        log.info(f"Global DBCV: {dbcv_global:.4f}")
    except Exception as e:
        log.warning(f"DBCV failed: {e}")
        dbcv_per    = np.full(n_clusters, float("nan"))
        dbcv_global = float("nan")

    # 8. Compute and print stats for TARGET_CLUSTER
    cl = TARGET_CLUSTER
    mask     = labels == cl
    pts      = embed[mask]
    centroid = pts.mean(axis=0)
    sse      = float(np.sum((pts - centroid) ** 2))
    cl_sil   = float(np.nanmean(sil_samples_arr[mask]))
    outlier_scores = clusterer.outlier_scores_[mask]
    noise_nbr_frac = float((outlier_scores > 0.5).sum()) / max(int(mask.sum()), 1)
    pers     = float(persistence[cl]) if cl < len(persistence) else float("nan")
    dbcv_cl  = float(dbcv_per[cl])    if cl < len(dbcv_per)    else float("nan")

    print(f"\n{'='*50}")
    print(f"Cluster {cl} metrics  (n={int(mask.sum()):,})")
    print(f"{'='*50}")
    print(f"  SSE                : {sse:.4f}")
    print(f"  Silhouette         : {cl_sil:.4f}")
    print(f"  Noise neighbor frac: {noise_nbr_frac:.4f}")
    print(f"  Persistence        : {pers:.4f}  (approx — from fresh HDBSCAN run)")
    print(f"  DBCV               : {dbcv_cl:.4f}  (approx — from fresh UMAP)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
