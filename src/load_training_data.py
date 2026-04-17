"""
Loads training data for a single cluster from:
  - databases/reports/cluster_assignments.json  (casNO -> cluster label)
  - databases/reference/tms/metadata.db         (casNO -> h5IDs)
  - databases/reference/tms/spectra.h5          (spectra arrays)
  - databases/reference/tms/structures.h5       (mol binaries)

Returns a list of TrainingEntry namedtuples (casNO, mol, spectrum), one per casNO:
    cas_no    : str
    mol       : RDKit Mol object
    spectrum  : np.ndarray shape (MZ_GRID_SIZE,) — binned, normalized, averaged
                across all replicates for this casNO

Usage
-----
    from load_training_data import load_cluster
    items = load_cluster(cluster_id=0, cfg=cfg)
    cas_no, mol, spectrum = items[0]
"""

# region Imports

import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Optional

import h5py
import numpy as np
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# endregion

# Fixed m/z grid — all spectra are binned to this range
MZ_MIN       = 100
MZ_MAX       = 800
MZ_GRID_SIZE = MZ_MAX - MZ_MIN + 1   # 701 bins

# Peaks below this fraction of the base peak are dropped before binning
MIN_RELATIVE_INTENSITY = 0.05

# Spectra with fewer than this many peaks after filtering are discarded
MIN_PEAKS = 5

class TrainingEntry(NamedTuple):
    cas_no:   str
    mol:      object          # RDKit Mol
    spectrum: np.ndarray      # shape (MZ_GRID_SIZE,) float32, base-peak normalised

def load_cluster(
    cluster_id:       int,
    cluster_json:     str | Path,
    db_path:          str | Path,
    verbose:          bool = True,
) -> list[TrainingEntry]:
    """
    Loads all molecules and spectra for a single cluster.

    Parameters
    ----------
    cluster_id    integer cluster label to load (e.g. 0, 5)
    cluster_json  path to cluster_assignments.json
    db_path       path to metadata.db (h5 paths are stored inside)
    verbose       if True, log progress to stdout

    Returns
    -------
    list of TrainingEntry, one per casNO that has both a mol structure
    and at least one valid spectrum
    """
    log = logging.getLogger(__name__)
    if verbose:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s  %(levelname)s  %(message)s")

    # 1. Load cluster assignments, filter to target cluster
 
    with open(cluster_json, "r") as f:
        assignments = json.load(f)

    target_cas = [cas for cas, label in assignments.items()
                  if label == cluster_id]

    log.info(f"Cluster {cluster_id}: {len(target_cas):,} casNOs")

    if not target_cas:
        raise ValueError(f"No casNOs found for cluster_id={cluster_id}")

    # 2. Query SQLite: mol rows first, then use sp_mol_map to get matched spectra

    conn   = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    placeholders = ",".join("?" * len(target_cas))

    # molecule table: pick lowest-variant mol per casNO
    cursor.execute(
        f"""
        SELECT casNO, molID, h5ID, h5file FROM molecule
        WHERE casNO IN ({placeholders})
        ORDER BY casNO, variant ASC
        """,
        target_cas,
    )
    mol_rows = cursor.fetchall()   # [(casNO, molID, h5ID, h5file), ...]

    # keep first (lowest variant) mol per casNO
    mol_by_cas: dict[str, tuple[int, str, str]] = {}   # casNO -> (molID, h5ID, h5file)
    for cas, mol_id, h5id, h5file in mol_rows:
        if cas not in mol_by_cas:
            mol_by_cas[cas] = (mol_id, h5id, h5file)

    # use sp_mol_map to get only spectra matched to those exact mol structures
    mol_ids       = [v[0] for v in mol_by_cas.values()]
    mol_id_to_cas = {v[0]: cas for cas, v in mol_by_cas.items()}

    if mol_ids:
        mol_id_placeholders = ",".join("?" * len(mol_ids))
        cursor.execute(
            f"""
            SELECT sm.molID, s.h5ID, s.h5file
            FROM sp_mol_map sm
            JOIN spectra s ON s.spID = sm.spID
            WHERE sm.molID IN ({mol_id_placeholders})
            """,
            mol_ids,
        )
        spectra_rows = cursor.fetchall()   # [(molID, h5ID, h5file), ...]
    else:
        spectra_rows = []

    conn.close()

    # group spectra by casNO (via molID lookup)
    spectra_by_cas: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for mol_id, h5id, h5file in spectra_rows:
        cas = mol_id_to_cas.get(mol_id)
        if cas is not None:
            spectra_by_cas[cas].append((h5id, h5file))

    log.info(f"  Spectra found for {len(spectra_by_cas):,} casNOs")
    log.info(f"  Mol structures found for {len(mol_by_cas):,} casNOs")

    # 3. Load and average spectra from spectra.h5

    # group by h5file so we only open each file once
    spectra_needed: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for cas, entries in spectra_by_cas.items():
        for h5id, h5file in entries:
            spectra_needed[h5file].append((cas, h5id))

    raw_spectra: dict[str, list[np.ndarray]] = defaultdict(list)
    # raw_spectra[casNO] = list of binned spectrum vectors

    for h5file, cas_h5id_pairs in spectra_needed.items():
        try:
            with h5py.File(h5file, "r") as hf:
                for cas, h5id in cas_h5id_pairs:
                    if h5id not in hf:
                        continue
                    arr = np.array(hf[h5id])           # shape (N, 2)
                    vec = _bin_spectrum(arr)
                    if vec is not None:
                        raw_spectra[cas].append(vec)
        except Exception as e:
            log.warning(f"Could not open spectra file {h5file}: {e}")

    # average replicates per casNO
    avg_spectra: dict[str, np.ndarray] = {}
    for cas, vecs in raw_spectra.items():
        avg_spectra[cas] = np.mean(vecs, axis=0).astype(np.float32)

    n_raw = sum(len(v) for v in raw_spectra.values())
    log.info(
        f"  Averaged spectra ready for {len(avg_spectra):,} casNOs "
        f"({n_raw:,} individual spectra survived filtering)"
    )

    # 4. Load mol objects from structures.h5

    mols_needed: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for cas, (mol_id, h5id, h5file) in mol_by_cas.items():
        mols_needed[h5file].append((cas, h5id))

    loaded_mols: dict[str, Chem.Mol] = {}

    for h5file, cas_h5id_pairs in mols_needed.items():
        try:
            with h5py.File(h5file, "r") as hf:
                for cas, h5id in cas_h5id_pairs:
                    if h5id not in hf:
                        continue
                    binary = bytes(np.array(hf[h5id]))
                    mol    = Chem.Mol(binary)  # type: ignore[arg-type]
                    if mol.GetNumAtoms() > 0:
                        loaded_mols[cas] = mol
        except Exception as e:
            log.warning(f"Could not open mol file {h5file}: {e}")

    log.info(f"  Mol objects loaded for {len(loaded_mols):,} casNOs")

    # 5. Assemble final list — only casNOs with both mol and spectrum

    items: list[TrainingEntry] = []
    n_no_mol  = 0
    n_no_spec = 0

    for cas in target_cas:
        mol = loaded_mols.get(cas)
        spec = avg_spectra.get(cas)

        if mol is None:
            n_no_mol += 1
            continue
        if spec is None:
            n_no_spec += 1
            continue

        items.append(TrainingEntry(cas_no=cas, mol=mol, spectrum=spec))

    log.info(
        f"  Final dataset: {len(items):,} entries  "
        f"(skipped {n_no_mol} no-mol, {n_no_spec} no-spectrum)"
    )

    return items

def _bin_spectrum(arr: np.ndarray) -> Optional[np.ndarray]:
    """
    Converts a raw (N, 2) spectrum array [m/z, intensity] to a fixed-length
    binned vector over [MZ_MIN, MZ_MAX], normalised to base peak = 1.0.

    Peaks below MIN_RELATIVE_INTENSITY * base_peak are dropped before binning.
    Returns None if fewer than MIN_PEAKS peaks survive filtering, or if the
    spectrum is empty/all-zero.
    """
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
        return None

    mzs         = arr[:, 0].astype(float)
    intensities = arr[:, 1].astype(float)

    max_intensity = intensities.max()
    if max_intensity == 0:
        return None

    # drop peaks below 5% of base peak
    keep        = intensities >= MIN_RELATIVE_INTENSITY * max_intensity
    mzs         = mzs[keep]
    intensities = intensities[keep]

    if len(mzs) < MIN_PEAKS:
        return None

    vec = np.zeros(MZ_GRID_SIZE, dtype=np.float32)
    for mz, intensity in zip(mzs, intensities):
        idx = int(round(mz)) - MZ_MIN
        if 0 <= idx < MZ_GRID_SIZE:
            vec[idx] += float(intensity)

    max_val = vec.max()
    if max_val == 0:
        return None

    vec /= max_val   # base-peak normalisation: highest peak = 1.0
    return vec
