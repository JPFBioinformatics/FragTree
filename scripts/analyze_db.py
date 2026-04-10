"""
Examines a given database and prduces plots

Plots produced:
    1. CAS number replicate histogram       — distribution of spectra per unique compound
    2. Top 20 most replicated compounds     — which compounds dominate the dataset
    3. Molecular weight distribution        — MW range and shape across all spectra
    4. Number of peaks distribution         — spectral complexity across all spectra
    5. Retention index distribution         — GC separation space coverage
    6. Top 20 most common formulas          — dominant compound classes

"""

# region Imports

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config_loader import ConfigLoader
from explore_db import ExploreDB
import matplotlib.pyplot as plt

# endregion

def main():

    # load paths from config
    config_path  = Path(__file__).resolve().parent.parent / "config.yaml"
    cfg          = ConfigLoader(config_path)
    db_path      = cfg.get_path("dbs","db_path",      must_exist=True)
    h5_path      = cfg.get_path("dbs","h5_path",      must_exist=True)
    reports_dir  = cfg.get_path("outputs","reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)

    db = ExploreDB(db_path, h5_path)

    # ------------------------------------------------------------------
    # Figure 1 — metadata plots (2x3)
    # ------------------------------------------------------------------

    fig1, axes1 = plt.subplots(2, 3, figsize=(22, 13))
    fig1.suptitle("TMS Reference Database — Metadata", fontsize=15)
    fig1.subplots_adjust(hspace=0.45, wspace=0.35)

    db.plot_replicate_histogram(axes1[0, 0])
    db.plot_top_replicated(axes1[0, 1])
    db.plot_mw_distribution(axes1[0, 2])
    db.plot_num_peaks_distribution(axes1[1, 0])
    db.plot_retention_index_distribution(axes1[1, 1])
    db.plot_formula_frequency(axes1[1, 2])

    pdf1 = reports_dir / "tms_metadata.pdf"
    fig1.savefig(pdf1, bbox_inches="tight")
    print(f"Saved {pdf1}")

    # ------------------------------------------------------------------
    # Figure 2 — spectral plots (3x3)
    # ------------------------------------------------------------------

    print("\nLoading all spectra for spectral plots...")
    all_spectra = db.load_all_spectra()
    print(f"  Loaded {len(all_spectra)} spectra\n")

    occurrence, summed_abundance = db.compute_fragment_stats(all_spectra)

    fig2, axes2 = plt.subplots(3, 3, figsize=(22, 19))
    fig2.suptitle("TMS Reference Database — Spectral Analysis", fontsize=15)
    fig2.subplots_adjust(hspace=0.45, wspace=0.35)

    db.plot_fragment_occurrence(axes2[0, 0], occurrence, len(all_spectra))
    db.plot_summed_abundance(axes2[0, 1], summed_abundance)
    db.plot_mz_density_hexbin(axes2[0, 2], all_spectra)
    db.plot_base_peak_distribution(axes2[1, 0], all_spectra)
    db.plot_intensity_distribution(axes2[1, 1], all_spectra)
    db.plot_fragment_count_vs_mw(axes2[1, 2])
    db.plot_similarity_distribution(axes2[2, 0])

    # hide unused axes in the bottom row
    axes2[2, 1].set_visible(False)
    axes2[2, 2].set_visible(False)

    pdf2 = reports_dir / "tms_spectral_analysis.pdf"
    fig2.savefig(pdf2, bbox_inches="tight")
    print(f"Saved {pdf2}")

    # ------------------------------------------------------------------
    # Figure 3 — Structure Coverage Plots (1x2)
    # ------------------------------------------------------------------

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle("TMS Reference Database — Mol Structure Coverage", fontsize=15)
    fig3.subplots_adjust(wspace=0.35)

    db.plot_mol_coverage(axes3[0])
    db.plot_mol_coverage_by_replicates(axes3[1])

    pdf3 = reports_dir / "tms_mol_coverage.pdf"
    fig3.savefig(pdf3, bbox_inches="tight")
    print(f"Saved {pdf3}")

if __name__ == '__main__':
    main()
