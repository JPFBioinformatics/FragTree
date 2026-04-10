"""

Clusters TMS-derivatized metabolites by structural similarity using Morgan
fingerprints, UMAP dimensionality reduction, and HDBSCAN density-based
clustering.

Workflow
--------
1. run_coarse_grid()
        2D sweep: min_cluster_size 5-100 step 10, min_samples 5-100.
        Produces heatmaps (DBCV, silhouette, noise fraction, mean persistence,
        BSS/TSS) and a top-10%% summary table.  Use to identify a region.

2. run_fine_grid(cs_min, cs_max, cs_step, ms_min, ms_max)
        Same report format, finer resolution centered on the coarse region.

3. run_final(min_cluster_size, min_samples)
        Full diagnostic report: global metrics, per-cluster stats table
        (size, SSE, silhouette, noise-neighbor fraction, persistence),
        distribution histograms, UMAP scatter, and cluster_assignments.json.

Usage
-----
    ca.run_coarse_grid()
    ca.run_fine_grid(cs_min=65, cs_max=85, cs_step=2, ms_min=5, ms_max=30)
    ca.run_final(min_cluster_size=75, min_samples=11)

"""

# region Imports

import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage, PageBreak)
from reportlab.lib import colors

# suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")  # type: ignore

# lazy imports for heavy ML libraries
try:
    import umap
    import hdbscan as hdbscan_lib
except ImportError as e:
    raise ImportError(
        f"Missing dependency: {e}.  "
        "Install with: conda install -c conda-forge umap-learn hdbscan"
    )

# endregion

# region Helpers

def _pdf_style():
    """Returns a dict of named ParagraphStyles for the PDF reports."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"],
                                fontSize=16, spaceAfter=6),
        "h2":    ParagraphStyle("h2",    parent=base["Heading2"],
                                fontSize=13, spaceBefore=14, spaceAfter=4),
        "body":  ParagraphStyle("body",  parent=base["Normal"],
                                fontSize=10, leading=14),
        "small": ParagraphStyle("small", parent=base["Normal"],
                                fontSize=8,  leading=11),
    }


def _table_style(header_color: str = "#0A1628") -> TableStyle:
    """Returns a consistent TableStyle for metric tables in reports."""
    return TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor(header_color)),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EEF4F4"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#CCDDDD")),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
    ])

# endregion

class ClusterAnalysis:
    """
    Clusters TMS metabolites by structural similarity using Morgan fingerprints,
    UMAP dimensionality reduction, and HDBSCAN.

    Parameters loaded from config.yaml
    ------------------------------------
    db_path         path to the combined SQLite database
    mol_h5_path     path to mol_objects.h5
    reports_dir     directory to write PDF reports and JSON output
    """

    # Morgan fingerprint settings
    FP_RADIUS   = 2
    FP_NBITS    = 2048

    # UMAP settings — fixed
    UMAP_N_COMPONENTS  = 2
    UMAP_N_NEIGHBORS   = 15
    UMAP_MIN_DIST      = 0.1
    UMAP_METRIC        = "jaccard"

    def __init__(self, cfg):
        """
        cfg     ConfigLoader instance
        """
        self.db_path     = str(cfg.get_path("dbs","db_path",     must_exist=True))
        self.mol_h5_path = str(cfg.get_path("dbs","mol_h5_path", must_exist=True))
        self.reports_dir = cfg.get_path("outputs","reports_dir")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # set up logging
        log_file = self.reports_dir / "cluster_analysis.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)s  %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ]
        )
        self.log = logging.getLogger(__name__)

        # these are populated by _load_data() and _run_umap() and cached
        # so repeated grid runs don't redo expensive steps
        self._cas_list:    Optional[list]       = None  # casNO per molecule
        self._fingerprints: Optional[np.ndarray] = None  # (n, FP_NBITS) bool
        self._umap_embed:  Optional[np.ndarray] = None  # (n, 2) float

    # region Public API

    def run_coarse_grid(
        self,
        cs_min:  int = 5,
        cs_max:  int = 100,
        cs_step: int = 10,
        ms_min:  int = 5,
        ms_max:  int = 100,
    ):
        """
        Coarse 2D parameter sweep.  Thin wrapper around run_grid with wide
        defaults.  Writes coarse_grid_report.pdf to reports_dir.

        Parameters
        ----------
        cs_min      min_cluster_size lower bound (default 5)
        cs_max      min_cluster_size upper bound (default 100)
        cs_step     step size (default 10)
        ms_min      min_samples lower bound (default 5)
        ms_max      min_samples upper bound (default 100)
        """
        self.run_grid(
            cs_min  = cs_min,
            cs_max  = cs_max,
            cs_step = cs_step,
            ms_min  = ms_min,
            ms_max  = ms_max,
            pdf_name = "coarse_grid_report.pdf",
            title    = "Coarse Grid Search Report",
        )

    def run_fine_grid(
        self,
        cs_min:  int,
        cs_max:  int,
        cs_step: int = 2,
        ms_min:  int = 1,
        ms_max:  Optional[int] = None,
    ):
        """
        Fine 2D parameter sweep centered on the region identified from the
        coarse report.  Thin wrapper around run_grid.
        Writes fine_grid_report.pdf to reports_dir.

        Parameters
        ----------
        cs_min      min_cluster_size lower bound
        cs_max      min_cluster_size upper bound
        cs_step     step size (default 2)
        ms_min      min_samples lower bound (default 1)
        ms_max      min_samples upper bound (default = cs_max)
        """
        self.run_grid(
            cs_min  = cs_min,
            cs_max  = cs_max,
            cs_step = cs_step,
            ms_min  = ms_min,
            ms_max  = ms_max if ms_max is not None else cs_max,
            pdf_name = "fine_grid_report.pdf",
            title    = "Fine Grid Search Report",
        )

    def run_grid(
        self,
        cs_min:   int,
        cs_max:   int,
        cs_step:  int,
        ms_min:   int,
        ms_max:   int,
        pdf_name: str,
        title:    str,
    ):
        """
        Core 2D grid sweep used by both run_coarse_grid and run_fine_grid.

        Sweeps (min_cluster_size, min_samples), skipping pairs where
        min_samples > min_cluster_size.  Computes DBCV, silhouette, noise
        fraction, mean persistence, and BSS/TSS.  Produces heatmaps for
        each metric and a top-10%% summary table.

        Parameters
        ----------
        cs_min      min_cluster_size lower bound
        cs_max      min_cluster_size upper bound
        cs_step     step size for min_cluster_size
        ms_min      min_samples lower bound
        ms_max      min_samples upper bound
        pdf_name    filename for the output PDF
        title       title shown in the PDF header
        """
        self.log.info(f"=== Grid search: {title} ===")
        self._ensure_data_loaded()

        cs_values = list(range(cs_min, cs_max + 1, cs_step))
        ms_values = list(range(ms_min, ms_max + 1, cs_step))
        results   = []

        total = len(cs_values) * len(ms_values)
        done  = 0
        for cs in cs_values:
            for ms in ms_values:
                if ms > cs:
                    continue
                done += 1
                if done % 10 == 0:
                    self.log.info(f"  Progress: {done} / {total}")
                metrics = self._run_hdbscan(cs, ms)
                results.append({"cs": cs, "ms": ms, **metrics})
                self.log.info(
                    f"  cs={cs} ms={ms}  "
                    f"DBCV={metrics['dbcv']:.4f}  "
                    f"sil={metrics['silhouette']:.4f}  "
                    f"noise={metrics['noise_frac']:.3f}  "
                    f"persist={metrics['mean_persistence']:.4f}  "
                    f"bss_tss={metrics['bss_tss']:.4f}  "
                    f"k={metrics['n_clusters']}"
                )

        pdf_path = self.reports_dir / pdf_name
        self._write_grid_pdf(
            results   = results,
            cs_values = cs_values,
            ms_values = [m for m in ms_values if m <= cs_max],
            output_path = pdf_path,
            title       = title,
        )
        self.log.info(f"Grid report saved to {pdf_path}")

    def run_final(self, min_cluster_size: int, min_samples: int):
        """
        Runs HDBSCAN with the chosen parameters and produces the full
        diagnostic report and cluster assignment JSON.

        Report includes: all six metrics, UMAP scatter plot colored by
        cluster, per-cluster persistence bar chart.

        JSON maps each casNO to its cluster label (-1 = noise).

        Parameters
        ----------
        min_cluster_size    chosen from fine grid report
        min_samples         chosen from fine grid report
        """
        self.log.info(
            f"=== Final clustering: "
            f"min_cluster_size={min_cluster_size}, "
            f"min_samples={min_samples} ==="
        )
        self._ensure_data_loaded()

        metrics  = self._run_hdbscan(min_cluster_size, min_samples,
                                     return_clusterer=True)
        labels   = metrics["labels"]
        clusterer = metrics["clusterer"]

        # ARI and AMI — use casNO as ground truth groupings
        cas_labels = self._encode_cas_labels()
        ari = float(adjusted_rand_score(cas_labels, labels))
        ami = float(adjusted_mutual_info_score(cas_labels, labels))
        self.log.info(
            f"  DBCV={metrics['dbcv']:.4f}  "
            f"sil={metrics['silhouette']:.4f}  "
            f"noise={metrics['noise_frac']:.3f}  "
            f"n_clusters={metrics['n_clusters']}  "
            f"ARI={ari:.4f}  AMI={ami:.4f}"
        )

        # write JSON
        json_path = self.reports_dir / "cluster_assignments.json"
        self._write_cluster_json(labels, json_path)

        # write full PDF
        pdf_path = self.reports_dir / "final_cluster_report.pdf"
        self._write_final_pdf(
            metrics           = metrics,
            ari               = ari,
            ami               = ami,
            clusterer         = clusterer,
            labels            = labels,
            min_cluster_size  = min_cluster_size,
            min_samples       = min_samples,
            output_path       = pdf_path,
        )
        self.log.info(f"Final report saved to {pdf_path}")
        self.log.info(f"Cluster assignments saved to {json_path}")

    # endregion

    # region Data loading and preprocessing

    def _ensure_data_loaded(self):
        """
        Loads fingerprints and cas list on first call, then runs UMAP.
        Cached so subsequent grid calls don't repeat the work.
        """
        if self._fingerprints is None:
            self._load_data()
        if self._umap_embed is None:
            self._run_umap()

    def _load_data(self):
        """
        Loads all mol objects from HDF5, computes Morgan fingerprints,
        and stores the casNO list aligned with the fingerprint rows.
        """
        self.log.info("Loading mol objects and computing fingerprints...")

        conn   = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT casNO, h5ID FROM molecule ORDER BY molID")
        rows = cursor.fetchall()
        conn.close()

        cas_list = []
        fps      = []
        n_failed = 0

        with h5py.File(self.mol_h5_path, "r") as h5f:
            for cas, h5_id in rows:
                if h5_id not in h5f:
                    n_failed += 1
                    continue
                try:
                    binary = bytes(np.array(h5f[h5_id]))
                    mol    = Chem.Mol(binary)  # type: ignore
                    fp     = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                        mol, self.FP_RADIUS, nBits=self.FP_NBITS
                    )
                    cas_list.append(cas)
                    fps.append(np.array(fp, dtype=np.uint8))
                except Exception as e:
                    self.log.warning(f"Fingerprint failed for {h5_id} (CAS {cas}): {e}")
                    n_failed += 1

        self._cas_list    = cas_list
        self._fingerprints = np.array(fps, dtype=np.uint8)
        self.log.info(
            f"  Loaded {len(cas_list):,} molecules  "
            f"({n_failed} skipped)"
        )

    def _run_umap(self):
        """
        Reduces fingerprint matrix to 2D using UMAP.
        Uses Jaccard metric which is appropriate for binary fingerprints.
        """
        self.log.info(
            f"Running UMAP  "
            f"(n_neighbors={self.UMAP_N_NEIGHBORS}, "
            f"min_dist={self.UMAP_MIN_DIST}, "
            f"metric={self.UMAP_METRIC})..."
        )
        reducer = umap.UMAP(
            n_components = self.UMAP_N_COMPONENTS,
            n_neighbors  = self.UMAP_N_NEIGHBORS,
            min_dist     = self.UMAP_MIN_DIST,
            metric       = self.UMAP_METRIC,
            random_state = 42,
        )
        self._umap_embed = np.array(reducer.fit_transform(self._fingerprints))
        self.log.info("  UMAP complete")

    # endregion 

    # region HDBSCAN

    def _run_hdbscan(
        self,
        min_cluster_size: int,
        min_samples:      int,
        return_clusterer: bool = False,
    ) -> dict:
        """
        Runs HDBSCAN on the UMAP embedding and returns a metrics dict.

        Global metrics always returned
        ------------------------------
        dbcv             Density-Based Clustering Validation (primary)
        silhouette       Mean silhouette score across non-noise points
        noise_frac       Fraction of points labelled noise (-1)
        mean_persistence Mean cluster persistence from condensed tree
        bss_tss          BSS / TSS — fraction of variance explained by clusters
        n_clusters       Number of clusters found (noise not counted)

        Additional keys when return_clusterer=True
        -------------------------------------------
        labels                    np.ndarray of cluster labels per point
        clusterer                 fitted HDBSCAN object
        persistence_per_cluster   np.ndarray of per-cluster persistence scores
        per_cluster_stats         list of dicts, one per cluster, containing:
                                      cluster_id, size, sse, silhouette,
                                      noise_neighbor_frac, persistence
        """
        assert self._umap_embed is not None, "UMAP embedding not computed"
        embed = self._umap_embed

        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size    = min_cluster_size,
            min_samples         = min_samples,
            gen_min_span_tree   = True,
        )
        clusterer.fit(embed)
        labels = clusterer.labels_

        n_clusters = int(labels.max()) + 1
        noise_mask = labels == -1
        noise_frac = float(noise_mask.sum()) / len(labels)

        # DBCV
        dbcv = float(clusterer.relative_validity_)

        # global silhouette
        non_noise = ~noise_mask
        if n_clusters >= 2 and non_noise.sum() > n_clusters:
            sil = float(silhouette_score(embed[non_noise], labels[non_noise]))
        else:
            sil = float("nan")

        # persistence
        persistence      = clusterer.cluster_persistence_
        mean_persistence = float(np.mean(persistence)) if len(persistence) > 0 else 0.0

        # BSS / TSS
        global_mean = embed.mean(axis=0)
        tss = float(np.sum((embed - global_mean) ** 2))
        bss = 0.0
        for cl in range(n_clusters):
            mask = labels == cl
            if mask.sum() == 0:
                continue
            centroid = embed[mask].mean(axis=0)
            bss += float(mask.sum()) * float(np.sum((centroid - global_mean) ** 2))
        bss_tss = bss / tss if tss > 0 else 0.0

        result = {
            "dbcv":             dbcv,
            "silhouette":       sil,
            "noise_frac":       noise_frac,
            "mean_persistence": mean_persistence,
            "bss_tss":          bss_tss,
            "n_clusters":       n_clusters,
        }

        if return_clusterer:
            from sklearn.metrics import silhouette_samples
            # per-point silhouette scores (non-noise only, mapped back by index)
            if n_clusters >= 2 and non_noise.sum() > n_clusters:
                sil_samples = np.full(len(labels), float("nan"))
                sil_samples[non_noise] = silhouette_samples(
                    embed[non_noise], labels[non_noise]
                )
            else:
                sil_samples = np.full(len(labels), float("nan"))

            per_cluster_stats = []
            for cl in range(n_clusters):
                mask     = labels == cl
                pts      = embed[mask]
                centroid = pts.mean(axis=0)
                sse      = float(np.sum((pts - centroid) ** 2))

                # mean silhouette for this cluster
                cl_sil_vals = sil_samples[mask]
                cl_sil = float(np.nanmean(cl_sil_vals)) if mask.sum() > 0 else float("nan")

                # noise-neighbor fraction: fraction of border points adjacent
                # to noise.  Approximated using HDBSCAN outlier scores —
                # points with outlier_score > 0.5 are considered border/noise-adjacent
                outlier_scores = clusterer.outlier_scores_[mask]
                noise_nbr_frac = float((outlier_scores > 0.5).sum()) / max(mask.sum(), 1)

                per_cluster_stats.append({
                    "cluster_id":        cl,
                    "size":              int(mask.sum()),
                    "sse":               sse,
                    "silhouette":        cl_sil,
                    "noise_neighbor_frac": noise_nbr_frac,
                    "persistence":       float(persistence[cl]) if cl < len(persistence) else 0.0,
                })

            result["labels"]               = labels
            result["clusterer"]            = clusterer
            result["persistence_per_cluster"] = persistence
            result["per_cluster_stats"]    = per_cluster_stats

        return result

    def _encode_cas_labels(self) -> np.ndarray:
        """
        Converts the casNO list to integer labels for ARI/AMI computation.
        Each unique casNO gets a unique integer.
        """
        assert self._cas_list is not None, "Data not loaded"
        unique = {cas: i for i, cas in enumerate(sorted(set(self._cas_list)))}
        return np.array([unique[c] for c in self._cas_list])
    
    # endregion

    # region JSON output

    def _write_cluster_json(self, labels: np.ndarray, output_path: Path):
        """
        Writes a JSON file mapping each casNO to its cluster label.
        Compounds assigned to noise are labeled -1.
        """
        assert self._cas_list is not None, "Data not loaded"
        assignments = {
            cas: int(label)
            for cas, label in zip(self._cas_list, labels)
        }
        with open(output_path, "w") as f:
            json.dump(assignments, f, indent=2)

    # endregion

    # region PDF reports

    def _write_grid_pdf(
        self,
        results:     list,
        cs_values:   list,
        ms_values:   list,
        output_path: Path,
        title:       str,
    ):
        """
        Writes a grid search report PDF with heatmaps (one per metric) and
        a top-10%% summary table of the best (cs, ms) parameter pairs.

        A pair is included in the summary table if it falls in the top 10%%
        of ANY metric — giving a comprehensive view of promising regions.
        """
        tmp_dir = self.reports_dir / "_tmp_plots"
        tmp_dir.mkdir(exist_ok=True)

        plot_paths = self._make_heatmaps(results, cs_values, ms_values, tmp_dir)

        doc    = SimpleDocTemplate(str(output_path), pagesize=letter,
                                   leftMargin=inch, rightMargin=inch,
                                   topMargin=inch, bottomMargin=inch)
        styles = _pdf_style()
        story  = []

        assert self._cas_list is not None, "Data not loaded"
        story.append(Paragraph(f"FragTree — {title}", styles["title"]))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
            f"n_molecules: {len(self._cas_list):,}",
            styles["body"]
        ))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Parameter Sweep Heatmaps", styles["h2"]))
        story.append(Paragraph(
            "Each cell shows the metric value for a (min_cluster_size, min_samples) pair.  "
            "Maximize DBCV, silhouette, mean persistence, and BSS/TSS.  "
            "Minimize noise fraction (keep below 0.30).  "
            "Cells with values in the top 10% are annotated with their score.",
            styles["body"]
        ))
        story.append(Spacer(1, 0.15 * inch))

        for path in plot_paths:
            story.append(RLImage(str(path), width=6.5 * inch, height=4.8 * inch))
            story.append(Spacer(1, 0.15 * inch))

        # top-10%% summary table
        story.append(PageBreak())
        story.append(Paragraph("Top-10% Summary Table", styles["h2"]))
        story.append(Paragraph(
            "A parameter pair is included if it falls in the top 10% of ANY "
            "metric (or bottom 10% for noise fraction, where lower is better).  "
            "Sorted by DBCV descending.",
            styles["body"]
        ))
        story.append(Spacer(1, 0.1 * inch))

        # compute per-metric top-10%% thresholds
        def top10_set(key: str, lower_is_better: bool = False) -> set:
            vals = [r[key] for r in results if not np.isnan(float(r[key] if r[key] is not None else float("nan")))]
            if not vals:
                return set()
            thresh = np.percentile(vals, 90 if not lower_is_better else 10)
            if lower_is_better:
                return {(r["cs"], r["ms"]) for r in results if r[key] <= thresh}
            return {(r["cs"], r["ms"]) for r in results if r[key] >= thresh}

        top_pairs = (
            top10_set("dbcv") |
            top10_set("silhouette") |
            top10_set("mean_persistence") |
            top10_set("bss_tss") |
            top10_set("noise_frac", lower_is_better=True)
        )

        top_results = sorted(
            [r for r in results if (r["cs"], r["ms"]) in top_pairs],
            key=lambda x: x["dbcv"], reverse=True
        )

        header = ["cs", "ms", "DBCV", "Silhouette", "Noise frac",
                  "Mean persistence", "BSS/TSS", "N clusters"]
        tbl_rows = [header] + [
            [r["cs"], r["ms"],
             f"{r['dbcv']:.4f}",
             f"{r['silhouette']:.4f}" if not np.isnan(r["silhouette"]) else "N/A",
             f"{r['noise_frac']:.3f}",
             f"{r['mean_persistence']:.4f}",
             f"{r['bss_tss']:.4f}",
             r["n_clusters"]]
            for r in top_results
        ]
        col_w = [0.65, 0.65, 0.95, 0.95, 0.9, 1.3, 0.9, 0.9]
        tbl = Table(tbl_rows, colWidths=[w * inch for w in col_w])
        tbl.setStyle(_table_style())
        story.append(tbl)

        doc.build(story)

        for p in plot_paths:
            p.unlink(missing_ok=True)

    def _write_final_pdf(
        self,
        metrics:          dict,
        ari:              float,
        ami:              float,
        clusterer,
        labels:           np.ndarray,
        min_cluster_size: int,
        min_samples:      int,
        output_path:      Path,
    ):
        """
        Writes the full diagnostic PDF for the chosen final parameters.

        Sections
        --------
        1. Global metrics table (DBCV, silhouette, noise fraction, BSS/TSS,
           mean persistence, n_clusters, ARI, AMI)
        2. UMAP scatter coloured by cluster
        3. Per-cluster distribution histograms (size, SSE, silhouette,
           noise-neighbour fraction, persistence)
        4. Top-10%% cluster table — any cluster in the top 10%% of ANY
           per-cluster metric is included, sorted by silhouette descending
        """
        tmp_dir = self.reports_dir / "_tmp_plots"
        tmp_dir.mkdir(exist_ok=True)

        scatter_path  = self._make_umap_scatter(labels, tmp_dir)
        hist_paths    = self._make_per_cluster_histograms(
            metrics["per_cluster_stats"], tmp_dir
        )

        doc    = SimpleDocTemplate(str(output_path), pagesize=letter,
                                   leftMargin=inch, rightMargin=inch,
                                   topMargin=inch, bottomMargin=inch)
        styles = _pdf_style()
        story  = []

        story.append(Paragraph("FragTree — Final Clustering Report", styles["title"]))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
            f"min_cluster_size={min_cluster_size}  |  min_samples={min_samples}",
            styles["body"]
        ))
        story.append(Spacer(1, 0.2 * inch))

        # ── global metrics table
        story.append(Paragraph("Global Clustering Metrics", styles["h2"]))
        story.append(Paragraph(
            "DBCV is the primary metric.  ARI and AMI use CAS number "
            "groupings as soft ground truth.  BSS/TSS measures the fraction "
            "of total variance explained by the cluster structure (higher = better).",
            styles["body"]
        ))
        story.append(Spacer(1, 0.1 * inch))

        sil = metrics["silhouette"]
        global_rows = [
            ["Metric", "Value", "Target", "Notes"],
            ["DBCV (primary)",
             f"{metrics['dbcv']:.4f}", "Maximize",
             "Density-based clustering validation"],
            ["Silhouette score",
             f"{sil:.4f}" if not np.isnan(sil) else "N/A", "Maximize",
             "Mean intra vs inter-cluster cohesion"],
            ["Noise fraction",
             f"{metrics['noise_frac']:.3f}", "< 0.30",
             f"{int(metrics['noise_frac'] * len(labels)):,} points labelled noise"],
            ["BSS / TSS",
             f"{metrics['bss_tss']:.4f}", "Maximize",
             "Fraction of variance explained by cluster structure"],
            ["Mean persistence",
             f"{metrics['mean_persistence']:.4f}", "Maximize",
             "Stability across density thresholds"],
            ["N clusters",
             str(metrics["n_clusters"]), "-",
             "Noise cluster (-1) not counted"],
            ["ARI",
             f"{ari:.4f}", "1.0 ideal",
             "Agreement with CAS number groupings"],
            ["AMI",
             f"{ami:.4f}", "1.0 ideal",
             "ARI normalised for cluster size distribution"],
        ]
        col_w = [1.5, 0.9, 1.0, 3.4]
        tbl   = Table(global_rows, colWidths=[w * inch for w in col_w])
        tbl.setStyle(_table_style())
        story.append(tbl)
        story.append(Spacer(1, 0.2 * inch))

        # ── UMAP scatter
        story.append(Paragraph("UMAP Embedding — Coloured by Cluster", styles["h2"]))
        story.append(Paragraph(
            "Each point is one compound.  Grey = noise (label -1).",
            styles["body"]
        ))
        story.append(Spacer(1, 0.1 * inch))
        story.append(RLImage(str(scatter_path), width=6.5 * inch, height=5.0 * inch))

        # ── per-cluster distribution histograms
        story.append(PageBreak())
        story.append(Paragraph("Per-Cluster Metric Distributions", styles["h2"]))
        story.append(Paragraph(
            "Each histogram shows how a metric is distributed across all clusters.  "
            "Ideal training clusters sit in the right tail of size, silhouette, "
            "and persistence, and the left tail of SSE and noise-neighbour fraction.",
            styles["body"]
        ))
        story.append(Spacer(1, 0.1 * inch))
        for hp in hist_paths:
            story.append(RLImage(str(hp), width=6.5 * inch, height=3.5 * inch))
            story.append(Spacer(1, 0.1 * inch))

        # ── top-10%% cluster table
        story.append(PageBreak())
        story.append(Paragraph("Top-10% Cluster Summary Table", styles["h2"]))
        story.append(Paragraph(
            "A cluster is included if it falls in the top 10% of ANY per-cluster "
            "metric (or bottom 10% for SSE and noise-neighbour fraction where lower "
            "is better).  Sorted by silhouette score descending.  Use this table to "
            "identify the best candidate cluster for model training.",
            styles["body"]
        ))
        story.append(Spacer(1, 0.1 * inch))

        pcs = metrics["per_cluster_stats"]

        def top10_cluster_set(key: str, lower_is_better: bool = False) -> set:
            vals = [s[key] for s in pcs if not np.isnan(float(s[key]))]
            if not vals:
                return set()
            thresh = np.percentile(vals, 10 if lower_is_better else 90)
            if lower_is_better:
                return {s["cluster_id"] for s in pcs if s[key] <= thresh}
            return {s["cluster_id"] for s in pcs if s[key] >= thresh}

        top_ids = (
            top10_cluster_set("size") |
            top10_cluster_set("silhouette") |
            top10_cluster_set("persistence") |
            top10_cluster_set("sse",               lower_is_better=True) |
            top10_cluster_set("noise_neighbor_frac", lower_is_better=True)
        )

        top_clusters = sorted(
            [s for s in pcs if s["cluster_id"] in top_ids],
            key=lambda x: x["silhouette"] if not np.isnan(x["silhouette"]) else -999,
            reverse=True,
        )

        cl_header = ["Cluster", "Size", "SSE", "Silhouette",
                     "Noise nbr frac", "Persistence"]
        cl_rows = [cl_header] + [
            [s["cluster_id"],
             f"{s['size']:,}",
             f"{s['sse']:.2f}",
             f"{s['silhouette']:.4f}" if not np.isnan(s["silhouette"]) else "N/A",
             f"{s['noise_neighbor_frac']:.3f}",
             f"{s['persistence']:.4f}"]
            for s in top_clusters
        ]
        cl_col_w = [0.9, 0.9, 1.1, 1.1, 1.3, 1.1]
        cl_tbl   = Table(cl_rows, colWidths=[w * inch for w in cl_col_w])
        cl_tbl.setStyle(_table_style())
        story.append(cl_tbl)

        doc.build(story)

        scatter_path.unlink(missing_ok=True)
        for hp in hist_paths:
            hp.unlink(missing_ok=True)

    # endregion

    # region Plot helpers

    def _make_heatmaps(
        self,
        results:   list,
        cs_values: list,
        ms_values: list,
        tmp_dir:   Path,
    ) -> list:
        """
        Creates five heatmaps (one per metric) for a 2D parameter sweep.
        Cells whose values fall in the top 10%% (or bottom 10%% for noise)
        are annotated with their numeric score.
        Returns a list of Path objects to the saved PNG files.
        """
        metrics_cfg = [
            ("dbcv",             "DBCV (primary)",   "viridis",   False),
            ("silhouette",       "Silhouette score", "viridis",   False),
            ("noise_frac",       "Noise fraction",   "viridis_r", True),   # lower better
            ("mean_persistence", "Mean persistence", "viridis",   False),
            ("bss_tss",          "BSS / TSS",        "viridis",   False),
        ]

        result_map = {(r["cs"], r["ms"]): r for r in results}
        valid_ms   = [m for m in ms_values if m <= max(cs_values)]
        paths      = []

        for key, label, cmap, lower_is_better in metrics_cfg:
            matrix = np.full((len(valid_ms), len(cs_values)), np.nan)
            for ci, cs in enumerate(cs_values):
                for mi, ms in enumerate(valid_ms):
                    if (cs, ms) in result_map:
                        v = result_map[(cs, ms)][key]
                        matrix[mi, ci] = v if not np.isnan(float(v)) else np.nan

            # top-10%% threshold for annotation
            flat = matrix[~np.isnan(matrix)]
            if len(flat) > 0:
                thresh = np.percentile(flat, 10 if lower_is_better else 90)
            else:
                thresh = np.nan

            fig, ax = plt.subplots(figsize=(max(8, len(cs_values) * 0.4 + 2),
                                            max(5, len(valid_ms) * 0.35 + 1.5)))
            im = ax.imshow(matrix, aspect="auto", cmap=cmap, origin="lower")
            plt.colorbar(im, ax=ax, label=label)

            ax.set_xticks(range(len(cs_values)))
            ax.set_yticks(range(len(valid_ms)))
            ax.set_xticklabels(cs_values, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(valid_ms, fontsize=8)
            ax.set_xlabel("min_cluster_size", fontsize=11)
            ax.set_ylabel("min_samples",      fontsize=11)
            ax.set_title(
                f"{label}  (annotated = top 10%%)",
                fontsize=12
            )

            # annotate top-10%% cells
            if not np.isnan(thresh):
                for ci in range(len(cs_values)):
                    for mi in range(len(valid_ms)):
                        v = matrix[mi, ci]
                        if np.isnan(v):
                            continue
                        in_top = (v <= thresh) if lower_is_better else (v >= thresh)
                        if in_top:
                            ax.text(ci, mi, f"{v:.3f}", ha="center", va="center",
                                    fontsize=6, color="white", fontweight="bold")

            if key == "noise_frac":
                ax.set_title(
                    f"{label}  (annotated = bottom 10%%, dashed = 0.30 target)",
                    fontsize=11
                )

            fig.tight_layout()
            out = tmp_dir / f"grid_{key}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            paths.append(out)

        return paths

    def _make_umap_scatter(self, labels: np.ndarray, tmp_dir: Path) -> Path:
        """
        Creates a UMAP scatter plot colored by cluster label.
        Noise points (-1) are drawn in light grey beneath the clusters.
        """
        assert self._umap_embed is not None, "UMAP embedding not computed"
        embed      = self._umap_embed
        n_clusters = int(labels.max()) + 1
        cmap       = cm.get_cmap("tab20", max(n_clusters, 1))

        fig, ax = plt.subplots(figsize=(9, 7))

        # noise points first (underneath)
        noise_mask = labels == -1
        ax.scatter(
            embed[noise_mask, 0], embed[noise_mask, 1],
            c="lightgrey", s=2, alpha=0.4, linewidths=0, label="Noise"
        )

        # cluster points
        for cl in range(n_clusters):
            mask = labels == cl
            ax.scatter(
                embed[mask, 0], embed[mask, 1],
                c=[cmap(cl)], s=4, alpha=0.6, linewidths=0,
                label=f"Cluster {cl} (n={mask.sum():,})"
            )

        ax.set_xlabel("UMAP 1", fontsize=11)
        ax.set_ylabel("UMAP 2", fontsize=11)
        ax.set_title(
            f"UMAP embedding — {n_clusters} clusters  "
            f"({noise_mask.sum():,} noise points)",
            fontsize=12
        )

        # legend only if there are few enough clusters to be readable
        if n_clusters <= 20:
            ax.legend(markerscale=3, fontsize=7, loc="best",
                      ncol=max(1, n_clusters // 10))

        fig.tight_layout()
        out = tmp_dir / "umap_scatter.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    def _make_per_cluster_histograms(
        self,
        per_cluster_stats: list,
        tmp_dir:           Path,
    ) -> list:
        """
        Creates one histogram per per-cluster metric showing the distribution
        of values across all clusters.  A vertical line marks the top-10%%
        threshold on each plot.

        Metrics plotted: size, SSE, silhouette score, noise-neighbour fraction,
        persistence.

        Returns a list of Path objects to the saved PNG files.
        """
        if not per_cluster_stats:
            return []

        metrics_cfg = [
            ("size",               "Cluster size (n compounds)",
             "#0D7377",  False, None),
            ("sse",                "SSE (sum of squared distances to centroid)",
             "#F4A261",  True,  None),
            ("silhouette",         "Silhouette score",
             "#14A5AB",  False, None),
            ("noise_neighbor_frac","Noise-neighbour fraction",
             "#C44E52",  True,  0.5),
            ("persistence",        "Cluster persistence",
             "#8172B2",  False, 0.05),
        ]

        paths = []
        for key, xlabel, color, lower_is_better, threshold_line in metrics_cfg:
            vals = [s[key] for s in per_cluster_stats
                    if not np.isnan(float(s[key]))]
            if not vals:
                continue

            fig, ax = plt.subplots(figsize=(8, 3.8))
            ax.hist(vals, bins=min(30, max(5, len(vals) // 3)),
                    color=color, edgecolor="white", linewidth=0.4, alpha=0.85)

            # top/bottom 10%% threshold line
            thresh = np.percentile(vals, 10 if lower_is_better else 90)
            ax.axvline(thresh, color="#1E1E1E", linewidth=1.5, linestyle="--",
                       label=f"{'Bottom' if lower_is_better else 'Top'} 10%% "
                             f"threshold: {thresh:.3f}")

            # optional domain reference line (e.g. noise 0.5, persistence 0.05)
            if threshold_line is not None:
                ax.axvline(threshold_line, color="#888888", linewidth=1.0,
                           linestyle=":", label=f"Reference: {threshold_line}")

            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel("Number of clusters", fontsize=11)
            ax.set_title(
                f"{xlabel} — distribution across {len(vals)} clusters",
                fontsize=12
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25)

            fig.tight_layout()
            out = tmp_dir / f"hist_{key}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            paths.append(out)

        return paths

    def _make_persistence_chart(
        self,
        persistence: np.ndarray,
        tmp_dir:     Path,
    ) -> Path:
        """
        Creates a horizontal bar chart of per-cluster persistence scores,
        sorted descending.  A dashed line marks the 0.05 marginal threshold.
        """
        n      = len(persistence)
        labels = [f"Cluster {i}" for i in range(n)]
        order  = np.argsort(persistence)[::-1]
        sorted_p = persistence[order]
        sorted_l = [labels[i] for i in order]

        fig_h = max(4, n * 0.25)
        fig, ax = plt.subplots(figsize=(8, fig_h))
        bars = ax.barh(sorted_l, sorted_p, color="#0D7377", edgecolor="white",
                       linewidth=0.4)
        ax.axvline(0.05, color="#C44E52", linewidth=1.2, linestyle="--",
                   label="Marginal threshold (0.05)")
        ax.set_xlabel("Persistence", fontsize=11)
        ax.set_title("Per-cluster persistence (higher = more stable)", fontsize=12)
        ax.legend(fontsize=9)
        ax.tick_params(axis="y", labelsize=7)

        fig.tight_layout()
        out = tmp_dir / "persistence_chart.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    # endregion