"""
cluster_analysis.py

Clusters TMS-derivatized metabolites by structural similarity using Morgan
fingerprints, UMAP dimensionality reduction, and HDBSCAN density-based
clustering.

Workflow
--------
1. run_coarse_grid()
        Sweeps min_cluster_size from 5 to 100 in steps of 10 with
        min_samples = min_cluster_size at each point.  Reports DBCV,
        silhouette score, noise fraction, and mean cluster persistence
        as line plots so you can identify a promising parameter region.

2. run_fine_grid(cs_min, cs_max, cs_step, ms_min, ms_max)
        2D sweep over (min_cluster_size, min_samples) centered on the
        region identified from the coarse results.  Reports the same four
        metrics as heatmaps so you can pick the best (cs, ms) pair.

3. run_final(min_cluster_size, min_samples)
        Runs once with your chosen parameters and produces the full
        diagnostic PDF (all six metrics, UMAP scatter plot, per-cluster
        persistence bar chart) and a JSON file mapping casNO to cluster
        label.

Usage
-----
    from cluster_analysis import ClusterAnalysis
    from config_loader import ConfigLoader
    from pathlib import Path

    cfg = ConfigLoader(Path("config.yaml"))
    ca  = ClusterAnalysis(cfg)

    ca.run_coarse_grid()
    # inspect coarse_grid_report.pdf, pick a region, then:
    ca.run_fine_grid(cs_min=20, cs_max=50, cs_step=2, ms_min=5, ms_max=30)
    # inspect fine_grid_report.pdf, pick best pair, then:
    ca.run_final(min_cluster_size=32, min_samples=14)
"""

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
matplotlib.use("Agg")   # non-interactive backend — safe for all platforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from rdkit import Chem  # type: ignore
from rdkit import RDLogger  # type: ignore
from rdkit.Chem import AllChem  # type: ignore
from rdkit.Chem import rdMolDescriptors  # type: ignore
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage, PageBreak)
from reportlab.lib import colors

# suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")  # type: ignore

# lazy imports for heavy ML libraries — imported once in __init__
try:
    import umap                      # type: ignore
    import hdbscan as hdbscan_lib    # type: ignore
except ImportError as e:
    raise ImportError(
        f"Missing dependency: {e}.  "
        "Install with: conda install -c conda-forge umap-learn hdbscan"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

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

    # UMAP settings — fixed, not swept
    UMAP_N_COMPONENTS  = 2
    UMAP_N_NEIGHBORS   = 15
    UMAP_MIN_DIST      = 0.1
    UMAP_METRIC        = "jaccard"   # appropriate for binary fingerprints

    def __init__(self, cfg):
        """
        cfg     ConfigLoader instance
        """
        self.db_path     = str(cfg.get_path("db_path",     must_exist=True))
        self.mol_h5_path = str(cfg.get_path("mol_h5_path", must_exist=True))
        self.reports_dir = cfg.get_path("reports_dir")
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

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run_coarse_grid(
        self,
        cs_min:  int = 5,
        cs_max:  int = 100,
        cs_step: int = 10,
    ):
        """
        Coarse parameter sweep: min_cluster_size from cs_min to cs_max in
        steps of cs_step, with min_samples = min_cluster_size at each point.

        Metrics computed: DBCV, silhouette score, noise fraction,
        mean cluster persistence.

        Writes coarse_grid_report.pdf to reports_dir.

        Parameters
        ----------
        cs_min      minimum min_cluster_size to try (default 5)
        cs_max      maximum min_cluster_size to try (default 100)
        cs_step     step size (default 10)
        """
        self.log.info("=== Coarse grid search ===")
        self._ensure_data_loaded()

        cs_values = list(range(cs_min, cs_max + 1, cs_step))
        results   = []

        for cs in cs_values:
            self.log.info(f"  min_cluster_size={cs}, min_samples={cs}")
            metrics = self._run_hdbscan(cs, cs)
            results.append({"cs": cs, "ms": cs, **metrics})
            self.log.info(
                f"    DBCV={metrics['dbcv']:.4f}  "
                f"sil={metrics['silhouette']:.4f}  "
                f"noise={metrics['noise_frac']:.3f}  "
                f"persistence={metrics['mean_persistence']:.4f}  "
                f"n_clusters={metrics['n_clusters']}"
            )

        pdf_path = self.reports_dir / "coarse_grid_report.pdf"
        self._write_grid_pdf(
            results        = results,
            sweep_label    = "min_cluster_size  (min_samples = min_cluster_size)",
            x_key          = "cs",
            output_path    = pdf_path,
            title          = "Coarse Grid Search Report",
            is_2d          = False,
        )
        self.log.info(f"Coarse grid report saved to {pdf_path}")

    def run_fine_grid(
        self,
        cs_min:  int,
        cs_max:  int,
        cs_step: int = 2,
        ms_min:  int = 1,
        ms_max:  Optional[int] = None,
    ):
        """
        Fine 2D parameter sweep over (min_cluster_size, min_samples).

        Call this after interpreting the coarse grid report to center the
        search on the promising region.  Results are shown as heatmaps.

        Writes fine_grid_report.pdf to reports_dir.

        Parameters
        ----------
        cs_min      lower bound for min_cluster_size
        cs_max      upper bound for min_cluster_size
        cs_step     step size for min_cluster_size (default 2)
        ms_min      lower bound for min_samples (default 1)
        ms_max      upper bound for min_samples (default = cs_max)
        """
        if ms_max is None:
            ms_max = cs_max

        self.log.info("=== Fine grid search ===")
        self._ensure_data_loaded()

        cs_values = list(range(cs_min, cs_max + 1, cs_step))
        ms_values = list(range(ms_min, ms_max + 1, cs_step))
        results   = []

        total = len(cs_values) * len(ms_values)
        done  = 0
        for cs in cs_values:
            for ms in ms_values:
                if ms > cs:
                    # min_samples > min_cluster_size is unusual and often
                    # produces degenerate results — skip to save time
                    continue
                done += 1
                if done % 10 == 0:
                    self.log.info(f"  Fine grid progress: {done} / {total}")
                metrics = self._run_hdbscan(cs, ms)
                results.append({"cs": cs, "ms": ms, **metrics})

        pdf_path = self.reports_dir / "fine_grid_report.pdf"
        self._write_grid_pdf(
            results     = results,
            sweep_label = "min_cluster_size vs min_samples",
            x_key       = "cs",
            output_path = pdf_path,
            title       = "Fine Grid Search Report",
            is_2d       = True,
            cs_values   = cs_values,
            ms_values   = [m for m in ms_values if m <= cs_max],
        )
        self.log.info(f"Fine grid report saved to {pdf_path}")

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

    # -----------------------------------------------------------------------
    # Data loading and preprocessing
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # HDBSCAN
    # -----------------------------------------------------------------------

    def _run_hdbscan(
        self,
        min_cluster_size: int,
        min_samples:      int,
        return_clusterer: bool = False,
    ) -> dict:
        """
        Runs HDBSCAN on the UMAP embedding and returns a metrics dict.

        Parameters
        ----------
        min_cluster_size    minimum points to form a cluster
        min_samples         controls density conservativeness
        return_clusterer    if True, also returns the fitted clusterer and
                            labels in the result dict (used for final run)

        Returns
        -------
        dict with keys: dbcv, silhouette, noise_frac, mean_persistence,
                        n_clusters, [labels, clusterer if return_clusterer]
        """
        assert self._umap_embed is not None, "UMAP embedding not computed"
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size    = min_cluster_size,
            min_samples         = min_samples,
            gen_min_span_tree   = True,
        )
        clusterer.fit(self._umap_embed)
        labels = clusterer.labels_

        n_clusters  = int(labels.max()) + 1
        noise_mask  = labels == -1
        noise_frac  = float(noise_mask.sum()) / len(labels)

        # DBCV — built into hdbscan as relative_validity_
        dbcv = float(clusterer.relative_validity_)

        # silhouette — requires at least 2 clusters and some non-noise points
        non_noise = ~noise_mask
        if n_clusters >= 2 and non_noise.sum() > n_clusters:
            sil = float(silhouette_score(
                self._umap_embed[non_noise], labels[non_noise]
            ))
        else:
            sil = float("nan")

        # mean cluster persistence from condensed tree
        persistence = clusterer.cluster_persistence_
        mean_persistence = float(np.mean(persistence)) if len(persistence) > 0 else 0.0

        result = {
            "dbcv":             dbcv,
            "silhouette":       sil,
            "noise_frac":       noise_frac,
            "mean_persistence": mean_persistence,
            "n_clusters":       n_clusters,
        }
        if return_clusterer:
            result["labels"]    = labels
            result["clusterer"] = clusterer
            result["persistence_per_cluster"] = persistence

        return result

    def _encode_cas_labels(self) -> np.ndarray:
        """
        Converts the casNO list to integer labels for ARI/AMI computation.
        Each unique casNO gets a unique integer.
        """
        assert self._cas_list is not None, "Data not loaded"
        unique = {cas: i for i, cas in enumerate(sorted(set(self._cas_list)))}
        return np.array([unique[c] for c in self._cas_list])

    # -----------------------------------------------------------------------
    # JSON output
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # PDF reports
    # -----------------------------------------------------------------------

    def _write_grid_pdf(
        self,
        results:     list,
        sweep_label: str,
        x_key:       str,
        output_path: Path,
        title:       str,
        is_2d:       bool,
        cs_values:   Optional[list] = None,
        ms_values:   Optional[list] = None,
    ):
        """
        Writes a grid search report PDF.

        For 1D sweeps (coarse): four line plots, one per metric.
        For 2D sweeps (fine):   four heatmaps, one per metric.
        """
        tmp_dir = self.reports_dir / "_tmp_plots"
        tmp_dir.mkdir(exist_ok=True)

        if is_2d:
            assert cs_values is not None and ms_values is not None
            plot_paths = self._make_heatmaps(results, cs_values, ms_values, tmp_dir)
        else:
            plot_paths = self._make_line_plots(results, x_key, sweep_label, tmp_dir)

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

        story.append(Paragraph("Parameter Sweep", styles["h2"]))
        story.append(Paragraph(
            f"Sweep: {sweep_label}.  "
            "Use these plots to identify the parameter region that maximizes "
            "DBCV and silhouette score while keeping noise fraction below 30% "
            "and cluster persistence high.",
            styles["body"]
        ))
        story.append(Spacer(1, 0.15 * inch))

        # embed plots
        for path in plot_paths:
            story.append(RLImage(str(path), width=6.5 * inch, height=4.5 * inch))
            story.append(Spacer(1, 0.2 * inch))

        # results table
        story.append(PageBreak())
        story.append(Paragraph("Full Results Table", styles["h2"]))

        if is_2d:
            header = ["cs", "ms", "DBCV", "Silhouette", "Noise frac",
                      "Mean persistence", "N clusters"]
            rows   = [header] + [
                [r["cs"], r["ms"],
                 f"{r['dbcv']:.4f}", f"{r['silhouette']:.4f}",
                 f"{r['noise_frac']:.3f}", f"{r['mean_persistence']:.4f}",
                 r["n_clusters"]]
                for r in sorted(results, key=lambda x: x["dbcv"], reverse=True)
            ]
            col_w = [0.7, 0.7, 1.1, 1.1, 1.1, 1.5, 1.1]
        else:
            header = ["min_cluster_size", "DBCV", "Silhouette",
                      "Noise frac", "Mean persistence", "N clusters"]
            rows   = [header] + [
                [r["cs"], f"{r['dbcv']:.4f}", f"{r['silhouette']:.4f}",
                 f"{r['noise_frac']:.3f}", f"{r['mean_persistence']:.4f}",
                 r["n_clusters"]]
                for r in results
            ]
            col_w = [1.5, 1.1, 1.1, 1.1, 1.5, 1.1]

        tbl = Table(rows, colWidths=[w * inch for w in col_w])
        tbl.setStyle(_table_style())
        story.append(tbl)

        doc.build(story)

        # clean up temp plots
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
        """Writes the full diagnostic PDF for the chosen final parameters."""
        tmp_dir = self.reports_dir / "_tmp_plots"
        tmp_dir.mkdir(exist_ok=True)

        scatter_path    = self._make_umap_scatter(labels, tmp_dir)
        persistence_path = self._make_persistence_chart(
            metrics["persistence_per_cluster"], tmp_dir
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

        # metrics summary table
        story.append(Paragraph("Clustering Metrics", styles["h2"]))
        story.append(Paragraph(
            "DBCV is the primary metric (density-based, maximize).  "
            "ARI and AMI use CAS number groupings as soft ground truth.",
            styles["body"]
        ))
        story.append(Spacer(1, 0.1 * inch))

        sil = metrics["silhouette"]
        metric_rows = [
            ["Metric", "Value", "Target", "Notes"],
            ["DBCV (primary)",    f"{metrics['dbcv']:.4f}",
             "Maximize", "Density-based clustering validation"],
            ["Silhouette score",  f"{sil:.4f}" if not np.isnan(sil) else "N/A",
             "Maximize", "Intra vs inter-cluster cohesion"],
            ["Noise fraction",    f"{metrics['noise_frac']:.3f}",
             "< 0.30",   f"{int(metrics['noise_frac'] * len(labels)):,} points labeled noise"],
            ["Mean persistence",  f"{metrics['mean_persistence']:.4f}",
             "Maximize", "Stability across density thresholds"],
            ["N clusters",        str(metrics["n_clusters"]),
             "-",        "Noise cluster (-1) not counted"],
            ["ARI",               f"{ari:.4f}",
             "1.0 ideal", "Agreement with CAS number groupings"],
            ["AMI",               f"{ami:.4f}",
             "1.0 ideal", "ARI normalized for cluster size distribution"],
        ]
        col_w = [1.5, 1.0, 1.0, 3.8]
        tbl   = Table(metric_rows, colWidths=[w * inch for w in col_w])
        tbl.setStyle(_table_style())
        story.append(tbl)
        story.append(Spacer(1, 0.2 * inch))

        # UMAP scatter
        story.append(Paragraph("UMAP Embedding — Colored by Cluster", styles["h2"]))
        story.append(Paragraph(
            "Each point is one compound.  Grey points are noise (label -1).  "
            "Cluster labels are assigned by HDBSCAN density estimation.",
            styles["body"]
        ))
        story.append(Spacer(1, 0.1 * inch))
        story.append(RLImage(str(scatter_path), width=6.5 * inch, height=5.0 * inch))
        story.append(Spacer(1, 0.2 * inch))

        # persistence bar chart
        story.append(PageBreak())
        story.append(Paragraph("Per-Cluster Persistence", styles["h2"]))
        story.append(Paragraph(
            "Persistence measures how stable each cluster is across density "
            "thresholds.  Higher persistence = more trustworthy cluster.  "
            "Clusters with persistence below ~0.05 are marginal.",
            styles["body"]
        ))
        story.append(Spacer(1, 0.1 * inch))
        story.append(RLImage(str(persistence_path),
                             width=6.5 * inch, height=4.5 * inch))

        doc.build(story)

        # clean up
        scatter_path.unlink(missing_ok=True)
        persistence_path.unlink(missing_ok=True)

    # -----------------------------------------------------------------------
    # Plot helpers
    # -----------------------------------------------------------------------

    def _make_line_plots(
        self,
        results:     list,
        x_key:       str,
        x_label:     str,
        tmp_dir:     Path,
    ) -> list:
        """
        Creates four line plots (one per metric) for a 1D parameter sweep.
        Returns a list of Path objects to the saved PNG files.
        """
        metrics_cfg = [
            ("dbcv",             "DBCV (primary)",     "Maximize", "#0D7377"),
            ("silhouette",       "Silhouette score",   "Maximize", "#14A5AB"),
            ("noise_frac",       "Noise fraction",     "< 0.30",   "#F4A261"),
            ("mean_persistence", "Mean persistence",   "Maximize", "#8172B2"),
        ]
        paths = []
        x_vals = [r[x_key] for r in results]

        for key, label, target, color in metrics_cfg:
            y_vals = [r[key] for r in results]

            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.plot(x_vals, y_vals, color=color, linewidth=2, marker="o",
                    markersize=5)
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(f"{label}  (target: {target})", fontsize=12)
            ax.grid(True, alpha=0.3)

            # noise fraction: add 30% threshold line
            if key == "noise_frac":
                ax.axhline(0.30, color="#C44E52", linewidth=1.2,
                           linestyle="--", label="30% threshold")
                ax.legend(fontsize=9)

            fig.tight_layout()
            out = tmp_dir / f"coarse_{key}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            paths.append(out)

        return paths

    def _make_heatmaps(
        self,
        results:   list,
        cs_values: list,
        ms_values: list,
        tmp_dir:   Path,
    ) -> list:
        """
        Creates four heatmaps (one per metric) for a 2D parameter sweep.
        Returns a list of Path objects to the saved PNG files.
        """
        metrics_cfg = [
            ("dbcv",             "DBCV (primary)",   "viridis"),
            ("silhouette",       "Silhouette score", "viridis"),
            ("noise_frac",       "Noise fraction",   "viridis_r"),  # lower = better
            ("mean_persistence", "Mean persistence", "viridis"),
        ]

        # build lookup dict (cs, ms) -> value
        result_map = {(r["cs"], r["ms"]): r for r in results}
        paths = []

        for key, label, cmap in metrics_cfg:
            # build matrix — rows = ms_values, cols = cs_values
            valid_ms = [m for m in ms_values if m <= max(cs_values)]
            matrix   = np.full((len(valid_ms), len(cs_values)), np.nan)

            for ci, cs in enumerate(cs_values):
                for mi, ms in enumerate(valid_ms):
                    if (cs, ms) in result_map:
                        matrix[mi, ci] = result_map[(cs, ms)][key]

            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(matrix, aspect="auto", cmap=cmap, origin="lower")
            plt.colorbar(im, ax=ax, label=label)

            ax.set_xticks(range(len(cs_values)))
            ax.set_yticks(range(len(valid_ms)))
            ax.set_xticklabels(cs_values, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(valid_ms, fontsize=8)
            ax.set_xlabel("min_cluster_size", fontsize=11)
            ax.set_ylabel("min_samples",      fontsize=11)
            ax.set_title(label, fontsize=12)

            fig.tight_layout()
            out = tmp_dir / f"fine_{key}.png"
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
