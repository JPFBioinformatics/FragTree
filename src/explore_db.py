"""

Class for exploring the TMS reference database visually.
Run this file directly to display all plots across two figure windows and
save both as PDFs in the reports/ directory at the project root.

Metadata plots (Figure 1):
    1.  CAS number replicate histogram       — distribution of spectra per unique compound
    2.  Top 20 most replicated compounds     — which compounds dominate the dataset
    3.  Molecular weight distribution        — MW range and shape across all spectra
    4.  Number of peaks distribution         — spectral complexity across all spectra
    5.  Retention index distribution         — GC separation space coverage
    6.  Top 20 most common formulas          — dominant compound classes

Spectral plots (Figure 2):
    7.  Most common fragment histogram       — top N m/z values by occurrence count
    8.  Summed relative abundance histogram  — top N m/z values weighted by intensity
    9.  m/z hexbin density plot              — 2D density of m/z vs log intensity
    10. Base peak distribution               — most intense fragment per spectrum
    11. Spectral similarity distribution     — pairwise cosine similarity on a random sample
    12. Intensity distribution               — all raw intensity values (log scale)
    13. Fragment count vs MW scatter         — peaks per spectrum vs molecular weight
"""

# region Imports

import re
import sqlite3
import sys
from pathlib import Path
from collections import defaultdict
from random import sample

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import h5py

# path setup so this can be run from the scripts/ directory
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config_loader import ConfigLoader

# endregion

class ExploreDB:
    """
    Connects to the TMS metadata SQLite database and HDF5 spectral file and
    provides methods to query them and generate exploratory plots.
    """

    def __init__(self, db_path, h5_path):
        """
        Stores paths to the database and HDF5 spectral file.

        Parameters
        ----------
        db_path             Path to metadata.db
        h5_path             Path to spectra.h5
        """
        self.db_path = str(db_path)
        self.h5_path = str(h5_path)

    # region Connection and query helpers

    def connect(self):
        """
        Opens a connection to the SQLite database.

        Returns
        -------
        (conn, cursor)      open connection and cursor, or (None, None) on failure
        """
        try:
            conn   = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            return conn, cursor
        except sqlite3.Error as e:
            print(f"Failed to connect to database at {self.db_path}\nError: {e}")
            return None, None

    def run_query(self, query, params=(), fetch=True):
        """
        Executes a SQL query and optionally returns results.

        Parameters
        ----------
        query               SQL string with ? as parameter placeholders
        params              tuple of values to substitute for placeholders
        fetch               if True, return rows for SELECT/PRAGMA queries

        Returns
        -------
        results             list of row tuples, or empty list
        """
        conn, cursor = self.connect()
        if not conn or not cursor:
            return []

        try:
            cursor.execute(query, params)
            command = query.strip().split()[0].upper()

            if command in {"INSERT", "UPDATE", "DELETE", "CREATE", "DROP"}:
                conn.commit()
            elif fetch and command in {"SELECT", "PRAGMA"}:
                return cursor.fetchall()

            return []

        except sqlite3.Error as e:
            conn.rollback()
            print(f"Query failed — {e}\nSQL: {query}")
            return []

        finally:
            conn.close()

    # endregion

    # region Metadata queries

    def count_spectra(self):
        """
        Returns the total number of spectra in the database.

        Returns
        -------
        count               integer total spectrum count
        """
        result = self.run_query("SELECT COUNT(*) FROM spectra")
        return result[0][0] if result else 0

    def count_unique_compounds(self):
        """
        Returns the number of unique CAS numbers in the database.

        Returns
        -------
        count               integer unique CAS number count
        """
        result = self.run_query("SELECT COUNT(DISTINCT casNO) FROM spectra")
        return result[0][0] if result else 0

    def get_replicate_counts(self):
        """
        Returns the number of spectra for each unique CAS number.

        Returns
        -------
        replicates          dict mapping casNO -> spectrum count
        """
        results = self.run_query(
            "SELECT casNO, COUNT(spID) FROM spectra GROUP BY casNO"
        )
        return dict(results)

    def get_mw_values(self):
        """
        Returns molecular weight values for all spectra.

        Returns
        -------
        mws                 list of float MW values
        """
        results = self.run_query("SELECT mw FROM spectra WHERE mw IS NOT NULL")
        return [r[0] for r in results]

    def get_num_peaks_values(self):
        """
        Returns peak counts for all spectra.

        Returns
        -------
        peaks               list of int peak counts
        """
        results = self.run_query("SELECT numPeaks FROM spectra WHERE numPeaks IS NOT NULL")
        return [r[0] for r in results]

    def get_retention_indices(self):
        """
        Returns numeric retention index values extracted from NIST compound
        strings such as 'SemiStdNP=843/5/6 StdNP=812/2/2 StdPolar=1522/8/14'.

        The first integer value found after any '=' sign is used as the
        retention index for that spectrum.  Entries with no parseable value
        (nulls, 'n/a', free-text only) are silently skipped.

        Returns
        -------
        values              list of float retention index values
        """
        results = self.run_query(
            "SELECT retentionIndex FROM spectra WHERE retentionIndex IS NOT NULL"
        )
        values = []
        for (val,) in results:
            if not val or val.strip().lower() == "n/a":
                continue
            # try plain float first (simple entries)
            try:
                values.append(float(val))
                continue
            except (ValueError, TypeError):
                pass
            # extract first number after an '=' sign from compound RI strings
            match = re.search(r"=(\d+)", str(val))
            if match:
                values.append(float(match.group(1)))
        return values

    def get_formula_counts(self):
        """
        Returns all molecular formulas and their occurrence counts.

        Returns
        -------
        rows                list of (formula, count) tuples sorted by count descending
        """
        return self.run_query(
            "SELECT formula, COUNT(*) as n FROM spectra "
            "WHERE formula IS NOT NULL GROUP BY formula ORDER BY n DESC"
        )

    def get_top_replicated(self, n=20):
        """
        Returns the top n compounds by replicate count.

        Parameters
        ----------
        n                   number of compounds to return (default 20)

        Returns
        -------
        rows                list of (spName, casNO, count) tuples
        """
        return self.run_query(
            "SELECT spName, casNO, COUNT(spID) as n FROM spectra "
            "GROUP BY casNO ORDER BY n DESC LIMIT ?",
            params=(n,)
        )

    # endregion

    # region Spectral data loaders

    def load_all_spectra(self):
        """
        Loads all spectra from the HDF5 file.

        Returns
        -------
        spectra             list of (N, 2) np.ndarray arrays — col 0 = m/z, col 1 = intensity
        """
        spectra = []
        with h5py.File(self.h5_path, "r") as f:
            for key in f.keys():
                spectra.append(np.array(f[key]))
        return spectra

    def load_spectra_sample(self, n=500):
        """
        Loads a random sample of up to n spectra from the HDF5 file.
        Used for computationally expensive plots like similarity distributions.

        Parameters
        ----------
        n                   maximum number of spectra to load (default 500)

        Returns
        -------
        spectra             list of (N, 2) np.ndarray arrays
        """
        with h5py.File(self.h5_path, "r") as f:
            keys         = list(f.keys())
            sampled_keys = sample(keys, min(n, len(keys)))
            return [np.array(f[k]) for k in sampled_keys]

    def compute_fragment_stats(self, spectra):
        """
        Computes per-m/z occurrence counts and summed relative abundances
        across all spectra.

        Intensities are normalized to the base peak within each spectrum
        before summing so all spectra contribute equally regardless of
        absolute intensity scale.

        Parameters
        ----------
        spectra             list of (N, 2) np.ndarray arrays

        Returns
        -------
        occurrence          dict mapping m/z -> number of spectra it appears in
        summed_abundance    dict mapping m/z -> sum of normalized intensities
        """
        occurrence       = defaultdict(int)
        summed_abundance = defaultdict(float)

        for arr in spectra:
            mzs         = arr[:, 0]
            intensities = arr[:, 1].astype(float)
            max_int     = intensities.max()
            if max_int == 0:
                continue
            norm = intensities / max_int
            for mz, rel_int in zip(mzs, norm):
                occurrence[int(mz)]       += 1
                summed_abundance[int(mz)] += rel_int

        return dict(occurrence), dict(summed_abundance)

    # endregion

    # region Metadata plots

    def plot_replicate_histogram(self, ax):
        """
        Histogram of spectra count per unique CAS number, annotated with
        total spectra and unique compound counts.

        Parameters
        ----------
        ax                  matplotlib Axes to draw on
        """
        replicates = self.get_replicate_counts()
        counts     = list(replicates.values())
        total      = self.count_spectra()
        n_unique   = self.count_unique_compounds()
        max_rep    = max(counts)

        ax.hist(counts, bins=range(1, max_rep + 2), color="#4C72B0",
                edgecolor="white", linewidth=0.4)
        ax.set_title(
            f"CAS number replicate distribution\n"
            f"Total spectra: {total:,}   |   Unique compounds: {n_unique:,}",
            fontsize=11, pad=10
        )
        ax.set_xlabel("Spectra per CAS number")
        ax.set_ylabel("Number of compounds")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    def plot_top_replicated(self, ax, n=20):
        """
        Table showing how many compounds have each replicate count.
 
        The n parameter is unused but kept for API compatibility with
        any existing call sites.
 
        Parameters
        ----------
        ax                  matplotlib Axes to draw on
        n                   unused, kept for API compatibility
        """
        from collections import Counter
 
        replicates  = self.get_replicate_counts()
        if not replicates:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return
 
        # count how many compounds have each replicate count
        rep_counter  = Counter(replicates.values())
        sorted_rows  = sorted(rep_counter.items())  # ascending by spectra count
        total        = sum(replicates.values())
 
        table_data = [
            [str(n_spectra), f"{n_compounds:,}"]
            for n_spectra, n_compounds in sorted_rows
        ]
 
        ax.axis("off")
        ax.set_title("Replicate count summary", fontsize=11, pad=10)
 
        table = ax.table(
            cellText=table_data,
            colLabels=["Spectra per CAS number", "Number of compounds"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.6)
 
        # style header row
        for col in range(2):
            table[0, col].set_facecolor("#4C72B0")
            table[0, col].set_text_props(color="white", fontweight="bold")
 
        # alternate row shading for readability
        for row in range(1, len(table_data) + 1):
            color = "#EEF2F8" if row % 2 == 0 else "white"
            for col in range(2):
                table[row, col].set_facecolor(color)
 
        ax.text(0.5, 0.02, f"Total spectra: {total:,}",
                transform=ax.transAxes, fontsize=9,
                ha="center", va="bottom", color="#555555")

    def plot_mw_distribution(self, ax):
        """
        Histogram of molecular weight across all spectra.

        Parameters
        ----------
        ax                  matplotlib Axes to draw on
        """
        mws = self.get_mw_values()
        if not mws:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        ax.hist(mws, bins=60, color="#55A868", edgecolor="white", linewidth=0.4)
        ax.set_title(
            f"Molecular weight distribution\n"
            f"Mean: {np.mean(mws):.1f}   Median: {np.median(mws):.1f}   "
            f"Range: {min(mws):.0f} – {max(mws):.0f}",
            fontsize=11, pad=10
        )
        ax.set_xlabel("Molecular weight (Da)")
        ax.set_ylabel("Number of spectra")

    def plot_num_peaks_distribution(self, ax):
        """
        Histogram of number of peaks per spectrum.

        Parameters
        ----------
        ax                  matplotlib Axes to draw on
        """
        peaks = self.get_num_peaks_values()
        if not peaks:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        ax.hist(peaks, bins=50, color="#C44E52", edgecolor="white", linewidth=0.4)
        ax.set_title(
            f"Peaks per spectrum\n"
            f"Mean: {np.mean(peaks):.1f}   Median: {np.median(peaks):.1f}   "
            f"Range: {min(peaks)} – {max(peaks)}",
            fontsize=11, pad=10
        )
        ax.set_xlabel("Number of peaks")
        ax.set_ylabel("Number of spectra")

    def plot_retention_index_distribution(self, ax):
        """
        Histogram of retention index values across all spectra.

        NIST stores RI as compound strings (e.g. 'SemiStdNP=843/5/6') —
        get_retention_indices() extracts the first numeric value from each.
        Entries with no parseable value are silently skipped.

        Parameters
        ----------
        ax                  matplotlib Axes to draw on
        """
        ris = self.get_retention_indices()
        if not ris:
            ax.text(0.5, 0.5, "No retention index data", ha="center", va="center")
            return

        ax.hist(ris, bins=60, color="#8172B2", edgecolor="white", linewidth=0.4)
        ax.set_title(
            f"Retention index distribution  (n={len(ris):,})\n"
            f"Mean: {np.mean(ris):.0f}   Median: {np.median(ris):.0f}   "
            f"Range: {min(ris):.0f} – {max(ris):.0f}",
            fontsize=11, pad=10
        )
        ax.set_xlabel("Retention index")
        ax.set_ylabel("Number of spectra")

    def plot_formula_frequency(self, ax, n=20):
        """
        Horizontal bar chart of the top n most common molecular formulas.

        Parameters
        ----------
        ax                  matplotlib Axes to draw on
        n                   number of formulas to show (default 20)
        """
        rows = self.get_formula_counts()[:n]
        if not rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        labels = [r[0] for r in rows]
        values = [r[1] for r in rows]

        ax.barh(labels[::-1], values[::-1], color="#CCB974",
                edgecolor="white", linewidth=0.4)
        ax.set_title(f"Top {n} most common molecular formulas", fontsize=11, pad=10)
        ax.set_xlabel("Number of spectra")
        ax.tick_params(axis="y", labelsize=8)

    # endregion

    # region Spectral plots

    def plot_fragment_occurrence(self, ax, occurrence, total_spectra, n=100):
        """
        Bar chart of the top n m/z values by number of spectra they appear in,
        annotated with the total fragment observation count.

        Parameters
        ----------
        ax                  matplotlib Axes to draw on
        occurrence          dict mapping m/z -> occurrence count
        total_spectra       total number of spectra, shown in title annotation
        n                   number of m/z values to show (default 40)
        """
        sorted_mz   = sorted(occurrence, key=occurrence.get, reverse=True)[:n]
        counts      = [occurrence[mz] for mz in sorted_mz]
        total_frags = sum(occurrence.values())

        ax.bar([str(mz) for mz in sorted_mz], counts, color="#4C72B0",
               edgecolor="white", linewidth=0.4)
        ax.set_title(
            f"Top {n} most common fragments\n"
            f"Total fragment observations: {total_frags:,}   |   Spectra: {total_spectra:,}",
            fontsize=11, pad=10
        )
        ax.set_xlabel("m/z")
        ax.set_ylabel("Number of spectra containing fragment")
        ax.tick_params(axis="x", rotation=90, labelsize=7)

    def plot_summed_abundance(self, ax, summed_abundance, n=100):
        """
        Bar chart of the top n m/z values by summed normalized relative abundance.
 
        Intensities are normalized to the base peak within each spectrum before
        summing, so all spectra contribute equally regardless of absolute scale.
 
        Parameters
        ----------
        ax                  matplotlib Axes to draw on
        summed_abundance    dict mapping m/z -> summed relative abundance
        n                   number of m/z values to show (default 40)
        """
        # exclude m/z 73 — it dominates so heavily it washes out everything else
        # its value is annotated in a box instead
        mz73_val    = summed_abundance.get(73, 0)
        filtered    = {mz: v for mz, v in summed_abundance.items() if mz != 73}
        sorted_mz   = sorted(filtered, key=lambda mz: filtered[mz], reverse=True)[:n]
        values      = [filtered[mz] for mz in sorted_mz]
 
        ax.bar([str(mz) for mz in sorted_mz], values, color="#55A868",
               edgecolor="white", linewidth=0.4)
        ax.set_title(f"Top {n} fragments by summed relative abundance\n(m/z 73 excluded — see annotation)",
                     fontsize=11, pad=10)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Summed normalized intensity")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
 
        # annotate excluded m/z 73 value in top-right corner
        ax.text(0.97, 0.97, f"m/z 73 (excluded)\n{mz73_val:,.0f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#55A868", linewidth=1))
        
    def plot_mz_density_hexbin(self, ax, spectra):
        """
        Hexbin density plot of m/z vs log intensity across all spectra.

        Hexbin handles the extreme density variation in mass spectral data much
        better than hist2d — each hex is colored by the log count of peaks
        falling within it, so both common low-mass fragments and rare high-mass
        fragments are visible simultaneously.

        Parameters
        ----------
        ax                  matplotlib Axes to draw on
        spectra             list of (N, 2) np.ndarray arrays (mz, intensity)
        """
        all_mz  = np.concatenate([arr[:, 0] for arr in spectra])
        all_int = np.concatenate([arr[:, 1] for arr in spectra]).astype(float)
        log_int = np.log1p(all_int)

        hb = ax.hexbin(all_mz, log_int, gridsize=80, cmap="viridis",
                       bins="log", mincnt=1)
        plt.colorbar(hb, ax=ax, label="log(count)")
        ax.set_title("m/z vs intensity density (hexbin)", fontsize=11, pad=10)
        ax.set_xlabel("m/z")
        ax.set_ylabel("log(intensity + 1)")

    def plot_base_peak_distribution(self, ax, spectra):
        """
        Bar chart of the most common base peak m/z values across all spectra.
 
        For a correctly filtered TMS dataset m/z 73 should strongly dominate.
        Unexpected base peaks may indicate misclassified spectra.
 
        Parameters
        ----------
        ax                  matplotlib Axes to draw on
        spectra             list of (N, 2) np.ndarray arrays (mz, intensity)
        """
        base_peaks = []
        for arr in spectra:
            if arr.shape[0] == 0:
                continue
            base_peaks.append(int(arr[np.argmax(arr[:, 1]), 0]))
 
        if not base_peaks:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return
 
        unique_bp, bp_counts = np.unique(base_peaks, return_counts=True)
 
        # pull out m/z 73 count before filtering so we can annotate it
        mz73_mask  = unique_bp == 73
        mz73_count = int(bp_counts[mz73_mask][0]) if mz73_mask.any() else 0
 
        # exclude m/z 73 — it dominates and washes out the rest
        keep       = unique_bp != 73
        unique_bp  = unique_bp[keep]
        bp_counts  = bp_counts[keep]
 
        top_idx    = np.argsort(bp_counts)[::-1][:30]
        top_mz     = unique_bp[top_idx]
        top_counts = bp_counts[top_idx]
        sort_order = np.argsort(top_mz)
 
        ax.bar([str(mz) for mz in top_mz[sort_order]], top_counts[sort_order],
               color="#C44E52", edgecolor="white", linewidth=0.4)
        ax.set_title(
            f"Base peak distribution  (n={len(base_peaks):,} spectra)\n"
            f"(m/z 73 excluded — see annotation)",
            fontsize=11, pad=10
        )
        ax.set_xlabel("m/z of base peak")
        ax.set_ylabel("Number of spectra")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
 
        # annotate excluded m/z 73 count in top-right corner
        ax.text(0.97, 0.97, f"m/z 73 (excluded)\n{mz73_count:,} spectra",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#C44E52", linewidth=1))

    def plot_intensity_distribution(self, ax, spectra):
        """
        Histogram of all raw intensity values across all spectra on a log scale.
        Useful for deciding whether normalization is needed before clustering.

        Parameters
        ----------
        ax                  matplotlib Axes to draw on
        spectra             list of (N, 2) np.ndarray arrays (mz, intensity)
        """
        all_intensities = np.concatenate([arr[:, 1] for arr in spectra])
        log_int         = np.log1p(all_intensities.astype(float))

        ax.hist(log_int, bins=80, color="#CCB974", edgecolor="white", linewidth=0.4)
        ax.set_title(
            f"Intensity distribution (log scale)  (n={len(all_intensities):,} peaks)\n"
            f"Raw range: {all_intensities.min():.0f} – {all_intensities.max():.0f}",
            fontsize=11, pad=10
        )
        ax.set_xlabel("log(intensity + 1)")
        ax.set_ylabel("Number of peaks")

    def plot_fragment_count_vs_mw(self, ax):
        """
        Scatter plot of number of peaks per spectrum vs molecular weight

        Expects a positive correlation — heavier molecules tend to fragment more.
        Outliers (high MW / few peaks, or low MW / many peaks) may indicate
        low-quality spectra worth investigating.

        Parameters
        ----------
        ax                  matplotlib Axes to draw on
        """
        rows = self.run_query(
            "SELECT mw, numPeaks FROM spectra "
            "WHERE mw IS NOT NULL AND numPeaks IS NOT NULL"
        )
        if not rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        mws   = np.array([r[0] for r in rows], dtype=float)
        peaks = np.array([r[1] for r in rows], dtype=float)

        ax.scatter(mws, peaks, alpha=0.15, s=4, color="#4C72B0", linewidths=0)
        ax.set_title("Fragment count vs molecular weight", fontsize=11, pad=10)
        ax.set_xlabel("Molecular weight (Da)")
        ax.set_ylabel("Number of peaks")

    def plot_similarity_distribution(self, ax, n_sample=500, top_pct = 0.2, min_mz = 100):
        """
        Histogram of pairwise cosine similarities across a random sample of spectra.

        Spectra are represented as intensity vectors binned by integer m/z and
        L2-normalized before computing dot products.  A right-skewed distribution
        (most pairs near 0) means good chemical diversity for clustering.  A peak
        near 1.0 means many spectra look nearly identical.

        Parameters
        ----------
        ax                  matplotlib Axes to draw on
        n_sample            number of spectra to sample for comparison (default 500)
        top_pct             percentile to display for visualization of thresholding
        min_mz              minimum m/z to be included in the calculations
        """
        if n_sample != 0:
            print(f"  Computing pairwise similarity for {n_sample} sampled spectra...")
            spectra = self.load_spectra_sample(n_sample)
        else:
            print("Computing pairwise similarity for all spectra")
            spectra = self.load_all_spectra()

        max_mz = 800
        matrix = np.zeros((len(spectra), max_mz + 1), dtype=np.float32)

        for i, arr in enumerate(spectra):
            arr = np.array(arr)
            for mz, intensity in arr:
                idx = int(mz)
                if idx <= max_mz:
                    matrix[i, idx] += float(intensity)

        # remove tms canonical fragments (inflates cos similarities)
        matrix[:, [73,147,149]] = 0

        # remove all fragments less than threshold
        matrix[:,:min_mz] = 0

        # normalize matrix
        norms             = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        matrix            = matrix / norms

        # generate cosine similarity matrix and grab upper triangle (prevent duplication)
        sim_matrix = matrix @ matrix.T
        upper_tri  = sim_matrix[np.triu_indices(len(spectra), k=1)]

        threshold = np.quantile(upper_tri, 1 - top_pct)

        ax.hist(upper_tri, bins=50, color="#8172B2", edgecolor="white", linewidth=0.4)
        ax.axvline(threshold, color="red", linestyle="--", linewidth=1.2,
                   label=f"Top {int(top_pct * 100)}% (≥{threshold:.3f})")
        ax.text(threshold, ax.get_ylim()[1] * 0.95, f"Top {int(top_pct * 100)}%",
                color="red", ha="left", va="top", fontsize=9)
        ax.set_title(
            f"Pairwise cosine similarity  (n={len(spectra)} spectra sampled)\n"
            f"Mean: {upper_tri.mean():.3f}   Median: {np.median(upper_tri):.3f}",
            fontsize=11, pad=10
        )
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Number of pairs")

    # endregion

    # region Structure plots

    def plot_mol_coverage(self, ax):
        """
        Bar chart showing how many unique CAS numbers have a mol structure
        vs how many do not.  Annotates the coverage percentage.

        Parameters
        ----------
        ax                  matplotlib Axes to draw on
        """
        # total unique CAS numbers in spectra table
        total = self.run_query("SELECT COUNT(DISTINCT casNO) FROM spectra")
        n_total = total[0][0] if total else 0

        # unique CAS numbers that have at least one mol entry
        covered = self.run_query(
            "SELECT COUNT(DISTINCT casNO) FROM molecule"
        )
        n_covered = covered[0][0] if covered else 0
        n_missing = n_total - n_covered
        pct = 100 * n_covered / n_total if n_total > 0 else 0

        ax.bar(["Has mol structure", "No mol structure"],
            [n_covered, n_missing],
            color=["#0D7377", "#C44E52"],
            edgecolor="white", linewidth=0.4)
        ax.set_title(
            f"Mol structure coverage\n"
            f"{n_covered:,} / {n_total:,} unique compounds ({pct:.1f}%)",
            fontsize=11, pad=10
        )
        ax.set_ylabel("Number of compounds")
        for i, v in enumerate([n_covered, n_missing]):
            ax.text(i, v + n_total * 0.01, f"{v:,}",
                    ha="center", fontsize=10)

    def plot_mol_coverage_by_replicates(self, ax):
        """
        Grouped bar chart showing mol structure coverage split by whether a
        compound has 1 spectrum or 2+ spectra.  Well-replicated compounds
        being covered structurally is a good sign for training data quality.

        Parameters
        ----------
        ax                  matplotlib Axes to draw on
        """
        # CAS numbers with exactly 1 spectrum, split by mol coverage
        rows = self.run_query("""
            SELECT
                CASE WHEN rep.n = 1 THEN '1 spectrum' ELSE '2+ spectra' END as group_,
                CASE WHEN m.casNO IS NOT NULL THEN 'Has mol' ELSE 'No mol' END as covered,
                COUNT(*) as n
            FROM (
                SELECT casNO, COUNT(spID) as n
                FROM spectra
                GROUP BY casNO
            ) rep
            LEFT JOIN (
                SELECT DISTINCT casNO FROM molecule
            ) m ON m.casNO = rep.casNO
            GROUP BY group_, covered
        """)

        if not rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        # parse into groups
        data: dict = {"1 spectrum": {}, "2+ spectra": {}}
        for group_, covered, n in rows:
            data[group_][covered] = n

        groups   = ["1 spectrum", "2+ spectra"]
        has_mol  = [data[g].get("Has mol", 0) for g in groups]
        no_mol   = [data[g].get("No mol",  0) for g in groups]

        x     = range(len(groups))
        width = 0.35
        ax.bar([i - width/2 for i in x], has_mol, width,
            label="Has mol", color="#0D7377", edgecolor="white", linewidth=0.4)
        ax.bar([i + width/2 for i in x], no_mol, width,
            label="No mol",  color="#C44E52", edgecolor="white", linewidth=0.4)

        ax.set_xticks(list(x))
        ax.set_xticklabels(groups)
        ax.set_title("Mol coverage by replicate count", fontsize=11, pad=10)
        ax.set_ylabel("Number of compounds")
        ax.legend(fontsize=9)

    # endregion
