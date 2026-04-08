"""
Class-based pipeline for converting NIST .mol files to RDKit Mol objects,
storing them as gzip-compressed binary datasets in HDF5, and populating the
molecule and sp_mol_map tables in the reference SQLite database.

Only mol files whose CAS number is present in the spectra table are processed.
Everything else is skipped and recorded in the run log.

Intended usage
--------------
    from build_mol_db import MolDB
    from config_loader import ConfigLoader

    cfg = ConfigLoader(config_path)
    db  = MolDB(cfg)
    db.run()
"""

import re
import sys
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np
from rdkit import Chem
from rdkit import RDLogger  # type: ignore
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

RDLogger.DisableLog("rdApp.*")  # type: ignore


class ParseMOL:
    """
    Converts NIST .mol files to RDKit Mol objects, stores them as
    gzip-compressed binary arrays in HDF5, and populates the molecule
    and sp_mol_map tables in the reference SQLite database.

    Only mol files whose CAS number exists in the spectra table are processed.
    """

    def __init__(self, cfg):
        """
        Initialises paths from the ConfigLoader instance and sets up logging.

        cfg         ConfigLoader instance with mol_dir, db_path, mol_h5_path,
                    and reports_dir keys defined in config.yaml
        """
        self.mol_dir     = cfg.get_path("mol_dir",     must_exist=True)
        self.db_path     = str(cfg.get_path("db_path", must_exist=True))
        self.mol_h5_path = str(cfg.get_path("mol_h5_path"))
        self.reports_dir = cfg.get_path("reports_dir")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # logging — writes to both console and a log file in reports_dir
        log_file = self.reports_dir / "build_mol_db.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)s  %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ]
        )
        self.log = logging.getLogger(__name__)

        # counters and failure log reset on each run() call
        self._reset_stats()

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def run(self):
        """
        Executes the full pipeline:
            1. Load valid CAS numbers from the spectra table
            2. Discover all .mol files under mol_dir
            3. Parse, convert, and store each matching mol file
            4. Populate molecule and sp_mol_map tables
            5. Write the summary PDF report
        """
        self._reset_stats()
        self.start_time = datetime.now()

        self.log.info("Starting mol object build pipeline")
        self.log.info(f"mol_dir:     {self.mol_dir}")
        self.log.info(f"db_path:     {self.db_path}")
        self.log.info(f"mol_h5_path: {self.mol_h5_path}")

        valid_cas = self._load_valid_cas()
        mol_files = self._discover_mol_files()

        self._process_files(mol_files, valid_cas)
        self._log_summary()
        self._write_report()

    # -----------------------------------------------------------------------
    # CAS extraction
    # -----------------------------------------------------------------------

    def _extract_cas(self, mol_file: Path) -> str | None:
        """
        Extracts the CAS registry number from line 3 of a NIST .mol file.

        NIST .mol files store the CAS number on line 3 in the format:
            CAS rn = 50000, Spec ID = 27

        Returns the CAS number as a string, or None if not parseable.
        """
        try:
            with open(mol_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            if len(lines) < 3:
                return None
            match = re.search(r"CAS\s+rn\s*=\s*(\d+)", lines[2], re.IGNORECASE)
            return match.group(1) if match else None
        except Exception:
            return None

    # -----------------------------------------------------------------------
    # Database helpers
    # -----------------------------------------------------------------------

    def _connect(self):
        """Returns (conn, cursor) for the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()
        return conn, cursor

    def _load_valid_cas(self) -> set:
        """
        Returns the set of all casNO values in the spectra table.
        Used to filter which mol files are worth processing.
        """
        self.log.info("Loading valid CAS numbers from spectra table...")
        conn, cursor = self._connect()
        cursor.execute("SELECT DISTINCT casNO FROM spectra")
        cas_set = {r[0] for r in cursor.fetchall()}
        conn.close()
        self.log.info(f"  {len(cas_set):,} unique CAS numbers found in spectra table")
        return cas_set

    def _get_sp_ids_for_cas(self, cursor, cas_no: str) -> list:
        """
        Returns all spID values from the spectra table for a given casNO.
        """
        cursor.execute("SELECT spID FROM spectra WHERE casNO = ?", (cas_no,))
        return [r[0] for r in cursor.fetchall()]

    def _insert_molecule(self, cursor, cas_no: str, h5_file: str) -> int:
        """
        Inserts a placeholder row into the molecule table and returns the
        new molID.  h5ID is set to an empty string initially and updated once
        the molID is known.
        """
        cursor.execute(
            "INSERT INTO molecule (casNO, h5ID, h5file) VALUES (?, ?, ?)",
            (cas_no, "", h5_file)
        )
        return cursor.lastrowid  # type: ignore

    def _update_h5id(self, cursor, mol_id: int, h5_id: str):
        """Updates the h5ID field for a molecule row after molID is known."""
        cursor.execute(
            "UPDATE molecule SET h5ID = ? WHERE molID = ?",
            (h5_id, mol_id)
        )

    def _insert_sp_mol_map(self, cursor, sp_id: int, mol_id: int):
        """
        Inserts a row into sp_mol_map.
        Silently ignores duplicate (spID, molID) pairs.
        """
        cursor.execute(
            "INSERT OR IGNORE INTO sp_mol_map (spID, molID) VALUES (?, ?)",
            (sp_id, mol_id)
        )

    # -----------------------------------------------------------------------
    # File discovery
    # -----------------------------------------------------------------------

    def _discover_mol_files(self) -> list:
        """
        Recursively finds all .mol / .MOL files under mol_dir and returns
        a deduplicated sorted list of Path objects.
        """
        self.log.info(f"Scanning for .mol files under {self.mol_dir}...")
        files = sorted(self.mol_dir.rglob("*.MOL")) + sorted(self.mol_dir.rglob("*.mol"))
        files = list(dict.fromkeys(files))  # deduplicate (case-insensitive filesystems)
        self.stats["folders_scanned"] = len({f.parent for f in files})
        self.stats["n_files_found"]   = len(files)
        self.log.info(
            f"  Found {len(files):,} .mol files across "
            f"{self.stats['folders_scanned']} folders"
        )
        return files

    # -----------------------------------------------------------------------
    # Core processing loop
    # -----------------------------------------------------------------------

    def _process_files(self, mol_files: list, valid_cas: set):
        """
        Iterates over all discovered .mol files, parses each one with RDKit,
        serializes to binary, writes to HDF5, and inserts database rows.

        Commits to the database every 5,000 files to keep transaction size
        manageable.
        """
        conn, cursor = self._connect()
        seen_cas     = {}  # casNO -> molID, tracks duplicates within this run

        with h5py.File(self.mol_h5_path, "w") as h5f:
            for i, mol_file in enumerate(mol_files):

                if i > 0 and i % 5000 == 0:
                    self.log.info(f"  Progress: {i:,} / {len(mol_files):,}")
                    conn.commit()

                # extract CAS from line 3
                cas = self._extract_cas(mol_file)
                if cas is None:
                    self.stats["n_cas_skipped"] += 1
                    self.skipped_log.append(f"[NO CAS]      {mol_file}")
                    continue

                # skip if CAS not in spectra database
                if cas not in valid_cas:
                    self.stats["n_cas_skipped"] += 1
                    continue

                self.stats["n_cas_matched"] += 1

                # track duplicates
                if cas in seen_cas:
                    self.stats["n_duplicate_cas"] += 1
                    self.log.debug(f"Duplicate CAS {cas}: {mol_file}")

                # parse with RDKit
                mol = Chem.MolFromMolFile(str(mol_file), sanitize=True, removeHs=False)
                if mol is None:
                    self.stats["n_parse_failed"] += 1
                    msg = f"[PARSE FAIL]  CAS {cas}  {mol_file}"
                    self.skipped_log.append(msg)
                    self.log.warning(msg)
                    continue

                # serialize to binary and convert to uint8 array for HDF5
                mol_binary = mol.ToBinary()
                arr        = np.frombuffer(mol_binary, dtype=np.uint8)

                # insert placeholder row to get molID, then update h5ID
                mol_id = self._insert_molecule(cursor, cas, self.mol_h5_path)
                h5_id  = f"mol_{mol_id}"
                self._update_h5id(cursor, mol_id, h5_id)

                # write binary blob to HDF5
                h5f.create_dataset(
                    h5_id,
                    data             = arr,
                    compression      = "gzip",
                    compression_opts = 4,
                )

                # link to all matching spectra
                sp_ids = self._get_sp_ids_for_cas(cursor, cas)
                for sp_id in sp_ids:
                    self._insert_sp_mol_map(cursor, sp_id, mol_id)
                    self.stats["n_map_rows"] += 1

                seen_cas[cas] = mol_id
                self.stats["n_inserted"] += 1

        conn.commit()
        conn.close()

        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.stats["run_duration_seconds"] = elapsed

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def _log_summary(self):
        """Prints the run summary to the log."""
        s    = self.stats
        mins = int(s["run_duration_seconds"] // 60)
        secs = int(s["run_duration_seconds"] % 60)
        self.log.info("=" * 60)
        self.log.info("Build complete")
        self.log.info(f"  .mol files found:          {s['n_files_found']:,}")
        self.log.info(f"  CAS matched to spectra DB: {s['n_cas_matched']:,}")
        self.log.info(f"  CAS skipped (no match):    {s['n_cas_skipped']:,}")
        self.log.info(f"  Parse failures:            {s['n_parse_failed']:,}")
        self.log.info(f"  Duplicate CAS numbers:     {s['n_duplicate_cas']:,}")
        self.log.info(f"  Molecule rows inserted:    {s['n_inserted']:,}")
        self.log.info(f"  sp_mol_map rows inserted:  {s['n_map_rows']:,}")
        self.log.info(f"  Run time:                  {mins}m {secs}s")
        self.log.info("=" * 60)

    def _write_report(self):
        """
        Writes a summary PDF report of the run to reports_dir.
        Includes a stats table and a list of any parse failures.
        """
        output_path = self.reports_dir / "mol_build_summary.pdf"
        s           = self.stats
        mins        = int(s["run_duration_seconds"] // 60)
        secs        = int(s["run_duration_seconds"] % 60)

        doc    = SimpleDocTemplate(str(output_path), pagesize=letter,
                                   leftMargin=inch, rightMargin=inch,
                                   topMargin=inch, bottomMargin=inch)
        styles = getSampleStyleSheet()
        story  = []

        title_style = ParagraphStyle("title", parent=styles["Title"],
                                     fontSize=16, spaceAfter=6)
        head_style  = ParagraphStyle("head",  parent=styles["Heading2"],
                                     fontSize=12, spaceBefore=14, spaceAfter=4)
        body_style  = ParagraphStyle("body",  parent=styles["Normal"],
                                     fontSize=10, leading=14)
        mono_style  = ParagraphStyle("mono",  parent=styles["Normal"],
                                     fontSize=8,  leading=11,
                                     fontName="Courier")

        story.append(Paragraph("FragTree — Mol Object Build Report", title_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            body_style
        ))
        story.append(Spacer(1, 0.2 * inch))

        # summary stats table
        story.append(Paragraph("Run Summary", head_style))
        table_data = [
            ["Metric",                          "Value"],
            ["Folders scanned",                 str(s["folders_scanned"])],
            [".mol files found",                f"{s['n_files_found']:,}"],
            ["CAS matched to spectra DB",       f"{s['n_cas_matched']:,}"],
            ["CAS not in spectra DB (skipped)", f"{s['n_cas_skipped']:,}"],
            ["Parse failures (RDKit)",          f"{s['n_parse_failed']:,}"],
            ["Duplicate CAS numbers seen",      f"{s['n_duplicate_cas']:,}"],
            ["Molecule rows inserted",          f"{s['n_inserted']:,}"],
            ["sp_mol_map rows inserted",        f"{s['n_map_rows']:,}"],
            ["Run time",                        f"{mins}m {secs}s"],
        ]
        tbl = Table(table_data, colWidths=[3.5 * inch, 2.5 * inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#0A1628")),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#EEF4F4"), colors.white]),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#CCDDDD")),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.2 * inch))

        # coverage note
        if s["n_files_found"] > 0:
            pct = 100 * s["n_cas_matched"] / s["n_files_found"]
            story.append(Paragraph(
                f"Coverage: {pct:.1f}% of discovered .mol files had a matching "
                f"CAS number in the spectra database and were processed.",
                body_style
            ))
        story.append(Spacer(1, 0.2 * inch))

        # parse failure list
        story.append(Paragraph("Parse Failures", head_style))
        if self.skipped_log:
            story.append(Paragraph(
                "The following .mol files were skipped because RDKit returned "
                "None. This usually indicates a malformed V2000 block or an "
                "unsupported feature. The full list is also in build_mol_db.log.",
                body_style
            ))
            story.append(Spacer(1, 0.1 * inch))
            for entry in self.skipped_log[:200]:
                story.append(Paragraph(entry, mono_style))
            if len(self.skipped_log) > 200:
                story.append(Paragraph(
                    f"... and {len(self.skipped_log) - 200} more "
                    f"(see build_mol_db.log for full list)",
                    body_style
                ))
        else:
            story.append(Paragraph("No parse failures.", body_style))

        doc.build(story)
        self.log.info(f"Summary report saved to {output_path}")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _reset_stats(self):
        """Resets all run counters and the failure log."""
        self.stats = {
            "folders_scanned":      0,
            "n_files_found":        0,
            "n_cas_matched":        0,
            "n_cas_skipped":        0,
            "n_parse_failed":       0,
            "n_duplicate_cas":      0,
            "n_inserted":           0,
            "n_map_rows":           0,
            "run_duration_seconds": 0.0,
        }
        self.skipped_log: list = []
        self.start_time        = datetime.now()
