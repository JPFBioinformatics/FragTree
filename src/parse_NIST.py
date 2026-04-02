"""

Parses NIST mass spectral data from .msp files (primary format for this
project) or .sdf files (parser retained for potential future use) and stores
TMS-derivatized spectra in a SQLite database and a gzip-compressed HDF5 file.

Classification strategy
-----------------------
A spectrum is accepted as TMS-derivatized only if it passes BOTH checks in
this order:

  1. Fragment check  — m/z list contains >= 2 of the canonical TMS ions
                       (73, 147, 149).  A compound without these ions cannot
                       be TMS-derivatized, so this gate eliminates the bulk
                       of non-TMS spectra cheaply before touching the name.

  2. Name check      — compound name contains 'tms' or 'trimethylsilyl'
                       (case-insensitive).  This removes false positives that
                       happen to carry those fragment ions for unrelated
                       chemical reasons.

Anything that fails either check is discarded.  If a compound would qualify
as both TMS and TBDMS it is kept (TMS takes priority).

.msp key normalization
----------------------
Raw keys from .msp files are mapped to a consistent set of uppercase keys via
MSP_KEY_MAP on parse.  The column_map in populate_refDB is the single source
of truth between those normalized keys and SQL column names.

Lib2NIST export settings
-------------------------
Binning option: add 0.3 to all m/z before rounding.
"""

from pathlib import Path
import h5py
import numpy as np
import sqlite3


# ---------------------------------------------------------------------------
# .msp key normalization
# ---------------------------------------------------------------------------

# Maps raw key strings from .msp files to normalized uppercase keys used
# everywhere else in the pipeline.  Add entries here if NIST adds new fields
# or if you encounter variant spellings in your export.
MSP_KEY_MAP = {
    "name":            "NAME",
    "retention_index": "RETENTION INDEX",
    "retentionindex":  "RETENTION INDEX",
    "inchikey":        "INCHIKEY",
    "formula":         "FORMULA",
    "mw":              "MW",
    "exactmass":       "EXACT MASS",
    "casno":           "CASNO",
    "id":              "ID",
    "comment":         "COMMENT",
    "num peaks":       "NUM PEAKS",
    "numpeaks":        "NUM PEAKS",
}


def _normalize_msp_key(raw_key: str) -> str:
    """Returns the canonical uppercase key for a raw .msp key string."""
    return MSP_KEY_MAP.get(raw_key.strip().lower(), raw_key.strip().upper())


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ParseNIST:
    """
    Parses NIST .msp files, filters to TMS-derivatized spectra only, and
    writes metadata to a SQLite database and spectral data to a
    gzip-compressed HDF5 file.

    .sdf parsing is also supported (see parse_sdf) but is not used by the
    active pipeline — all production paths use .msp.
    """

    def __init__(self, db_dir):
        """
        Sets up directory paths and ensures the TMS output directory exists.

        db_dir      name of the subdirectory to create under
                    <project_root>/databases/reference/ for TMS output
                    (e.g. 'tms')
        """
        self.src_dir   = Path(__file__).resolve().parent
        self.proj_root = self.src_dir.parent
        self.data_dir  = self.proj_root / "databases"
        self.ref_dir   = self.data_dir / "reference"

        self.tms_dir = self.ref_dir / db_dir
        self.tms_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Parsers
    # -----------------------------------------------------------------------

    def parse_msp(self, file):
        """
        Parses a single .msp file and returns a list of record dicts.

        All metadata keys are normalized via MSP_KEY_MAP so downstream code
        works identically regardless of minor formatting differences between
        NIST exports.  Optional keys default to 'n/a' or '' if absent.

        Each returned dict contains:
            NAME, RETENTION INDEX, INCHIKEY, FORMULA, MW, EXACT MASS,
            CASNO, ID, COMMENT, NUM PEAKS,
            mzs (list[int]), intensities (list[int])
        """
        records             = []
        current_record      = {}
        current_mzs         = []
        current_intensities = []

        def _flush():
            if not current_record:
                return
            if current_mzs:
                current_record["mzs"]         = current_mzs.copy()
                current_record["intensities"] = current_intensities.copy()
            # ensure optional fields always exist so downstream inserts don't
            # need to guard against missing keys
            current_record.setdefault("RETENTION INDEX", "n/a")
            current_record.setdefault("INCHIKEY",        "n/a")
            current_record.setdefault("COMMENT",         "")
            records.append(current_record.copy())

        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # blank line signals the end of one record
                if not line:
                    _flush()
                    current_record      = {}
                    current_mzs         = []
                    current_intensities = []
                    continue

                # metadata line — key: value
                if ":" in line:
                    raw_key, value = line.split(":", 1)
                    key = _normalize_msp_key(raw_key)
                    current_record[key] = value.strip()

                # peak line — mz intensity (no colon present)
                else:
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        try:
                            current_mzs.append(int(parts[0]))
                            current_intensities.append(int(parts[1]))
                        except ValueError:
                            pass  # malformed peak line — skip silently

        # flush the last record if the file has no trailing blank line
        _flush()
        return records

    def parse_sdf(self, file):
        """
        Parses a single .sdf file and returns a list of record dicts.

        NOTE: not used in the active pipeline — retained for potential future
        use.  Output format mirrors parse_msp() so the rest of the pipeline
        is format-agnostic if .sdf support is reactivated.
        """
        records        = []
        current_record = {}

        with open(file, "r", encoding="utf-8") as f:
            line_num  = 1
            line      = next(f, None)
            next_line = next(f, None)

            while line is not None:
                line = line.strip()

                if line == "$$$$":
                    if current_record:
                        current_record.setdefault("INCHIKEY",        "n/a")
                        current_record.setdefault("RETENTION INDEX", "n/a")
                        current_record.setdefault("COMMENT",         "")
                        records.append(current_record)
                        current_record = {}
                    line      = next_line
                    next_line = next(f, None)
                    line_num  += 1
                    continue

                elif line.startswith(">"):
                    parts = line.split("<", 1)
                    if len(parts) > 1:
                        key = parts[1].rstrip(">").strip()

                        if key == "MASS SPECTRAL PEAKS":
                            mz_list        = []
                            intensity_list = []
                            line      = next_line
                            next_line = next(f, None)
                            line_num  += 1

                            while line is not None and line.strip() != "":
                                line = line.strip()
                                if line in ("", "$$$$") or line.startswith(">"):
                                    break
                                parts2 = line.split(None, 1)
                                if len(parts2) == 2:
                                    try:
                                        mz_list.append(int(parts2[0]))
                                        intensity_list.append(int(parts2[1]))
                                    except ValueError:
                                        pass
                                line      = next_line
                                next_line = next(f, None)
                                line_num  += 1

                            current_record["mzs"]         = mz_list
                            current_record["intensities"] = intensity_list
                            line      = next_line
                            next_line = next(f, None)
                            line_num  += 1
                            continue

                        elif next_line is not None:
                            current_record[key] = next_line.strip()

                line      = next_line
                next_line = next(f, None)
                line_num  += 1

        return records

    # -----------------------------------------------------------------------
    # Bulk file loading
    # -----------------------------------------------------------------------

    def combine_to_list(self, directory, file_type="msp"):
        """
        Parses every .msp (default) or .sdf file in directory and returns a
        flat list of all record dicts.
        """
        directory   = Path(directory)
        all_records = []

        files = sorted(directory.glob(f"*.{file_type}"))
        if not files:
            print(f"[warn] no .{file_type} files found in {directory}")
            return all_records

        for file in files:
            records = self.parse_msp(file) if file_type == "msp" else self.parse_sdf(file)
            all_records.extend(records)
            print(f"  parsed {file.name}  ({len(records)} spectra)")

        print(f"\nTotal spectra loaded: {len(all_records)}")
        return all_records

    # -----------------------------------------------------------------------
    # TMS classification
    # -----------------------------------------------------------------------

    # canonical TMS fragment ions:
    # m/z 73  = Si(CH3)3+      — trimethylsilyl cation, ubiquitous in TMS spectra
    # m/z 147 = [TMS2 - CH3]+  — characteristic of bis-TMS compounds
    # m/z 149 = [TMS2 - CH3]+1 — isotope / rearrangement partner of 147
    _TMS_FRAGMENTS    = frozenset({73, 147, 149})
    _TMS_FRAGMENT_MIN = 2  # minimum matching ions required to pass the fragment gate

    # name substrings that confirm TMS derivatization (matched case-insensitively)
    _TMS_NAME_KEYWORDS = frozenset(["tms", "trimethylsilyl"])

    def _passes_fragment_check(self, record):
        """
        Returns True if the record's m/z list contains at least
        _TMS_FRAGMENT_MIN of the canonical TMS fragment ions.
        """
        mzs = frozenset(record.get("mzs", []))
        return len(mzs & self._TMS_FRAGMENTS) >= self._TMS_FRAGMENT_MIN

    def _passes_name_check(self, record):
        """
        Returns True if the record's compound name contains at least one
        TMS keyword (case-insensitive).
        """
        name = record.get("NAME", "").lower()
        return any(kw in name for kw in self._TMS_NAME_KEYWORDS)

    def filter_tms(self, records):
        """
        Filters records down to confirmed TMS-derivatized spectra only.

        Both checks must pass (fragment check first, name check second).
        Anything failing either check is discarded — no tbdms or other buckets.
        Compounds that would qualify as both TMS and TBDMS are kept (TMS wins).
        """
        tms      = []
        n_no_frag = 0
        n_no_name = 0

        for record in records:

            # gate 1: must have >= 2 canonical TMS fragment ions
            if not self._passes_fragment_check(record):
                n_no_frag += 1
                continue

            # gate 2: name must contain a TMS keyword
            if not self._passes_name_check(record):
                n_no_name += 1
                continue

            tms.append(record)

        print(f"\nInput spectra:        {len(records)}")
        print(f"Failed fragment gate: {n_no_frag}")
        print(f"Failed name gate:     {n_no_name}")
        print(f"TMS confirmed:        {len(tms)}\n")

        return tms

    # -----------------------------------------------------------------------
    # HDF5 storage
    # -----------------------------------------------------------------------

    def write_to_hdf5(self, records, output_file):
        """
        Writes m/z and intensity arrays to a gzip-compressed HDF5 file.

        Each spectrum is stored as an (N, 2) int32 array under the dataset key
        'spectra_{i}' where i is the record's position in records.

        Mutates each record dict in-place to add three keys that
        populate_refDB() uses to cross-reference metadata with spectral data:
            'h5_ID'   — dataset key within the HDF5 file (e.g. 'spectra_42')
            'h5_file' — absolute path to the HDF5 file as a string
            'deriv'   — derivatizing agent label derived from the filename stem
        """
        output_file = Path(output_file)
        n_written = n_skipped = 0

        with h5py.File(output_file, "w") as f:
            for i, record in enumerate(records):
                if not record.get("mzs"):
                    print(f"  [warn] skipping record {i} '{record.get('NAME', '?')}' — no m/z data")
                    n_skipped += 1
                    continue

                mzs         = np.array(record["mzs"],         dtype=np.int32)
                intensities = np.array(record["intensities"], dtype=np.int32)
                arr         = np.column_stack((mzs, intensities))
                sp_id       = f"spectra_{i}"

                f.create_dataset(
                    sp_id,
                    data             = arr,
                    compression      = "gzip",
                    compression_opts = 4,
                )

                record["h5_ID"]   = sp_id
                record["h5_file"] = str(output_file)
                record["deriv"]   = output_file.stem
                n_written += 1

        print(f"Wrote {n_written} spectra to {output_file}")
        if n_skipped:
            print(f"  ({n_skipped} skipped — missing m/z data)")

    # -----------------------------------------------------------------------
    # SQLite helpers
    # -----------------------------------------------------------------------

    def build_DB(self, path):
        """
        Creates (or silently reuses if already present) a SQLite database at
        path using the CREATE TABLE statement in metadata.sql.
        """
        path     = Path(path)
        sql_file = self.data_dir / "metadata.sql"

        with open(sql_file, "r") as f:
            sql_script = f.read()

        conn, cursor = self.sqlite_connect(path)
        cursor.executescript(sql_script)
        conn.commit()
        conn.close()

    def populate_refDB(self, records, path):
        """
        Inserts records into the metadata table of the SQLite database at path.
        Call this after write_to_hdf5() so that h5_ID, h5_file, and deriv are
        already present on each record dict.

        Missing optional fields are inserted as NULL rather than raising an
        error and skipping the row.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Database not found: {path}")

        # normalized record key  →  SQL column name
        # single source of truth — update here when the schema changes
        column_map = {
            "NAME":            "spName",
            "RETENTION INDEX": "retentionIndex",
            "INCHIKEY":        "INCHIKEY",
            "FORMULA":         "formula",
            "MW":              "mw",
            "EXACT MASS":      "exactMass",
            "CASNO":           "casNO",
            "COMMENT":         "comment",
            "NUM PEAKS":       "numPeaks",
            "h5_ID":           "h5ID",
            "h5_file":         "h5file",
            "deriv":           "deriv",
        }

        columns      = list(column_map.values())
        ordered_keys = list(column_map.keys())
        placeholders = ", ".join(["?"] * len(columns))
        sql          = (
            f"INSERT INTO metadata ({', '.join(columns)}) "
            f"VALUES ({placeholders})"
        )

        conn, cursor = self.sqlite_connect(path)
        n_ok = n_err = 0

        for record in records:
            try:
                values = [record.get(key) for key in ordered_keys]
                cursor.execute(sql, values)
                n_ok += 1
            except Exception as e:
                print(f"  [warn] insert failed for '{record.get('NAME', '?')}': {e}")
                n_err += 1

        conn.commit()
        conn.close()
        print(f"Inserted {n_ok} records into {path}  ({n_err} errors)")

    def sqlite_connect(self, path):
        """Returns (conn, cursor) for the SQLite database at path."""
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()
        return conn, cursor
