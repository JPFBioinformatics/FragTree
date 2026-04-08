from pathlib import Path
import h5py
import numpy as np
import sqlite3

class ParseMOL:

    def __init__(self, db_dir):
        """
        Sets up directory paths and ensures the TMS output directory exists.
        Params:
            db_dir      name of the subdirectory to create under
                        <project_root>/databases/reference/ for TMS output
                        (e.g. 'tms')
        """
        self.src_dir   = Path(__file__).resolve().parent
        self.proj_root = self.src_dir.parent
        self.data_dir  = self.proj_root / "databases"
        self.ref_dir   = self.data_dir / "reference"

        self.tms_dir = self.ref_dir / db_dir

    def 