"""
Entry point script for the mol object build pipeline.
Run this from the scripts/ directory:

    python run_build_mol_db.py

Config is loaded from config.yaml at the project root.  The following keys
must be set (run config.py GUI or edit config.yaml manually):

    mol_dir      path to the root directory containing .mol file subdirectories
    db_path      path to the combined SQLite database
    mol_h5_path  path to write the mol_objects.h5 HDF5 file
    reports_dir  path to write the summary PDF report
"""
# region Imports

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config_loader import ConfigLoader
from parse_MOL import ParseMOL

# endregion

def main():
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    cfg = ConfigLoader(config_path)
    db  = ParseMOL(cfg)
    db.run()

if __name__ == "__main__":
    main()
