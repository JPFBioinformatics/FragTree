"""

Builds and populates the TMS reference database from NIST .msp files.

"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from parse_NIST import ParseNIST
from config_loader import ConfigLoader


def main():

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    cfg     = ConfigLoader(config_path)
    msp_dir = cfg.get_path("raw_data","msp_dir", must_exist=True)

    # ------------------------------------------------------------------
    # 2. Initialize parser and create the TMS database
    # ------------------------------------------------------------------
    nist = ParseNIST(db_dir="tms")

    print("Building database...")
    nist.build_DB(nist.tms_dir / "spectra_metadata.db")
    print("Database ready.\n")

    # ------------------------------------------------------------------
    # 3. Parse all .msp files
    # ------------------------------------------------------------------
    print(f"Parsing .msp files from: {msp_dir}")
    all_records = nist.combine_to_list(msp_dir, file_type="msp")

    # ------------------------------------------------------------------
    # 4. Filter to confirmed TMS spectra only
    #    fragment check (>= 2 of m/z 73, 147, 149) then name check
    #    ('tms' or 'trimethylsilyl' in name, case-insensitive)
    # ------------------------------------------------------------------
    tms = nist.filter_tms(all_records)

    # sanity check — every kept entry should have m/z data
    missing = [r.get("NAME", "?") for r in tms if not r.get("mzs")]
    if missing:
        print(f"[warn] {len(missing)} TMS entries missing m/z data:")
        for name in missing[:5]:
            print(f"  - {name}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    else:
        print(f"All {len(tms)} TMS entries have m/z data.\n")

    # ------------------------------------------------------------------
    # 5. Write spectral data to HDF5
    #    write_to_hdf5 stamps h5_ID, h5_file, and deriv onto each record
    #    dict in-place so populate_refDB can cross-reference them
    # ------------------------------------------------------------------
    print("Writing HDF5 file...")
    nist.write_to_hdf5(tms, nist.tms_dir / "spectra.h5")

    # ------------------------------------------------------------------
    # 6. Populate metadata database
    # ------------------------------------------------------------------
    print("\nPopulating database...")
    nist.populate_refDB(tms, nist.tms_dir / "spectra_metadata.db")

    print("\nDone.")


if __name__ == "__main__":
    main()
