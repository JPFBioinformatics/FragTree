"""

Entry point for the clustering pipeline.  Edit the three sections below to
control which stage runs and with what parameters.

Stages
------
1. Coarse grid  — always run this first.  Inspect coarse_grid_report.pdf
                  and identify a promising min_cluster_size region.

2. Fine grid    — set RUN_FINE = True and fill in the range you identified
                  from the coarse report.  Inspect fine_grid_report.pdf and
                  pick your best (min_cluster_size, min_samples) pair.

3. Final        — set RUN_FINAL = True and fill in your chosen parameters.
                  Produces final_cluster_report.pdf and cluster_assignments.json.

All output goes to the reports_dir specified in config.yaml.
"""

# region Imports

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config_loader import ConfigLoader
from cluster_analysis import ClusterAnalysis

# endregion

# region Stage and Parameter Control

RUN_COARSE = False

RUN_FINE   = False
FINE_CS_MIN  = 71     # min_cluster_size lower bound
FINE_CS_MAX  = 95     # min_cluster_size upper bound
FINE_CS_STEP = 2      # min_cluster_size step size
FINE_MS_MIN  = 51     # min_samples lower bound
FINE_MS_MAX  = 65     # min_samples upper bound  (None = same as cs_max)

RUN_FINAL       = True
FINAL_CS        = 75  # chosen min_cluster_size
FINAL_MS        = 53  # chosen min_samples

# endregion

def main():
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    cfg = ConfigLoader(config_path)
    ca  = ClusterAnalysis(cfg)

    if RUN_COARSE:
        ca.run_coarse_grid()

    if RUN_FINE:
        ca.run_fine_grid(
            cs_min  = FINE_CS_MIN,
            cs_max  = FINE_CS_MAX,
            cs_step = FINE_CS_STEP,
            ms_min  = FINE_MS_MIN,
            ms_max  = FINE_MS_MAX,
        )

    if RUN_FINAL:
        ca.run_final(
            min_cluster_size = FINAL_CS,
            min_samples      = FINAL_MS,
        )


if __name__ == "__main__":
    main()
