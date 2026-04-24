"""

Script to generate a network of spectra based on cosine similarity to use for
contrast learning supervision to train embedding of molecular structures

"""

# region Imports

import logging, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from spectra_network import SpectraNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# endregion

def main():

    root = Path(__file__).resolve().parent.parent
    ref = root / "databases" / "reference" / "tms"

    metadata = ref / "metadata.db"
    spectra = ref / "spectra.h5"
    structure = ref / "structure.h5"

    net = SpectraNetwork("TMS", metadata, spectra, structure, rep_threshold=0.75)
    net.generate_network()

if __name__ == "__main__":
    main()
    