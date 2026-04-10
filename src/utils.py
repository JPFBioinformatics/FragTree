"""

Shared utility functions used across the FragTree pipeline.

"""

import re
from collections import defaultdict


def to_hill_notation(formula: str) -> str:
    """
    Converts a molecular formula string to Hill notation.

    Hill notation orders elements as:
        1. Carbon first
        2. Hydrogen second
        3. All remaining elements alphabetically

    Handles formulas with or without explicit counts on single atoms
    (e.g. 'CH4', 'C2H6O', 'H2O', 'C10H12BrN2Si').

    Parameters
    ----------
    formula         molecular formula string in any element ordering

    Returns
    -------
    hill_formula    formula string in Hill notation

    Examples
    --------
    to_hill_notation("H2OC")       -> "CH2O"
    to_hill_notation("SiC3H9")     -> "C3H9Si"
    to_hill_notation("BrC10H12N2") -> "C10H12BrN2"
    to_hill_notation("H2O")        -> "H2O"   (no carbon — alphabetical)
    """
    
    # parse element/count pairs — matches e.g. "C", "C3", "Si", "Br2"
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)

    counts: dict = defaultdict(int)
    for element, count in tokens:
        if element:
            counts[element] += int(count) if count else 1

    if not counts:
        return formula

    parts = []

    # carbon first if present
    if "C" in counts:
        n = counts.pop("C")
        parts.append("C" if n == 1 else f"C{n}")

        # hydrogen second if present (only when carbon is present)
        if "H" in counts:
            n = counts.pop("H")
            parts.append("H" if n == 1 else f"H{n}")

    # remaining elements alphabetically
    for element in sorted(counts.keys()):
        n = counts[element]
        parts.append(element if n == 1 else f"{element}{n}")

    # if no carbon was present, hydrogen goes in alphabetical order
    # (already handled since H was not popped from counts)

    return "".join(parts)