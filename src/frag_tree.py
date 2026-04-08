"""
Fragmentation tree class that is used to model all possible molecular fragments created by 70 eV EI fragmentation in GCMS
Uses a bipartite graph structure where we have fragment nodes connected to each other through event nodes.  Currently only
bond breaking events are modeled, but this structure allows for other events (charge migration, rearrangements...) to be 
added in the future
"""

class FragTree:

    def __init__(self, molecule):
        """
        Initializes fragmentation tree for a specified molecule
        Params:
            molecule                full molecular structure for tree (root)
        """

        self.root_mol = molecule

        