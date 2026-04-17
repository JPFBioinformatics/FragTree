"""

Bipartite fragmentation tree for modeling all possible EI-MS fragments of a
TMS-derivatized metabolite.

Graph structure
---------------
The tree is a directed acyclic bipartite graph alternating between two node
types:

    FragmentNode    represents a specific ion structure (molecular ion or
                    any fragment produced by bond cleavage).  Every fragment
                    is treated as potentially charged — either product of a
                    bond break can carry the charge in 70 eV EI.  Stores the
                    RDKit Mol object, mass, and radical atom index.

    BondBreakNode   represents a single bond cleavage event connecting a
                    parent FragmentNode to two child FragmentNodes (child_a
                    and child_b, both potentially charged).  Stores the bond
                    index, basic bond descriptors, and a charge_split_prior
                    (initialised to 0.5) that the GNN learns — it is the
                    probability that child_a carries the charge.

Edges run:
    FragmentNode  ->  BondBreakNode   (one edge per cleavable bond)
    BondBreakNode ->  FragmentNode    (two edges: child_a, child_b)

Fragment generation
-------------------
Tree construction is eager — the full tree is built at instantiation time by
recursively enumerating all cleavable bonds in each fragment until no fragment
above the minimum mass threshold can be further cleaved.

Hydrogen atoms are excluded from bond enumeration (bonds to H are never
cleaved) but are included when computing fragment masses, since a radical
fragment can either retain its H (mass M) or lose it to form a closed-shell
ion (mass M-1).

Embeddings
----------
No embeddings are stored on the nodes.  The GNN is responsible for reading
raw node attributes and constructing feature vectors.  FragmentNode and
BondBreakNode expose all chemically relevant attributes as plain Python
attributes so the GNN code can access them directly.
"""

# region Imports

from __future__ import annotations

import uuid
from typing import Optional

import numpy as np
from rdkit import Chem                                        # type: ignore
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem # type: ignore
from rdkit.Chem.rdchem import BondType                        # type: ignore

# endregion

# region Electronetagivity Table

# Electronegativity table (Pauling scale, most common elements in metabolites)

_ELECTRONEGATIVITY = {
    1:  2.20,   # H
    5:  2.04,   # B
    6:  2.55,   # C
    7:  3.04,   # N
    8:  3.44,   # O
    9:  3.98,   # F
    14: 1.90,   # Si
    15: 2.19,   # P
    16: 2.58,   # S
    17: 3.16,   # Cl
    35: 2.96,   # Br
    53: 2.66,   # I
}

_DEFAULT_EN = 2.20  # fallback for elements not in the table

def _get_en(atomic_num: int) -> float:
    """Returns the Pauling electronegativity for an atomic number."""
    return _ELECTRONEGATIVITY.get(atomic_num, _DEFAULT_EN)

# endregion

# region Node classes

class FragmentNode:
    """
    Represents a specific ion structure in the fragmentation tree.

    This is either the intact molecular ion (root node) or a charged fragment
    produced by bond cleavage.  Neutral fragments are also represented as
    FragmentNode instances but are flagged as unmeasured since the mass
    spectrometer only detects charged species.

    Attributes
    ----------
    node_id         unique string identifier for this node
    mol             RDKit Mol object for this fragment
    mass            exact monoisotopic mass of the fragment (Da)
    mass_minus_h    mass - 1.00794, the M-H loss variant (radical -> closed shell)
    is_root         True if this is the intact molecular ion
    radical_atom    index of the atom bearing the free radical (-1 if none)
    depth           depth in the tree (root = 0)
    parent_bond_break_id    node_id of the BondBreakNode that produced this
                            fragment, or None for the root
    children        list of BondBreakNode objects where this fragment is parent
    """

    def __init__(
        self,
        mol:                  Chem.Mol,
        atom_map_to_root:     list[int],
        is_root:              bool = False,
        radical_atom:         int  = -1,
        depth:                int  = 0,
        parent_bond_break_id: Optional[str] = None,
    ):
        self.node_id              = str(uuid.uuid4())
        self.mol                  = mol
        self.atom_map_to_root     = atom_map_to_root
        self.is_root              = is_root
        self.radical_atom         = radical_atom
        self.depth                = depth
        self.parent_bond_break_id = parent_bond_break_id
        self.children: list[BondBreakNode] = []

        # compute masses once at construction
        self.mass         = Descriptors.ExactMolWt(mol)
        self.mass_minus_h = self.mass - 1.00794

    def __repr__(self) -> str:
        smiles = Chem.MolToSmiles(self.mol) if self.mol else "?"
        return (
            f"FragmentNode(id={self.node_id[:8]}, "
            f"mass={self.mass:.2f}, "
            f"smiles={smiles[:30]})"
        )

class BondBreakNode:
    """
    Represents a single bond cleavage event in the fragmentation tree.

    Sits between a parent FragmentNode and one or two child FragmentNodes.
    For acyclic bonds, two children are produced (child_a and child_b).  For
    ring bonds, a single child is produced (child_a only, child_b_id = None)
    representing the ring-opened species.  Stores all bond-level descriptors
    needed by the GNN to construct the bond embedding, but does NOT construct
    the embedding itself — that is the GNN's responsibility.

    For two-child nodes, charge_split_prior is the prior probability that
    child_a carries the charge; the GNN learns this from training data.
    For single-child (ring-open) nodes, charge_split_prior is fixed at 1.0
    since the single fragment trivially carries the charge.

    Attributes
    ----------
    node_id             unique string identifier
    parent_id           node_id of the parent FragmentNode
    bond_idx            index of the cleaved bond in the parent fragment mol
    atom_i_idx          index of the first atom of the cleaved bond
    atom_j_idx          index of the second atom of the cleaved bond

    -- bond descriptors (used by GNN to build local embedding component) --
    bond_order          numeric bond order (1.0, 1.5, 2.0, 3.0)
    is_conjugated       whether the bond is part of a conjugated system
    is_in_ring          whether the bond is part of a ring
    en_diff             absolute electronegativity difference between the atoms
    bond_length_est     estimated bond length in angstroms from covalent radii
                        (or from conformer if 3D coordinates are available)

    -- atom descriptors for atom_i --
    i_atomic_num        atomic number
    i_formal_charge     formal charge
    i_degree            number of bonded neighbors
    i_valence_electrons number of valence electrons
    i_hybridization     hybridization state as a string (SP, SP2, SP3, OTHER)
    i_is_aromatic       aromaticity flag
    i_electronegativity Pauling electronegativity
    i_covalent_radius   covalent radius in angstroms
    i_in_ring           ring membership flag
    i_num_hs            number of attached hydrogens

    -- atom descriptors for atom_j (same set as atom_i) --
    j_atomic_num ... j_num_hs

    -- charge split prior --
    charge_split_prior  prior probability that child_a carries the charge
                        (initialised to 0.5).  The GNN learns this value
                        during training; 1 - charge_split_prior is the
                        probability that child_b carries the charge.

    -- children --
    child_a_id          node_id of the heavier child FragmentNode
    child_b_id          node_id of the lighter child FragmentNode, or None
                        for ring-opening events where only one fragment is
                        produced
    """

    # approximate covalent radii in angstroms
    _COVALENT_RADIUS = {
        1:  0.31,   # H
        6:  0.76,   # C
        7:  0.71,   # N
        8:  0.66,   # O
        9:  0.57,   # F
        14: 1.11,   # Si
        15: 1.07,   # P
        16: 1.05,   # S
        17: 1.02,   # Cl
        35: 1.20,   # Br
        53: 1.39,   # I
    }
    _DEFAULT_RADIUS = 0.77

    def __init__(
        self,
        parent_fragment: FragmentNode,
        bond_idx:        int,
    ):
        self.node_id   = str(uuid.uuid4())
        self.parent_id = parent_fragment.node_id

        mol  = parent_fragment.mol
        bond = mol.GetBondWithIdx(bond_idx)

        self.bond_idx    = bond_idx
        self.atom_i_idx  = bond.GetBeginAtomIdx()
        self.atom_j_idx  = bond.GetEndAtomIdx()

        atom_i = mol.GetAtomWithIdx(self.atom_i_idx)
        atom_j = mol.GetAtomWithIdx(self.atom_j_idx)

        # bond descriptors
        self.bond_order    = self._bond_order_numeric(bond.GetBondType())
        self.is_conjugated = bond.GetIsConjugated()
        self.is_in_ring    = bond.IsInRing()
        self.en_diff       = abs(
            _get_en(atom_i.GetAtomicNum()) - _get_en(atom_j.GetAtomicNum())
        )
        self.bond_length_est = self._estimate_bond_length(
            atom_i.GetAtomicNum(), atom_j.GetAtomicNum()
        )

        # atom_i descriptors
        self.i_atomic_num        = atom_i.GetAtomicNum()
        self.i_formal_charge     = atom_i.GetFormalCharge()
        self.i_degree            = atom_i.GetDegree()
        self.i_valence_electrons = self._valence_electrons(atom_i.GetAtomicNum())
        self.i_hybridization     = self._hybridization_str(atom_i.GetHybridization())
        self.i_is_aromatic       = atom_i.GetIsAromatic()
        self.i_electronegativity = _get_en(atom_i.GetAtomicNum())
        self.i_covalent_radius   = self._COVALENT_RADIUS.get(
            atom_i.GetAtomicNum(), self._DEFAULT_RADIUS
        )
        self.i_in_ring           = atom_i.IsInRing()
        self.i_num_hs            = atom_i.GetTotalNumHs()

        # atom_j descriptors
        self.j_atomic_num        = atom_j.GetAtomicNum()
        self.j_formal_charge     = atom_j.GetFormalCharge()
        self.j_degree            = atom_j.GetDegree()
        self.j_valence_electrons = self._valence_electrons(atom_j.GetAtomicNum())
        self.j_hybridization     = self._hybridization_str(atom_j.GetHybridization())
        self.j_is_aromatic       = atom_j.GetIsAromatic()
        self.j_electronegativity = _get_en(atom_j.GetAtomicNum())
        self.j_covalent_radius   = self._COVALENT_RADIUS.get(
            atom_j.GetAtomicNum(), self._DEFAULT_RADIUS
        )
        self.j_in_ring           = atom_j.IsInRing()
        self.j_num_hs            = atom_j.GetTotalNumHs()

        # charge split prior — GNN learns this; 0.5 means equal prior
        # for child_a vs child_b carrying the charge
        self.charge_split_prior: float = 0.5

        # child node IDs populated by FragTree after child nodes are created
        self.child_a_id: Optional[str] = None   # heavier fragment
        self.child_b_id: Optional[str] = None   # lighter fragment

    def __repr__(self) -> str:
        return (
            f"BondBreakNode(id={self.node_id[:8]}, "
            f"bond={self.atom_i_idx}-{self.atom_j_idx}, "
            f"order={self.bond_order})"
        )

# endregion

    # region Helpers

    @staticmethod
    def _bond_order_numeric(bond_type: BondType) -> float:
        """Converts an RDKit BondType to a numeric bond order."""
        mapping = {
            BondType.SINGLE:    1.0,
            BondType.DOUBLE:    2.0,
            BondType.TRIPLE:    3.0,
            BondType.AROMATIC:  1.5,
        }
        return mapping.get(bond_type, 1.0)

    @staticmethod
    def _hybridization_str(hybridization) -> str:
        """Converts an RDKit hybridization constant to a readable string."""
        from rdkit.Chem.rdchem import HybridizationType
        mapping = {
            HybridizationType.SP:    "SP",
            HybridizationType.SP2:   "SP2",
            HybridizationType.SP3:   "SP3",
            HybridizationType.SP3D:  "SP3D",
            HybridizationType.SP3D2: "SP3D2",
        }
        return mapping.get(hybridization, "OTHER")

    @staticmethod
    def _valence_electrons(atomic_num: int) -> int:
        """
        Returns the number of valence electrons for common elements.
        Falls back to 4 for unknown elements.
        """
        mapping = {
            1: 1, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7,
            14: 4, 15: 5, 16: 6, 17: 7,
            35: 7, 53: 7,
        }
        return mapping.get(atomic_num, 4)

    def _estimate_bond_length(
        self,
        atomic_i:  int,
        atomic_j:  int,
    ) -> float:
        """
        Estimates bond length in angstroms using bond-order-adjusted covalent
        radii.

        The 2D coordinates in NIST .mol files are layout coordinates, not
        true geometric distances — they are chosen for visual clarity rather
        than chemical accuracy and are unreliable for sp3 centres, TMS groups,
        and anything with significant out-of-plane geometry.

        Instead we use the sum of covalent radii corrected for bond order.
        Each additional bond order shortens the bond by approximately 0.15 A
        (the empirical Pauling bond-order correction), giving chemically
        meaningful estimates without requiring 3D coordinate generation:

            single:   r_i + r_j               (no correction)
            aromatic: r_i + r_j - 0.075 A     (halfway between single/double)
            double:   r_i + r_j - 0.15 A
            triple:   r_i + r_j - 0.30 A

        Returns the estimated length in angstroms.
        """
        r_i = self._COVALENT_RADIUS.get(atomic_i, self._DEFAULT_RADIUS)
        r_j = self._COVALENT_RADIUS.get(atomic_j, self._DEFAULT_RADIUS)
        base = r_i + r_j

        # bond-order correction — each additional order shortens by ~0.15 A
        correction = {
            1.0: 0.00,   # single
            1.5: 0.075,  # aromatic
            2.0: 0.15,   # double
            3.0: 0.30,   # triple
        }
        return base - correction.get(self.bond_order, 0.0)

    # endregion

class FragmentationTree:
    """
    Bipartite fragmentation tree for a single TMS-derivatized molecule.

    Builds an eager directed acyclic graph of all chemically feasible
    fragmentation pathways from the molecular ion down to fragments at or
    above min_mass_da.

    Parameters
    ----------
    mol             RDKit Mol object for the intact molecule (from mol_objects.h5)
    min_mass_da     minimum fragment mass in Da to include (default 100.0)

    Attributes
    ----------
    root            FragmentNode for the intact molecular ion
    fragment_nodes  dict mapping node_id -> FragmentNode
    bond_break_nodes  dict mapping node_id -> BondBreakNode
    n_fragments     total number of FragmentNode instances in the tree
    n_bond_breaks   total number of BondBreakNode instances in the tree
    """

    def __init__(self, mol: Chem.Mol, min_mass_da: float = 100.0, max_depth: int = 3):
        self.min_mass_da = min_mass_da
        self.max_depth   = max_depth

        self.fragment_nodes:   dict[str, FragmentNode]   = {}
        self.bond_break_nodes: dict[str, BondBreakNode]  = {}

        # build root node from the intact molecular ion
        self.root = FragmentNode(
            mol,
            atom_map_to_root=list(range(mol.GetNumAtoms())),
            is_root=True,
            depth=0,
        )
        self._register_fragment(self.root)

        # eagerly build the full tree
        self._expand(self.root)

    # region Public interface

    @property
    def n_fragments(self) -> int:
        return len(self.fragment_nodes)

    @property
    def n_bond_breaks(self) -> int:
        return len(self.bond_break_nodes)

    def get_fragment(self, node_id: str) -> Optional[FragmentNode]:
        """Returns the FragmentNode with the given node_id, or None."""
        return self.fragment_nodes.get(node_id)

    def get_bond_break(self, node_id: str) -> Optional[BondBreakNode]:
        """Returns the BondBreakNode with the given node_id, or None."""
        return self.bond_break_nodes.get(node_id)

    def get_all_fragments(self) -> list[FragmentNode]:
        """
        Returns all fragment nodes sorted by mass descending.  Every fragment
        is a potentially observable ion — the GNN's charge_split_prior on each
        BondBreakNode determines the probability that a given fragment carries
        the charge and appears in the spectrum.
        """
        return sorted(
            self.fragment_nodes.values(),
            key=lambda n: n.mass,
            reverse=True,
        )

    def get_paths_to_fragment(self, target_id: str) -> list[list]:
        """
        Returns all directed paths from the root to the fragment with
        target_id.  Each path is a list alternating between FragmentNode
        and BondBreakNode instances.

        Used for computing fragment probability as the sum over paths of
        the product of edge weights along each path.
        """
        paths = []
        self._dfs_paths(self.root, target_id, [self.root], paths)
        return paths

    def summary(self) -> str:
        """Returns a short human-readable summary of the tree."""
        return (
            f"FragmentationTree\n"
            f"  root mass:       {self.root.mass:.2f} Da\n"
            f"  min mass cutoff: {self.min_mass_da:.1f} Da\n"
            f"  fragment nodes:  {self.n_fragments}\n"
            f"  bond break nodes:{self.n_bond_breaks}\n"
        )

    # endregion

    # region Tree construction

    def _register_fragment(self, node: FragmentNode):
        """Adds a FragmentNode to the lookup dict."""
        self.fragment_nodes[node.node_id] = node

    def _register_bond_break(self, node: BondBreakNode):
        """Adds a BondBreakNode to the lookup dict."""
        self.bond_break_nodes[node.node_id] = node

    def _expand(self, parent: FragmentNode):
        """
        Recursively enumerates all cleavable bonds in parent.mol, creates
        a BondBreakNode for each, generates the two child FragmentNodes,
        and recurses into charged children that are above the mass threshold.

        Hydrogen bonds are skipped — H atoms contribute to mass but are not
        fragmentation sites.

        Expansion stops when:
          - parent.depth >= max_depth  (prevents runaway recursion)
          - the fragment's canonical SMILES has already been expanded from
            another path (prevents combinatorial re-expansion of shared
            substructures)
        """
        mol = parent.mol
        if mol is None:
            return

        if parent.depth >= self.max_depth:
            return

        for bond in mol.GetBonds():
            atom_i = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            atom_j = mol.GetAtomWithIdx(bond.GetEndAtomIdx())

            # skip bonds to hydrogen
            if atom_i.GetAtomicNum() == 1 or atom_j.GetAtomicNum() == 1:
                continue

            # create the bond break event node
            bb_node = BondBreakNode(parent, bond.GetIdx())
            self._register_bond_break(bb_node)
            parent.children.append(bb_node)

            # generate child fragments by cleaving this bond
            child_a, child_b = self._cleave_bond(
                mol,
                bond.GetIdx(),
                parent.atom_map_to_root,
                parent.depth + 1,
                bb_node.node_id,
            )

            if child_a is None:
                # hard RDKit failure — discard this bond break event
                parent.children.remove(bb_node)
                del self.bond_break_nodes[bb_node.node_id]
                continue

            # link child_a (always present)
            bb_node.child_a_id = child_a.node_id
            self._register_fragment(child_a)

            if child_b is not None:
                # two-fragment cleavage (acyclic bond) — GNN learns the charge
                # split between child_a and child_b
                bb_node.child_b_id = child_b.node_id
                self._register_fragment(child_b)
                if child_b.mass >= self.min_mass_da:
                    self._expand(child_b)
            else:
                # single-fragment cleavage (ring-opening) — child_a trivially
                # carries the charge, so charge_split_prior is fixed at 1.0
                bb_node.charge_split_prior = 1.0

            if child_a.mass >= self.min_mass_da:
                self._expand(child_a)

    def _cleave_bond(
        self,
        mol:                Chem.Mol,
        bond_idx:           int,
        parent_atom_map:    list[int],
        depth:              int,
        parent_bb_id:       str,
        
    ) -> tuple[Optional[FragmentNode], Optional[FragmentNode]]:
        """
        Cleaves a bond in mol and returns (child_a, child_b).

        Uses RDKit's FragmentOnBonds to break the bond and splits the
        resulting disconnected mol into two separate Mol objects.  Both
        children are treated as potentially charged — the GNN learns which
        side preferentially carries the charge via charge_split_prior on the
        BondBreakNode.  child_a is the heavier fragment, child_b the lighter,
        as a stable ordering convention (not a charge assignment).

        For ring bonds, cleavage opens the ring without disconnecting the
        molecule.  A single child_a is returned representing the ring-opened
        species; child_b is None.  The ring-opened species has the same mass
        as the parent but different connectivity and can fragment further.

        Returns (None, None) only on a hard RDKit failure.
        """
        try:
            # FragmentOnBonds breaks the bond; addDummies=False leaves radical
            # atoms at the cleavage site rather than inserting dummy atoms
            frag_mol = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=False)
            if frag_mol is None:
                return None, None

            # frag_atom_tuples[i] are the atom indices (in frag_mol/mol space,
            # preserved by FragmentOnBonds) for fragment i.  Compose with
            # parent_atom_map to get root-molecule atom indices.
            frag_atom_tuples = Chem.GetMolFrags(frag_mol)
            frags            = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)

            if len(frags) == 1:
                # ring bond — same atoms as parent, just different connectivity
                map_a   = [parent_atom_map[i] for i in frag_atom_tuples[0]]
                child_a = FragmentNode(
                    frags[0],
                    atom_map_to_root=map_a,
                    radical_atom=-1,
                    depth=depth,
                    parent_bond_break_id=parent_bb_id,
                )
                return child_a, None

            if len(frags) != 2:
                return None, None

            frag_a,  frag_b  = frags[0],            frags[1]
            tuple_a, tuple_b = frag_atom_tuples[0], frag_atom_tuples[1]
            mass_a           = Descriptors.ExactMolWt(frag_a)
            mass_b           = Descriptors.ExactMolWt(frag_b)

            # child_a = heavier, child_b = lighter — swap both mol and tuple
            if mass_a < mass_b:
                frag_a,  frag_b  = frag_b,  frag_a
                tuple_a, tuple_b = tuple_b, tuple_a

            map_a = [parent_atom_map[i] for i in tuple_a]
            map_b = [parent_atom_map[i] for i in tuple_b]

            child_a = FragmentNode(
                frag_a,
                atom_map_to_root=map_a,
                radical_atom=-1,
                depth=depth,
                parent_bond_break_id=parent_bb_id,
            )
            child_b = FragmentNode(
                frag_b,
                atom_map_to_root=map_b,
                radical_atom=-1,
                depth=depth,
                parent_bond_break_id=parent_bb_id,
            )

            return child_a, child_b

        except Exception:
            return None, None
        
    # endregion

    # region Path enumeration

    def _dfs_paths(
        self,
        current:   FragmentNode,
        target_id: str,
        path:      list,
        results:   list,
    ):
        """
        Depth-first search collecting all paths from current to target_id.
        Path entries alternate: FragmentNode, BondBreakNode, FragmentNode, ...
        """
        if current.node_id == target_id:
            results.append(path.copy())
            return

        for bb in current.children:
            path.append(bb)

            # follow both children — either can carry the charge
            for child_id in (bb.child_a_id, bb.child_b_id):
                child = self.fragment_nodes.get(child_id or "")
                if child:
                    path.append(child)
                    self._dfs_paths(child, target_id, path, results)
                    path.pop()

            path.pop()

    # endregion
    