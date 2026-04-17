"""
MPNN for predicting EI-MS fragmentation patterns from molecular structure.

Forward pass summary
--------------------
    mol graph
        │
    GCN message passing  (n_layers rounds)
        │ per-atom embeddings h
    bond rep = cat(h_i, h_j, edge_attr)  for each non-H bond
        │
    bond scoring MLP  →  softmax  →  break probability p_k per bond
        │
    scatter into m/z grid  (charge_split_prior = 0.5 fixed)
        │
    predicted spectrum  (MZ_GRID_SIZE,)

Loss: 1 - cosine_similarity(predicted_spectrum, observed_spectrum)

Only first-generation fragmentations (bonds in the root fragment) are
modelled.  Multi-step pathways are left for future work.
"""

# region Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondType

from load_training_data import MZ_MIN, MZ_GRID_SIZE
from frag_tree import FragmentationTree

# endregion

# region Feature constants

_ATOMIC_NUMS    = [1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]   # H C N O F Si P S Cl Br I
_HYBRIDIZATIONS = ["SP", "SP2", "SP3"]   # "OTHER" is the implicit fallback bin added by _one_hot
_BOND_ORDERS    = [1.0, 1.5, 2.0, 3.0]

_EN = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44,  9: 3.98,
       14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66}
_CR = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66,  9: 0.57,
       14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39}
_DEFAULT_EN = 2.20
_DEFAULT_CR = 0.77

# ATOM_DIM: one-hot atomic_num(12) + formal_charge(1) + degree(1) + num_hs(1)
#           + is_in_ring(1) + is_aromatic(1) + one-hot hybridization(4) = 21
# BOND_DIM: one-hot bond_order(5) + is_in_ring(1) + is_conjugated(1)
#           + en_diff(1) + bond_length_est(1) = 9
ATOM_DIM = 21
BOND_DIM = 9

# endregion

# region Featurisation

def _one_hot(val, choices: list) -> list:
    """One-hot encodes val; last position is the 'other' bin."""
    vec = [0.0] * (len(choices) + 1)
    try:
        vec[choices.index(val)] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


def featurize_mol(
    mol: Chem.Mol,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts an RDKit Mol to PyG tensors.

    Bonds are added in RDKit bond-index order: bond k occupies edges at
    positions 2k (forward i→j) and 2k+1 (reverse j→i).  This ordering
    lets the MPNN forward pass look up bond k via edge_index[:, 2k].

    Returns
    -------
    x           (N_atoms, ATOM_DIM)  float32
    edge_index  (2, 2*N_bonds)       long
    edge_attr   (2*N_bonds, BOND_DIM) float32
    """
    _hybrid_map = {
        HybridizationType.SP:  "SP",
        HybridizationType.SP2: "SP2",
        HybridizationType.SP3: "SP3",
    }
    _bo_map = {
        BondType.SINGLE:   1.0,
        BondType.DOUBLE:   2.0,
        BondType.TRIPLE:   3.0,
        BondType.AROMATIC: 1.5,
    }
    _bo_correction = {1.0: 0.0, 1.5: 0.075, 2.0: 0.15, 3.0: 0.30}

    atom_feats = []
    for atom in mol.GetAtoms():
        hyb = _hybrid_map.get(atom.GetHybridization(), "OTHER")
        atom_feats.append(
            _one_hot(atom.GetAtomicNum(), _ATOMIC_NUMS)
            + [float(atom.GetFormalCharge())]
            + [float(atom.GetDegree())]
            + [float(atom.GetTotalNumHs())]
            + [float(atom.IsInRing())]
            + [float(atom.GetIsAromatic())]
            + _one_hot(hyb, _HYBRIDIZATIONS)
        )
    x = torch.tensor(atom_feats, dtype=torch.float32)

    rows, cols, bond_feats = [], [], []
    for bond in mol.GetBonds():
        i, j   = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ai, aj = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)
        bo     = _bo_map.get(bond.GetBondType(), 1.0)
        en_diff = abs(
            _EN.get(ai.GetAtomicNum(), _DEFAULT_EN) -
            _EN.get(aj.GetAtomicNum(), _DEFAULT_EN)
        )
        bond_len = (
            _CR.get(ai.GetAtomicNum(), _DEFAULT_CR) +
            _CR.get(aj.GetAtomicNum(), _DEFAULT_CR) -
            _bo_correction.get(bo, 0.0)
        )
        feat = (
            _one_hot(bo, _BOND_ORDERS)
            + [float(bond.IsInRing())]
            + [float(bond.GetIsConjugated())]
            + [en_diff]
            + [bond_len]
        )
        # forward edge (bond k → index 2k), reverse (→ index 2k+1)
        rows       += [i, j]
        cols       += [j, i]
        bond_feats += [feat, feat]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr  = torch.tensor(bond_feats,   dtype=torch.float32)
    return x, edge_index, edge_attr


def _bond_feats_from_bb(bb) -> list[float]:
    """
    Reconstructs the BOND_DIM=9 edge feature vector from a BondBreakNode,
    matching the format produced by featurize_mol.
    """
    return (
        _one_hot(bb.bond_order, _BOND_ORDERS)
        + [float(bb.is_in_ring)]
        + [float(bb.is_conjugated)]
        + [bb.en_diff]
        + [bb.bond_length_est]
    )


def build_bond_targets(tree: FragmentationTree) -> dict:
    """
    Walks the full fragmentation tree in BFS order and returns precomputed
    integer arrays for fully vectorized forward passes.

    All string node IDs are converted to integer indices here so the forward
    pass never touches Python dicts or loops over records — it works entirely
    with tensor operations.

    Returns a dict with:
        atom_i        (N,) int  — root-mol atom index for bond atom i
        atom_j        (N,) int  — root-mol atom index for bond atom j
        edge_feat     (N, BOND_DIM) float  — bond features
        charge_split  (N,) float  — charge split prior per bond break
        group_id      (N,) int  — fragment index of parent (for segment softmax)
        parent_frag   (N,) int  — same as group_id
        child_a       (N,) int  — fragment index of child_a
        child_b       (N,) int  — fragment index of child_b (-1 if ring-open)
        mass_a_bin    (N,) int  — m/z bin for child_a  (-1 if out of range)
        mass_b_bin    (N,) int  — m/z bin for child_b  (-1 if ring-open or OOR)
        depth         (N,) int  — depth of the parent fragment
        root_frag_idx int       — integer index of the root fragment
        n_frags       int       — total number of distinct fragment nodes
        max_depth     int       — tree max_depth (loop bound in forward pass)
    """
    # -- assign integer indices to every fragment node in BFS order ----------
    frag_to_idx: dict[str, int] = {}
    bfs: list = [tree.root]
    seen: set[str] = set()
    while bfs:
        frag = bfs.pop(0)
        if frag.node_id in seen:
            continue
        seen.add(frag.node_id)
        frag_to_idx[frag.node_id] = len(frag_to_idx)
        for bb in frag.children:
            child_a = tree.fragment_nodes.get(bb.child_a_id or "")
            child_b = tree.fragment_nodes.get(bb.child_b_id or "") if bb.child_b_id else None
            if child_a is not None:
                bfs.append(child_a)
            if child_b is not None:
                bfs.append(child_b)

    # -- collect per-bond-break arrays in BFS order --------------------------
    atom_i_list:       list[int]        = []
    atom_j_list:       list[int]        = []
    edge_feat_list:    list[list[float]] = []
    charge_split_list: list[float]      = []
    group_id_list:     list[int]        = []
    child_a_list:      list[int]        = []
    child_b_list:      list[int]        = []
    mass_a_bin_list:   list[int]        = []
    mass_b_bin_list:   list[int]        = []
    depth_list:        list[int]        = []

    queue:   list = [tree.root]
    visited: set[str] = set()
    while queue:
        frag = queue.pop(0)
        if frag.node_id in visited:
            continue
        visited.add(frag.node_id)

        for bb in frag.children:
            child_a = tree.fragment_nodes.get(bb.child_a_id or "")
            child_b = tree.fragment_nodes.get(bb.child_b_id or "") if bb.child_b_id else None
            if child_a is None:
                continue

            bin_a = int(round(child_a.mass)) - MZ_MIN
            bin_b = (int(round(child_b.mass)) - MZ_MIN) if child_b is not None else -1

            atom_i_list.append(frag.atom_map_to_root[bb.atom_i_idx])
            atom_j_list.append(frag.atom_map_to_root[bb.atom_j_idx])
            edge_feat_list.append(_bond_feats_from_bb(bb))
            charge_split_list.append(bb.charge_split_prior)
            group_id_list.append(frag_to_idx[frag.node_id])
            child_a_list.append(frag_to_idx[child_a.node_id])
            child_b_list.append(frag_to_idx[child_b.node_id] if child_b is not None else -1)
            mass_a_bin_list.append(bin_a if 0 <= bin_a < MZ_GRID_SIZE else -1)
            mass_b_bin_list.append(bin_b if 0 <= bin_b < MZ_GRID_SIZE else -1)
            depth_list.append(frag.depth)

            if child_a.mass >= tree.min_mass_da:
                queue.append(child_a)
            if child_b is not None and child_b.mass >= tree.min_mass_da:
                queue.append(child_b)

    return {
        "atom_i":        atom_i_list,
        "atom_j":        atom_j_list,
        "edge_feat":     edge_feat_list,
        "charge_split":  charge_split_list,
        "group_id":      group_id_list,
        "parent_frag":   group_id_list,   # same as group_id
        "child_a":       child_a_list,
        "child_b":       child_b_list,
        "mass_a_bin":    mass_a_bin_list,
        "mass_b_bin":    mass_b_bin_list,
        "depth":         depth_list,
        "root_frag_idx": frag_to_idx[tree.root.node_id],
        "n_frags":       len(frag_to_idx),
        "max_depth":     tree.max_depth,
    }

# endregion

# region Model

class FragTreeMPNN(nn.Module):
    """
    Message-passing network for EI-MS spectrum prediction.

    Architecture
    ------------
    1. Linear projection of atom features → hidden_dim.
    2. n_layers rounds of GCNConv message passing with ReLU + dropout.
    3. Bond scoring MLP: cat(h_i, h_j, edge_attr) → scalar score per bond.
    4. Softmax over valid (non-H) bonds → break probability distribution.
    5. Scatter probabilities into a MZ_GRID_SIZE spectrum vector using
       fragment masses from build_bond_targets.  charge_split_prior is
       fixed at 0.5: each child fragment receives half the break probability.

    Parameters
    ----------
    hidden_dim  width of all hidden layers (default 128)
    n_layers    number of GCN message passing rounds (default 3)
    dropout     dropout rate applied after each GCN layer (default 0.1)
    """

    def __init__(
        self,
        hidden_dim: int   = 128,
        n_layers:   int   = 3,
        dropout:    float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(ATOM_DIM, hidden_dim)

        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout)

        # bond representation: cat(h_i, h_j, edge_attr) → [break_score, charge_split_logit]
        self.bond_scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim + BOND_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        x:            torch.Tensor,
        edge_index:   torch.Tensor,
        bond_targets: dict,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x             (N_atoms, ATOM_DIM)    — from featurize_mol
        edge_index    (2, 2*N_bonds)         — from featurize_mol
        bond_targets  dict — from build_bond_targets

        Returns
        -------
        predicted_spectrum  (MZ_GRID_SIZE,) float32
        """
        bt  = bond_targets
        dev = x.device

        if not bt["atom_i"]:
            return torch.zeros(MZ_GRID_SIZE, dtype=torch.float32, device=dev)

        # -- convert precomputed arrays to tensors (cheap, done once per sample)
        atom_i_t  = torch.tensor(bt["atom_i"],       dtype=torch.long,    device=dev)
        atom_j_t  = torch.tensor(bt["atom_j"],       dtype=torch.long,    device=dev)
        ef_t      = torch.tensor(bt["edge_feat"],    dtype=torch.float32, device=dev)
        group_t   = torch.tensor(bt["group_id"],     dtype=torch.long,    device=dev)
        parent_t  = torch.tensor(bt["parent_frag"],  dtype=torch.long,    device=dev)
        child_a_t = torch.tensor(bt["child_a"],      dtype=torch.long,    device=dev)
        child_b_t = torch.tensor(bt["child_b"],      dtype=torch.long,    device=dev)
        bin_a_t   = torch.tensor(bt["mass_a_bin"],   dtype=torch.long,    device=dev)
        bin_b_t   = torch.tensor(bt["mass_b_bin"],   dtype=torch.long,    device=dev)
        depth_t   = torch.tensor(bt["depth"],        dtype=torch.long,    device=dev)
        n_frags   = bt["n_frags"]

        # 1. GCN — one forward pass on the root molecule
        h = F.relu(self.input_proj(x))
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = self.dropout_layer(h)

        # 2. Score all bond breaks in one batched MLP call.
        #    Output col 0 = break score (softmax within fragment group).
        #    Output col 1 = charge split logit (sigmoid → prob child_a is charged).
        bond_reps  = torch.cat([h[atom_i_t], h[atom_j_t], ef_t], dim=1)  # (N, 2H+BOND_DIM)
        out        = self.bond_scorer(bond_reps)                           # (N, 2)
        all_scores = out[:, 0]                                             # (N,)
        cs_t       = torch.sigmoid(out[:, 1])                             # (N,) learned charge split

        # 3. Segment softmax — bonds within the same fragment compete
        #    with each other, not with bonds in other fragments.
        #    Uses torch_geometric's numerically stable segment softmax.
        from torch_geometric.utils import softmax as segment_softmax
        local_probs = segment_softmax(all_scores, group_t, num_nodes=n_frags)  # (N,)

        # 4. Propagate existence probabilities level by level.
        #    existence[i] = probability that fragment i is present.
        #    Root starts at 1.0; each child gets parent_exist * local_prob * charge_split.
        #    Processing one depth level at a time ensures parent values are
        #    always written before their children are read.
        existence = torch.zeros(n_frags, device=dev)
        existence[bt["root_frag_idx"]] = 1.0

        for depth in range(bt["max_depth"]):
            mask = depth_t == depth
            if not mask.any():
                continue

            p_exist = existence[parent_t[mask]]   # read parents before any writes
            lp      = local_probs[mask]
            cs      = cs_t[mask]

            existence.scatter_add_(0, child_a_t[mask], p_exist * lp * cs)

            b_mask = mask & (child_b_t >= 0)
            if b_mask.any():
                p_exist_b = existence[parent_t[b_mask]]
                existence.scatter_add_(
                    0, child_b_t[b_mask],
                    p_exist_b * local_probs[b_mask] * (1.0 - cs_t[b_mask]),
                )

        # 5. Scatter fragment existence probabilities into the spectrum
        spectrum = torch.zeros(MZ_GRID_SIZE, dtype=torch.float32, device=dev)

        valid_a = bin_a_t >= 0
        if valid_a.any():
            spectrum.scatter_add_(0, bin_a_t[valid_a], existence[child_a_t[valid_a]])

        valid_b = (bin_b_t >= 0) & (child_b_t >= 0)
        if valid_b.any():
            spectrum.scatter_add_(0, bin_b_t[valid_b], existence[child_b_t[valid_b]])

        return spectrum

# endregion
