"""IPNet: IPDiff's frozen, pretrained binding-interaction prior.

Ported from IPDiff's ``graphbap/bapnet.py`` (commit ``00ed078``). This is
**not** a diffusion model and it is never trained here: it is a separately
pretrained binding-affinity network whose *hidden* representation is the
conditioning signal IPDiff injects into the denoiser
(:class:`~...ipdiff.score_model.IPDiffScorePosNet3D`).

Three parallel EGNN stacks run over the same complex -- the joint
ligand+pocket graph, the ligand alone, and the pocket alone -- and a single
``GATConv`` fuses the complex embedding with the concatenated
ligand-only/pocket-only embeddings. :meth:`BAPNet.extract_features` returns
the fused per-node vectors ``(h_ligand, h_pocket)``, each ``(N, 128)``.

Note what it does **not** return: the scalar affinity. ``OutputLayer`` and
``FinalOutput`` exist in the released ``ipnet`` checkpoint and are
constructed here for state-dict key parity, but ``extract_features`` -- the
only entry point IPDiff ever calls (``models/molopt_score_model.py:488``
during training, ``:624`` inside the sampler) -- stops at the fusion layer.
They are kept rather than deleted so the checkpoint maps with zero dropped
tensors.

**Cost warning.** :func:`get_edges` builds a *fully connected* graph within
each batch element (upstream passes no ``edge_cutoff``), and the sampler
re-runs this every reverse step. With ~300-1000 pocket atoms that is
10^5-10^6 edges per complex. Upstream trains at ``batch_size: 4``; treat OOM
here as a batch-size question, not a bug.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional

import torch
from torch import nn
from torch_geometric.nn import GATConv
from torch_scatter import scatter_mean

#: Vocabularies the three embedding tables are sized against. Each table is
#: ``len(vocab) + 1`` wide, so the 13 / 6 / 20 class ranges this platform's
#: KGDiff featurisation produces all fit without remapping.
LIGAND_ATOM_ADD_AROMATIC_TYPES = [
    "H", "C1", "C2", "N1", "N2", "O1", "O2",
    "F", "P1", "P2", "S1", "S2", "Cl",
]
POCKET_ATOM_TYPES = ["H", "C", "N", "O", "S", "Se"]
RESIDUE_TYPES = [
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR",
]


#: torch-geometric version drift, not a port decision. The released ``ipnet``
#: was saved when ``GATConv`` with an INT ``in_channels`` built ``lin_src``
#: and ``lin_dst`` as the SAME module registered twice; PyG >= 2.4 builds a
#: single ``lin`` for that case (``nn/conv/gat_conv.py``, ``if
#: isinstance(in_channels, int)``). So one key is renamed and its byte-
#: identical twin is dropped -- verified with ``torch.equal``, never assumed.
GATCONV_RENAMES = {"FusionGraph.0.lin_src.weight": "FusionGraph.0.lin.weight"}
GATCONV_ALIASES = {"FusionGraph.0.lin_dst.weight": "FusionGraph.0.lin.weight"}


def get_edges(mask: torch.Tensor) -> torch.Tensor:
    """Fully connected edges within each batch element.

    Upstream's ``edge_cutoff`` argument is ``None`` everywhere it is called,
    so the distance-gated branch is dropped. The ``.cpu()`` round trip
    upstream does between building and using this is also dropped -- it was
    a no-op.
    """
    adj = mask[:, None] == mask[None, :]
    return torch.stack(torch.where(adj), dim=0)


def coord2diff(x, edge_index, norm_constant: float = 1):
    """Squared edge length and the normalised edge direction."""
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    return radial, coord_diff / (norm + norm_constant)


def unsorted_segment_sum(
    data, segment_ids, num_segments, normalization_factor, aggregation_method
):
    """Scatter-add with either a constant divisor or a per-node mean."""
    result = data.new_full((num_segments, data.size(1)), 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == "sum":
        result = result / normalization_factor
    if aggregation_method == "mean":
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


def remove_pocket_mean(x_lig, x_pocket, lig_indices, pocket_indices):
    """Centre both clouds on the *pocket* centroid.

    IPNet was pretrained in this frame ("pocketCoM" in the released
    checkpoint's path), and the diffusion model centres the same way
    (``center_pos_mode: protein``), so features stay consistent.
    """
    pocket_mean = scatter_mean(x_pocket, pocket_indices, dim=0)
    return (
        x_lig - pocket_mean[lig_indices],
        x_pocket - pocket_mean[pocket_indices],
    )


class SinusoidsEmbeddingNew(nn.Module):
    """Multi-resolution distance expansion (unused: ``sin_embedding=False``).

    Kept because it is reachable through the constructor flag and costs
    nothing; the released checkpoint does not use it (it holds no parameters
    either way).
    """

    def __init__(
        self,
        max_res: float = 15.0,
        min_res: float = 15.0 / 2000.0,
        div_factor: int = 4,
    ) -> None:
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = (
            2 * math.pi * div_factor ** torch.arange(self.n_frequencies)
        ) / max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        return torch.cat((emb.sin(), emb.cos()), dim=-1).detach()


class GCL(nn.Module):
    """Invariant message-passing layer (EGNN's ``h`` update)."""

    def __init__(
        self,
        input_nf: int,
        output_nf: int,
        hidden_nf: int,
        normalization_factor: float,
        aggregation_method: str,
        edges_in_d: int = 0,
        nodes_att_dim: int = 0,
        act_fn: Optional[nn.Module] = None,
        attention: bool = False,
    ) -> None:
        super().__init__()
        act_fn = act_fn if act_fn is not None else nn.SiLU()
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def forward(self, h, edge_index, edge_attr=None):
        row, _col = edge_index
        source, target = h[edge_index[0]], h[edge_index[1]]
        if edge_attr is None:
            edge_in = torch.cat([source, target], dim=1)
        else:
            edge_in = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(edge_in)
        edge_feat = mij * self.att_mlp(mij) if self.attention else mij

        agg = unsorted_segment_sum(
            edge_feat,
            row,
            num_segments=h.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        return h + self.node_mlp(torch.cat([h, agg], dim=1))


class EquivariantUpdate(nn.Module):
    """Coordinate update: a scalar per edge times the edge direction."""

    def __init__(
        self,
        hidden_nf: int,
        normalization_factor: float,
        aggregation_method: str,
        edges_in_d: int = 1,
        act_fn: Optional[nn.Module] = None,
        tanh: bool = False,
        coords_range: float = 10.0,
    ) -> None:
        super().__init__()
        act_fn = act_fn if act_fn is not None else nn.SiLU()
        self.tanh = tanh
        self.coords_range = coords_range
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 2 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer,
        )
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def forward(self, h, coord, edge_index, coord_diff, edge_attr):
        row, _col = edge_index
        scale = self.coord_mlp(
            torch.cat([h[edge_index[0]], h[edge_index[1]], edge_attr], dim=1)
        )
        if self.tanh:
            trans = coord_diff * torch.tanh(scale) * self.coords_range
        else:
            trans = coord_diff * scale
        agg = unsorted_segment_sum(
            trans,
            row,
            num_segments=coord.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        return coord + agg


class EquivariantBlock(nn.Module):
    """``n_layers`` invariant GCLs followed by one coordinate update.

    Submodules are registered with upstream's ``gcl_%d`` / ``gcl_equiv``
    names via ``add_module`` so the released checkpoint's keys match.
    """

    def __init__(
        self,
        hidden_nf: int,
        edge_feat_nf: int = 2,
        act_fn: Optional[nn.Module] = None,
        n_layers: int = 2,
        attention: bool = True,
        norm_diff: bool = True,
        tanh: bool = False,
        coords_range: float = 15,
        norm_constant: float = 1,
        sin_embedding: Optional[nn.Module] = None,
        normalization_factor: float = 100,
        aggregation_method: str = "sum",
    ) -> None:
        super().__init__()
        act_fn = act_fn if act_fn is not None else nn.SiLU()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(n_layers):
            self.add_module(
                f"gcl_{i}",
                GCL(
                    hidden_nf,
                    hidden_nf,
                    hidden_nf,
                    edges_in_d=edge_feat_nf,
                    act_fn=act_fn,
                    attention=attention,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method,
                ),
            )
        self.add_module(
            "gcl_equiv",
            EquivariantUpdate(
                hidden_nf,
                edges_in_d=edge_feat_nf,
                act_fn=nn.SiLU(),
                tanh=tanh,
                coords_range=self.coords_range_layer,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            ),
        )

    def forward(self, h, x, edge_index, edge_attr):
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(self.n_layers):
            h = self._modules[f"gcl_{i}"](h, edge_index, edge_attr=edge_attr)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr)
        return h, x


class BAPNet(nn.Module):
    """The pretrained interaction prior. Frozen; never trained in-platform.

    ``ckpt_path`` is required (upstream asserts it too): an un-pretrained
    IPNet emits noise, and IPDiff's whole conditioning signal is this
    network's output, so silently running without weights would train a
    model on garbage. Weights are loaded at construction and then frozen;
    because the module is a real submodule of the task, they also round-trip
    through the platform checkpoint and are restored at generate time.
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        hidden_nf: int = 128,
        act_fn: Optional[nn.Module] = None,
        GAT_head: int = 2,  # noqa: N803 - upstream's name, kept for configs
        graph_layers: int = 1,
        attention: bool = False,
        norm_diff: bool = True,
        tanh: bool = False,
        coords_range: float = 15,
        norm_constant: float = 1,
        inv_sublayers: int = 1,
        sin_embedding: bool = False,
        normalization_factor: float = 100,
        aggregation_method: str = "sum",
        ignore_keys: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        act_fn = act_fn if act_fn is not None else nn.SiLU()
        graph_dim = hidden_nf
        self.graph_dim = graph_dim
        self.hidden_nf = hidden_nf
        self.graph_layers = graph_layers

        self.ligand_atom_type_embed = nn.Embedding(
            len(LIGAND_ATOM_ADD_AROMATIC_TYPES) + 1, graph_dim
        )
        self.pocket_atom_type_embed = nn.Embedding(
            len(POCKET_ATOM_TYPES) + 1, graph_dim
        )
        self.pocket_residue_type_embed = nn.Embedding(
            len(RESIDUE_TYPES) + 1, graph_dim
        )
        self.pocket_type_fusion = nn.Linear(graph_dim * 2, graph_dim)
        self.id_embed = nn.Embedding(2, 4)
        self.embed_fusion = nn.Linear(graph_dim + 4, graph_dim)

        if sin_embedding:
            self.sin_embedding: Optional[nn.Module] = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        def block() -> EquivariantBlock:
            return EquivariantBlock(
                hidden_nf,
                edge_feat_nf=edge_feat_nf,
                act_fn=act_fn,
                n_layers=inv_sublayers,
                attention=attention,
                norm_diff=norm_diff,
                tanh=tanh,
                coords_range=coords_range,
                norm_constant=norm_constant,
                sin_embedding=self.sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )

        self.ComplexesGraph = nn.ModuleList(
            [block() for _ in range(graph_layers)]
        )
        self.LigandGraph = nn.ModuleList([block() for _ in range(graph_layers)])
        self.PocketGraph = nn.ModuleList([block() for _ in range(graph_layers)])
        self.FusionGraph = nn.ModuleList(
            [GATConv(graph_dim * 2, graph_dim, GAT_head, concat=False)]
        )

        # Present in the released checkpoint and constructed for key parity,
        # but NOT reached: extract_features() (the only caller, upstream
        # molopt_score_model.py:488 and :624) returns the fusion output and
        # never touches the affinity read-out.
        self.OutputLayer = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.Hardswish(),
            nn.Linear(graph_dim, graph_dim),
        )
        self.FinalOutput = nn.Linear(graph_dim, 1)

        if ckpt_path is None:
            raise ValueError(
                "BAPNet needs the pretrained IPNet weights: set "
                "tasks.net_cond_ckpt (default: "
                "docs/model_integrations/ipdiff/checkpoints/ipnet). IPDiff's "
                "entire conditioning signal is this network's output."
            )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"net_cond_ckpt not found: {ckpt_path}")
        self.init_from_ckpt(ckpt_path, ignore_keys or [])
        self.freeze()

    def freeze(self) -> None:
        """Upstream's ``freeze_the_model``: eval mode, no gradients."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> "BAPNet":
        """Stay in eval mode even when the parent task calls ``.train()``.

        Nothing here is mode-dependent (no dropout, no batchnorm), so this is
        belt-and-braces -- but it keeps "frozen" true rather than nominal.
        """
        return super().train(False)

    def init_from_ckpt(self, path: str, ignore_keys: List[str]) -> None:
        """Load the released ``ipnet`` weights, strictly.

        Strict on purpose: this is the one place a silent partial load would
        be invisible and fatal (an unloaded fusion layer would emit noise
        that still looks like a 128-d feature). The released file predates
        the current torch-geometric, so its ``GATConv`` keys are translated
        first -- see :data:`GATCONV_RENAMES`.
        """
        sd = torch.load(path, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        sd = {
            k: v
            for k, v in sd.items()
            if not any(k.startswith(ik) for ik in ignore_keys)
        }
        for old, new in GATCONV_RENAMES.items():
            if old in sd:
                sd[new] = sd.pop(old)
        for dropped, twin in GATCONV_ALIASES.items():
            if dropped in sd and torch.equal(sd[dropped], sd[twin]):
                del sd[dropped]
        self.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def extract_features(
        self,
        lig_coords,
        pocket_coords,
        lig_a_hidx,
        pocket_a_hidx,
        pocket_r_hidx,
        lig_mask,
        pocket_mask,
    ):
        """Fused per-node interaction features ``(h_ligand, h_pocket)``.

        ``lig_a_hidx`` is the 13-class ``(element, aromatic)`` ligand index;
        ``pocket_a_hidx`` / ``pocket_r_hidx`` are the pocket element and
        amino-acid indices (``argmax`` of the 27-dim protein feature's first
        6 and next 20 columns). ``*_mask`` are the scatter batch indices.
        """
        lig_coords, pocket_coords = remove_pocket_mean(
            lig_coords, pocket_coords, lig_mask, pocket_mask
        )
        device = lig_coords.device
        num_lig = lig_coords.shape[0]

        complexes_mask = torch.cat([lig_mask, pocket_mask], dim=0)
        complexes_coords = torch.cat(
            [lig_coords, pocket_coords], dim=0
        ).to(torch.float32)
        complexes_id = torch.cat(
            [
                torch.zeros(num_lig, dtype=torch.long, device=device),
                torch.ones(
                    pocket_coords.shape[0], dtype=torch.long, device=device
                ),
            ]
        )

        pocket_type_emb = self.pocket_type_fusion(
            torch.cat(
                [
                    self.pocket_atom_type_embed(pocket_a_hidx),
                    self.pocket_residue_type_embed(pocket_r_hidx),
                ],
                dim=1,
            )
        )
        complexes_type_emb = torch.cat(
            [self.ligand_atom_type_embed(lig_a_hidx), pocket_type_emb], dim=0
        )
        complexes_emb = self.embed_fusion(
            torch.cat([complexes_type_emb, self.id_embed(complexes_id)], dim=-1)
        )

        complexes_edge_index = get_edges(complexes_mask)
        ligand_edge_index = get_edges(lig_mask)
        pocket_edge_index = get_edges(pocket_mask)

        def distances(coords, edge_index):
            d, _ = coord2diff(coords, edge_index)
            if self.sin_embedding is None:
                return d
            return self.sin_embedding(d)

        complexes_distances = distances(complexes_coords, complexes_edge_index)
        lig_distances = distances(lig_coords, ligand_edge_index)
        pocket_distances = distances(pocket_coords, pocket_edge_index)

        o_c = complexes_emb
        o_l, o_p = complexes_emb.clone()[:num_lig], complexes_emb.clone()[num_lig:]
        for i in range(self.graph_layers):
            o_c, complexes_coords = self.ComplexesGraph[i](
                o_c, complexes_coords, complexes_edge_index, complexes_distances
            )
            o_l, lig_coords = self.LigandGraph[i](
                o_l, lig_coords, ligand_edge_index, lig_distances
            )
            o_p, pocket_coords = self.PocketGraph[i](
                o_p, pocket_coords, pocket_edge_index, pocket_distances
            )

        o_c = self.FusionGraph[0](
            torch.cat([o_c, torch.cat([o_l, o_p], dim=0)], dim=1),
            complexes_edge_index,
        )
        return o_c[:num_lig].detach(), o_c[num_lig:].detach()
