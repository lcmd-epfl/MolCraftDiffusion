"""Adapter binding the ported GFMDiff ``EquiGNN`` backbone to the
``dynamics._forward(t, xh, node_mask, edge_mask, context)`` contract that
``EnVariationalDiffusion.phi`` (``modules/models/en_diffusion.py``) calls,
mirroring the existing ``EGT_dynamics`` wrapper
(``modules/models/egt.py::EGT_dynamics._forward``).

``EquiGNN._forward(z_t, t, node_mask, pair_mask, context)`` already expects
the same conceptual shapes: ``z_t``/``xh`` is ``(B, N, 3 + in_node_nf)``
(position + feature channels concatenated), ``node_mask`` is ``(B, N, 1)``,
and ``pair_mask``/``edge_mask`` is the same dense ``(B, N, N, 1)`` tensor
family — so this wrapper only reorders the ``t``/state args and renames
``edge_mask`` -> ``pair_mask``; no reshaping/data-model translation is
needed.
"""

import torch.nn as nn

from MolecularDiffusion.modules.models.gfmdiff.equignn import EquiGNN


class GFMDiffDynamics(nn.Module):
    """Thin wrapper constructing the ported ``EquiGNN`` internally.

    Args:
        in_node_nf (int): Number of node feature channels consumed by
            ``EnVariationalDiffusion`` (atom-type one-hot + atomic number
            [+ extra normalized values]), *not* including diffusion time or
            context — ``EquiGNN`` concatenates both internally.
        context_node_nf (int): Number of context/conditioning channels.
        n_dims (int): Number of spatial dimensions (3 for 3D coordinates).
        num_layers, emb_dim, hidden_dim, num_heads, dropout: Dual-Track
            Transformer backbone hyperparameters (see ``EquiGNN``).
        add_time (bool): Whether distance/angle embeddings are also time
            conditioned (GFMDiff's own ``add_time`` flag).
        block_calc (bool): Whether pairwise/triplet embeddings are
            recomputed after every layer (GFMDiff's own ``block_calc`` flag).
        dataset_name (str): Selects the ``InterBlock`` low-to-high pair
            update variant ("qm9" or "drugs"); unrelated to which dataset is
            actually trained on in this repo.
    """

    def __init__(
        self,
        in_node_nf: int,
        context_node_nf: int = 0,
        n_dims: int = 3,
        num_layers: int = 5,
        emb_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        pair_loss_scale: float = 0.0,
        add_time: bool = False,
        block_calc: bool = True,
        dataset_name: str = "qm9",
    ):
        super().__init__()
        self.n_dims = n_dims
        self.context_node_nf = context_node_nf

        # EquiGNN adds the atomic-number channel (include_an=True) and
        # concatenates time/context itself, so the x_class it needs to be
        # built with is in_node_nf minus that +1 atomic-number channel.
        x_class = in_node_nf - 1

        model_config = {
            "num_layers": num_layers,
            "emb_dim": emb_dim,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "dropout": dropout,
            "pair_loss_scale": pair_loss_scale,
            "context": context_node_nf > 0,
            "context_col": list(range(context_node_nf)) if context_node_nf > 0 else [0],
            "include_an": True,
            "include_de": False,
            "add_time": add_time,
            "block_calc": block_calc,
        }
        data_config = {
            "x_class": x_class,
            "c_class": x_class,
            "name": dataset_name,
        }
        self.equignn = EquiGNN(model_config, data_config)

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)

        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context=None):
        bs, n_nodes = xh.shape[0], xh.shape[1]
        pair_mask = edge_mask.view(bs, n_nodes, n_nodes, -1)
        out = self.equignn._forward(xh, t, node_mask, pair_mask, context)
        # ponytail: EquiGNN's own remove_mean_with_mask only recenters the
        # mean, it never zeroes individual masked rows, so padded atoms can
        # drift nonzero through the transformer's coordinate layers. Mirror
        # EGT_dynamics._forward's defensive re-mask (egt.py) before
        # returning, since EnVariationalDiffusion.sample_p_zs_given_zt
        # asserts masked rows are exactly zero.
        return out * node_mask
