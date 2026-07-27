"""DiffSMol point-cloud shape autoencoder (VN-DGCNN encoder + implicit
signed-distance decoder).

Ported from DiffSMol ``source/models/shape_pointcloud_modelAE.py``. Only the
encoder is used by the integration: it maps 512 points sampled from a
molecular surface mesh to a ``(128, 3)`` SO(3)-*equivariant* latent that
rotates with the molecule. That latent is DiffSMol's conditioning signal.

Two deliberate departures from upstream, both required for correctness:

1. **``blocks`` is an ``nn.ModuleList``, not a plain Python list.** Upstream
   stores the 6 encoder blocks in a bare ``list``, so PyTorch never registers
   them: they are absent from ``state_dict()``, were never saved into the
   released ``se.pt``, and are therefore re-randomised on every process start.
   That makes upstream's shape latent irreproducible across runs -- fatal
   here, where the precompute, the training run and the generation run are
   three separate processes that must agree on the same embedding function.
   Registering them properly plus vendoring a checkpoint that *contains* them
   (see ``checkpoints/shape_ae_pointcloud.pt``) fixes it.
2. **No ``easydict``.** The vendored checkpoint stores a plain dict config.

The AE is frozen and eval-only here; nothing in this file is trained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from MolecularDiffusion.modules.layers.vn import (
    ResnetBlockFC,
    VNLeakyReLU,
    VNLinear,
    VNLinearLeakyReLU,
    VNResnetBlockFC,
    get_graph_feature_cross,
    mean_pool,
)

EPS = 1e-6

#: Vendored VN-DGCNN shape-AE weights. Vendored (rather than referenced by
#: absolute path into the DiffSMol clone) so the integration survives the
#: ``others/`` tree being deleted.
DEFAULT_SHAPE_AE_CHECKPOINT = str(
    Path(__file__).parent / "checkpoints" / "shape_ae_pointcloud.pt"
)


class DecoderInner(nn.Module):
    """Implicit decoder: (query points, VN latent) -> signed distance.

    Unused by training/generation -- kept so the vendored checkpoint loads
    strictly and so gradient-guidance can be added later without a re-port.
    """

    def __init__(
        self,
        dim: int = 3,
        z_dim: int = 128,
        hidden_size: int = 128,
        layer_num: int = 4,
        loss_type: str = "occupancy",
        leaky: bool = False,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.layer_num = layer_num
        if z_dim > 0:
            self.z_in = VNLinear(z_dim, z_dim)
        self.fc_in = nn.Linear(z_dim * 2 + 1, hidden_size)
        self.blocks = nn.ModuleList(
            ResnetBlockFC(hidden_size) for _ in range(layer_num)
        )
        self.fc_out = nn.Linear(hidden_size, 1)
        self.loss_type = loss_type
        self.actvn = (
            F.relu if not leaky else (lambda x: F.leaky_relu(x, 0.2))
        )

    def forward(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch_size, t, d = p.size()
        net = (p * p).sum(2, keepdim=True)
        if self.z_dim != 0:
            z = z.view(batch_size, -1, d).contiguous()
            net_z = torch.einsum("bmi,bni->bmn", p, z)
            z_dir = self.z_in(z)
            z_inv = (z * z_dir).sum(-1).unsqueeze(1).repeat(1, t, 1)
            net = torch.cat([net, net_z, z_inv], dim=2)

        net = self.fc_in(net)
        for block in self.blocks:
            net = block(net)
        out = self.fc_out(self.actvn(net)).squeeze(-1)
        if self.loss_type == "occupancy":
            out = torch.sigmoid(out)
        return out


class VNDGCNNEncoder(nn.Module):
    """VN-DGCNN encoder: centered point cloud -> ``(latent_dim, 3)`` latent."""

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        layer_num: int,
        num_k: int,
    ) -> None:
        super().__init__()
        self.layer_num = layer_num
        self.num_k = num_k
        self.conv_pos = VNLinearLeakyReLU(2, hidden_dim)
        self.blocks = nn.ModuleList(
            VNLinearLeakyReLU(2 * hidden_dim, hidden_dim)
            for _ in range(layer_num)
        )
        self.pool = mean_pool
        self.conv_c = VNLinearLeakyReLU(
            layer_num * hidden_dim, latent_dim, dim=4, share_nonlinearity=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, 1, N, 3]`` (already surface-centered).

        Returns ``[B, latent_dim, 3]``.
        """
        x = x.transpose(2, 3)  # -> [B, 1, 3, N]
        hidden = self.pool(
            self.conv_pos(get_graph_feature_cross(x, k=self.num_k)), dim=-1
        )
        hiddens = []
        for block in self.blocks:
            hidden = self.pool(
                block(get_graph_feature_cross(hidden, k=self.num_k))
            )
            hiddens.append(hidden)
        latent_vecs = self.conv_c(torch.cat(hiddens, dim=1))
        return latent_vecs.mean(dim=-1, keepdim=False)


class VNResnetEncoder(nn.Module):
    """Alternative VN-ResNet encoder (``encoder: VN_Resnet``). Not used by the
    shipped checkpoint; kept for config parity."""

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        layer_num: int,
        num_k: int,
    ) -> None:
        super().__init__()
        self.layer_num = layer_num
        self.num_k = num_k
        self.conv_pos = VNLinearLeakyReLU(
            3, hidden_dim, negative_slope=0.2, use_batchnorm=False
        )
        self.fc_pos = VNLinear(hidden_dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList(
            VNResnetBlockFC(2 * hidden_dim, hidden_dim)
            for _ in range(layer_num)
        )
        self.pool = mean_pool
        self.fc_c = VNLinear(hidden_dim, latent_dim)
        self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(2, 3)
        hidden = self.pool(
            self.conv_pos(
                get_graph_feature_cross(x, k=self.num_k, if_cross=True)
            ),
            dim=-1,
        )
        hidden = self.fc_pos(hidden)
        for i, block in enumerate(self.blocks):
            hidden = block(hidden)
            pooled = self.pool(hidden, dim=-1, keepdim=True).expand(
                hidden.size()
            )
            hidden = (
                torch.cat((hidden, pooled), dim=1)
                if i < self.layer_num - 1
                else pooled
            )
        hidden = self.pool(hidden, dim=-1)
        return self.fc_c(self.actvn_c(hidden))


class PointCloudAE(nn.Module):
    """VN point-cloud autoencoder. Only ``encode`` is used downstream."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        encoder = config.get("encoder", "VN_DGCNN")
        cls = {
            "VN_DGCNN": VNDGCNNEncoder,
            "VN_Resnet": VNResnetEncoder,
        }[encoder]
        self.encoder = cls(
            config["hidden_dim"],
            config["latent_dim"],
            config["enc_layer_num"],
            config["num_k"],
        )
        self.generator = DecoderInner(
            config["point_dim"],
            config["latent_dim"],
            config["hidden_dim"],
            config["dec_layer_num"],
            config["loss_type"],
        )
        self.loss_type = config["loss_type"]

    def encode(self, point_clouds: torch.Tensor) -> torch.Tensor:
        """``point_clouds``: ``[B, N, 3]``, surface-centered.

        Returns the equivariant latent ``[B, latent_dim, 3]``.
        """
        return self.encoder(point_clouds.unsqueeze(1))

    def forward(
        self, point_clouds: torch.Tensor, query_points: torch.Tensor
    ) -> torch.Tensor:
        return self.generator(query_points, self.encode(point_clouds))


def load_shape_ae(
    checkpoint: str | None = None, device: str | torch.device = "cpu"
) -> PointCloudAE:
    """Load the frozen shape autoencoder from a vendored checkpoint.

    The checkpoint is a plain ``{"config": {...}, "model": state_dict}`` dict
    (no ``easydict``). Parameters are detached and the module is put in eval
    mode -- the AE is a fixed featuriser, never trained here.
    """
    path = checkpoint or DEFAULT_SHAPE_AE_CHECKPOINT
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PointCloudAE(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()
