"""EGNN wrapper networks, ported from GeoLDM's `egnn/models.py`
(commit 03ae2031c712a1a6c1678e747bdcdc7a7560e00b).

`remove_mean` / `remove_mean_with_mask` are reused from
`MolecularDiffusion.utils` (same implementations, no need to re-port
GeoLDM's own `equivariant_diffusion/utils.py` copies) per the integration
plan.
"""

import numpy as np
import torch
import torch.nn as nn

from MolecularDiffusion.utils import remove_mean, remove_mean_with_mask

from .egnn import EGNN, GNN


class EGNN_dynamics_QM9(nn.Module):
    def __init__(
        self,
        in_node_nf,
        context_node_nf,
        n_dims,
        hidden_nf=64,
        device="cpu",
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        attention=False,
        condition_time=True,
        tanh=False,
        mode="egnn_dynamics",
        norm_constant=0,
        inv_sublayers=2,
        sin_embedding=False,
        normalization_factor=100,
        aggregation_method="sum",
    ):
        super().__init__()
        self.mode = mode
        if mode == "egnn_dynamics":
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf,
                in_edge_nf=1,
                hidden_nf=hidden_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                attention=attention,
                tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers,
                sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )
            self.in_node_nf = in_node_nf
        elif mode == "gnn_dynamics":
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3,
                in_edge_nf=0,
                hidden_nf=hidden_nf,
                out_node_nf=3 + in_node_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                attention=attention,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)

        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs * n_nodes, 1)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask
        x = xh[:, 0 : self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs * n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims :].clone()

        if self.condition_time:
            if np.prod(t.size()) == 1:
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

        if context is not None:
            context = context.view(bs * n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        if self.mode == "egnn_dynamics":
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            vel = (x_final - x) * node_mask
        elif self.mode == "gnn_dynamics":
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]
        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            h_final = h_final[:, : -self.context_node_nf]

        if self.condition_time:
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print("Warning: detected nan, resetting EGNN output to zero.")
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)


class EGNN_encoder_QM9(nn.Module):
    def __init__(
        self,
        in_node_nf,
        context_node_nf,
        out_node_nf,
        n_dims,
        hidden_nf=64,
        device="cpu",
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        attention=False,
        tanh=False,
        mode="egnn_dynamics",
        norm_constant=0,
        inv_sublayers=2,
        sin_embedding=False,
        normalization_factor=100,
        aggregation_method="sum",
        include_charges=True,
    ):
        """
        :param in_node_nf: Number of invariant features for input nodes.
        """
        super().__init__()

        include_charges = int(include_charges)
        num_classes = in_node_nf - include_charges

        self.mode = mode
        if mode == "egnn_dynamics":
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf,
                out_node_nf=hidden_nf,
                in_edge_nf=1,
                hidden_nf=hidden_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                attention=attention,
                tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers,
                sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )
            self.in_node_nf = in_node_nf
        elif mode == "gnn_dynamics":
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3,
                out_node_nf=hidden_nf + 3,
                in_edge_nf=0,
                hidden_nf=hidden_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                attention=attention,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, out_node_nf * 2 + 1),
        )

        self.num_classes = num_classes
        self.include_charges = include_charges
        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}

        self.out_node_nf = out_node_nf

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)

        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs * n_nodes, 1)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask
        x = xh[:, 0 : self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs * n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims :].clone()

        if context is not None:
            context = context.view(bs * n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        if self.mode == "egnn_dynamics":
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            vel = x_final * node_mask
        elif self.mode == "gnn_dynamics":
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]
        else:
            raise Exception("Wrong mode %s" % self.mode)

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print("Warning: detected nan, resetting EGNN output to zero.")
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        h_final = self.final_mlp(h_final)
        h_final = h_final * node_mask if node_mask is not None else h_final
        h_final = h_final.view(bs, n_nodes, -1)

        vel_mean = vel
        vel_std = h_final[:, :, :1].sum(dim=1, keepdim=True).expand(-1, n_nodes, -1)
        vel_std = torch.exp(0.5 * vel_std)

        h_mean = h_final[:, :, 1 : 1 + self.out_node_nf]
        h_std = torch.exp(0.5 * h_final[:, :, 1 + self.out_node_nf :])

        if torch.any(torch.isnan(vel_std)):
            print("Warning: detected nan in vel_std, resetting to one.")
            vel_std = torch.ones_like(vel_std)

        if torch.any(torch.isnan(h_std)):
            print("Warning: detected nan in h_std, resetting to one.")
            h_std = torch.ones_like(h_std)

        return vel_mean, vel_std, h_mean, h_std

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)


class EGNN_decoder_QM9(nn.Module):
    def __init__(
        self,
        in_node_nf,
        context_node_nf,
        out_node_nf,
        n_dims,
        hidden_nf=64,
        device="cpu",
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        attention=False,
        tanh=False,
        mode="egnn_dynamics",
        norm_constant=0,
        inv_sublayers=2,
        sin_embedding=False,
        normalization_factor=100,
        aggregation_method="sum",
        include_charges=True,
    ):
        super().__init__()

        include_charges = int(include_charges)
        num_classes = out_node_nf - include_charges

        self.mode = mode
        if mode == "egnn_dynamics":
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf,
                out_node_nf=out_node_nf,
                in_edge_nf=1,
                hidden_nf=hidden_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                attention=attention,
                tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers,
                sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )
            self.in_node_nf = in_node_nf
        elif mode == "gnn_dynamics":
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3,
                out_node_nf=out_node_nf + 3,
                in_edge_nf=0,
                hidden_nf=hidden_nf,
                device=device,
                act_fn=act_fn,
                n_layers=n_layers,
                attention=attention,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
            )

        self.num_classes = num_classes
        self.include_charges = include_charges
        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)

        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs * n_nodes, 1)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask
        x = xh[:, 0 : self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs * n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims :].clone()

        if context is not None:
            context = context.view(bs * n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        if self.mode == "egnn_dynamics":
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            vel = x_final * node_mask
        elif self.mode == "gnn_dynamics":
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]
        else:
            raise Exception("Wrong mode %s" % self.mode)

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print("Warning: detected nan, resetting EGNN output to zero.")
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if node_mask is not None:
            h_final = h_final * node_mask
        h_final = h_final.view(bs, n_nodes, -1)

        return vel, h_final

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)
