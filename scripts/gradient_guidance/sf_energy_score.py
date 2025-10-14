import sys
from math import sqrt
from typing import List

import numpy as np
import torch

# from navicatGA.timeout import timeout
from numpy import dot
from MolecularDiffusion.core import Engine
from torch_geometric.data import Data
from torch_geometric.nn import  radius_graph
from torch_geometric.data import Batch
#%% sf energy score
def distance_to_line(point, line_start, line_end):
    """Calculate the perpendicular distance from a point to a line segment."""
    v1 = torch.cat([line_end - line_start, torch.zeros(1, device=line_end.device, dtype=line_end.dtype)])
    v2 = torch.cat([line_start - point, torch.zeros(1, device=line_end.device, dtype=line_end.dtype)])
    cross = torch.cross(v1, v2, dim=0)
    return torch.norm(cross) / torch.norm(line_end - line_start)

def is_within_triangle(x, y, T1_cutoff, S1_cutoff):
    """Check if the point is within the defined triangle area."""
    return (x >= T1_cutoff) and (y >= 2 * x) and (y <= S1_cutoff)

def pointTriangleDistance(TRI, P):
    """Torch implementation of the point-to-triangle distance in 3D."""
    B = TRI[0, :]
    E0 = TRI[1, :] - B
    E1 = TRI[2, :] - B
    D = B - P

    a = torch.dot(E0, E0)
    b = torch.dot(E0, E1)
    c = torch.dot(E1, E1)
    d = torch.dot(E0, D)
    e = torch.dot(E1, D)
    f = torch.dot(D, D)

    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    if (s + t) <= det:
        if s < 0.0:
            if t < 0.0:
                if d < 0:
                    t = 0.0
                    if -d >= a:
                        s = 1.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
                else:
                    s = 0.0
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        if -e >= c:
                            t = 1.0
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f
            else:
                s = 0.0
                if e >= 0.0:
                    t = 0.0
                    sqrdistance = f
                else:
                    if -e >= c:
                        t = 1.0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
        else:
            if t < 0.0:
                t = 0.0
                if d >= 0.0:
                    s = 0.0
                    sqrdistance = f
                else:
                    if -d >= a:
                        s = 1.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
            else:
                invDet = 1.0 / det
                s *= invDet
                t *= invDet
                sqrdistance = s * (a * s + b * t + 2.0 * d) + \
                              t * (b * s + c * t + 2.0 * e) + f
    else:
        if s < 0.0:
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer >= denom:
                    s = 1.0
                    t = 0.0
                    sqrdistance = a + 2.0 * d + f
                else:
                    s = numer / denom
                    t = 1 - s
                    sqrdistance = s * (a * s + b * t + 2 * d) + \
                                  t * (b * s + c * t + 2 * e) + f
            else:
                s = 0.0
                if tmp1 <= 0.0:
                    t = 1.0
                    sqrdistance = c + 2.0 * e + f
                else:
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
        else:
            if t < 0.0:
                tmp0 = b + e
                tmp1 = a + d
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        t = 1.0
                        s = 0.0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = numer / denom
                        s = 1 - t
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + \
                                      t * (b * s + c * t + 2.0 * e) + f
                else:
                    t = 0.0
                    if tmp1 <= 0.0:
                        s = 1.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        if d >= 0.0:
                            s = 0.0
                            sqrdistance = f
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
            else:
                numer = c + e - b - d
                if numer <= 0.0:
                    s = 0.0
                    t = 1.0
                    sqrdistance = c + 2.0 * e + f
                else:
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = numer / denom
                        t = 1 - s
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + \
                                      t * (b * s + c * t + 2.0 * e) + f

    if sqrdistance < 0.0:
        sqrdistance = torch.tensor(0.0, device=det.device, dtype=det.dtype)
    
    dist = torch.sqrt(sqrdistance)
    PP0 = B + s * E0 + t * E1
    return dist, PP0



def energy_score(x, y, S1_cutoff=3.8, scaling_S1=1 / 3.0, T1_cutoff=1.5):
    """
    Torch-compatible and differentiable energy score function.
    Inputs x, y must be torch tensors (with or without requires_grad).
    """
    device = x.device
    TRIANGLE_SCALING = 1.0 / 0.11871871871871865

    p1 = torch.tensor([T1_cutoff, 1.0], device=device)
    p2 = torch.tensor([T1_cutoff, 100.0], device=device)
    p0 = torch.tensor([0.0, 0.0], device=device)
    p4 = torch.tensor([50.0, 100.0], device=device)
    pS1_1 = torch.tensor([1.0, S1_cutoff], device=device)
    pS1_2 = torch.tensor([10.0, S1_cutoff], device=device)
    p3 = torch.stack([x, y])

    if is_within_triangle(x, y, T1_cutoff, S1_cutoff):
        dist1 = TRIANGLE_SCALING * distance_to_line(p3, p1, p2)
        dist2 = TRIANGLE_SCALING * distance_to_line(p3, p0, p4)
        dist3 = scaling_S1 * TRIANGLE_SCALING * distance_to_line(p3, pS1_1, pS1_2)
        return torch.min(torch.stack([dist1, dist2, dist3]))
    else:
        TRI = torch.tensor([
            [T1_cutoff, 2 * T1_cutoff, 0.0],
            [T1_cutoff, S1_cutoff, 0.0],
            [S1_cutoff / 2.0, S1_cutoff, 0.0]
        ], device=device)

        # avoid breaking autograd by not calling .item() or converting to float
        P = torch.stack([x, y, torch.tensor(0.0, device=device, dtype=x.dtype)])
        dist, _ = pointTriangleDistance(TRI, P)
        return -TRIANGLE_SCALING * dist

#%% wrapper


class SFEnergyScore:
    def __init__(
        self,
        chkpt_directory: str,
        atom_vocab: List[str] = None,
        norm_factor: List[int] = [1, 4, 10], # as used in training the guidance model
        node_feature: str = None,
    ):
        self.chkpt_directory = chkpt_directory
        self.atom_vocab = atom_vocab if atom_vocab is not None else ["H", "C", "N", "O", "F"]
        self.norm_factor = norm_factor
        self.solver = self._load_model(self.chkpt_directory)
        
        # self.norm_factor = self.solver.model.norm_values
        self.norm_factor =  getattr(self.solver.model, 'norm_values', [1,1,1])
        self.node_feature = node_feature 

        self.n_dim = 3  # 3D coordinates
        self._initialize_model_attributes()

    def _load_model(self, chkpt_path):
        """
        Loads a pre-trained model from a checkpoint file.

        Args:
            chkpt_path (str): The path to the checkpoint file.

        Returns:
            MolecularDiffusion.core.Engine: The loaded Engine object with the model in evaluation mode.
        """
        engine = Engine(None, None, None, None, None)
        engine = engine.load_from_checkpoint(chkpt_path)
        engine.model.eval()
        return engine

    def _initialize_model_attributes(self):
        """
        Initializes atom_vocab, node_feature, std, weight, and mean attributes
        of the solver's model if they are not already present.
        """
        if not hasattr(self.solver.model, 'std'):
            chkpt = torch.load(self.chkpt_directory)
            self.solver.model.std = chkpt["model"]["std"].to(self.solver.model.device)
            self.solver.model.weight = chkpt["model"]["weight"].to(self.solver.model.device)
            self.solver.model.mean = chkpt["model"]["mean"].to(self.solver.model.device)

        if not hasattr(self.solver.model, 'atom_vocab'):
            self.solver.model.atom_vocab = self.atom_vocab
        if not hasattr(self.solver.model, 'node_feature'):
            self.solver.model.node_feature = self.node_feature

    def __call__(self, xh, t):
        """
        Calculates the negative energy score for a given molecular configuration.
        This method makes the class instance callable.
        """
        bs, n_nodes, _ = xh.shape
        RADIUS = 4
        device = xh.device
        
        mol = {} # This dictionary is used to pass graph data to the model
        coords = xh[:, :, :self.n_dim] / self.norm_factor[0]
        h = xh[:, :, self.n_dim:-1] / self.norm_factor[1]
        charge = xh[:, :, -1] / self.norm_factor[2]
        
        coords = coords.view(n_nodes*bs, self.n_dim).to(device)
        h = h.view(n_nodes*bs, -1).to(device)
        charge = charge.view(n_nodes*bs).to(device)

        edge_index = radius_graph(coords, r=RADIUS)
        tags = torch.zeros(n_nodes, dtype=torch.long, device=device)

        times = torch.zeros(n_nodes, dtype=torch.float32, device=device) + t.item()
        times = times.view(n_nodes, 1)

        graph_data = Data(
                            x=h,
                            pos=coords,
                            atomic_numbers=charge,
                            natoms=n_nodes,
                            smiles=None,
                            xyz=None,
                            edge_index=edge_index,
                            tags=tags,
                            times=times,
                        )
        graph_data = Batch.from_data_list([graph_data])
        
        graph_data = graph_data.to(device)
        mol["graph"] = graph_data # Assign the batched graph data to the 'graph' key
        preds = self.solver.model.predict(mol, evaluate=True)[0]
        target = -energy_score(preds[1], preds[0])
    
        return target
