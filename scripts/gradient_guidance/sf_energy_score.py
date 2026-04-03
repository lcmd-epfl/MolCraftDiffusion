import sys
import os
from math import sqrt
from typing import List

import numpy as np
import torch
import rootutils

# Setup root directory
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# from navicatGA.timeout import timeout
import e3nn
import torch.fx
import torch.nn as nn

# Monkeypatch GraphModule.__init__ to handle e3nn's self-referential initialization
# which fails in some Hydra/PyTorch environments due to isinstance checks
_original_graph_module_init = torch.fx.GraphModule.__init__

def _patched_graph_module_init(self, root, graph, class_name="GraphModule"):
    if isinstance(root, torch.nn.Module):
        # Normal case
        _original_graph_module_init(self, root, graph, class_name)
    elif root is self:
        # e3nn Legendre case: root IS self, and for some reason isinstance check failed
        # We manually proceed as if it passed, relying on the fact that self IS a Module
        # We need to bypass the check in original init.
        
        # IMPORTANT: Initialize nn.Module base class
        # Use torch.nn.modules.module.Module explicitly to bypass torchdrug's patch of torch.nn.Module
        torch.nn.modules.module.Module.__init__(self)
        
        # Reimplementing the specific branch for Module root:
        self.__class__.__name__ = class_name
        if hasattr(root, "training"):
            self.training = root.training
        
        # When we pickle/unpickle graph module, we don't want to drop any module or attributes.
        # (Skipping _CodeOnlyModule check for now as likely not relevant for fresh init)
        
        for node in graph.nodes:
            if node.op in ["get_attr", "call_module"]:
                assert isinstance(node.target, str)
                # _copy_attr is local to graph_module.py, we can't easily call it.
                # But typically Legendre graph is empty or simple at this stage?
                # Accessing private _copy_attr via sys.modules or similar is hacky.
                pass 
                # e3nn Legendre.__init__ populates the graph AFTER super().__init__
                # So graph is empty here!
        
        self.graph = graph
        self._tracer_cls = None
        self._tracer_extras = {}
        self.meta = {}
        self._replace_hook = None
        self._create_node_hooks = []
        self._erase_node_hooks = []
    else:
        # Fallback to original to raise error
        _original_graph_module_init(self, root, graph, class_name)

torch.fx.GraphModule.__init__ = _patched_graph_module_init

try:
    e3nn.set_optimization_defaults(jit_script_fx=False)
except Exception as e:
    print(f"Warning: Could not set e3nn optimization defaults: {e}")

from numpy import dot
from MolecularDiffusion.core import Engine
from MolecularDiffusion.utils.plot_function import (
    plot_kde_distribution,
    plot_histogram_distribution,
)
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
        edge_type: str = "fully_connected",
        model_type: str = "auto",  # "auto", "egcn", or "esen"
    ):
        self.chkpt_directory = chkpt_directory
        self.atom_vocab = atom_vocab if atom_vocab is not None else ["H", "C", "N", "O", "F"]
        self.norm_factor = norm_factor
        self.solver = self._load_model(self.chkpt_directory)
        
        # self.norm_factor = self.solver.model.norm_values
        self.norm_factor =  [1,4,10] #getattr(self.solver.model, 'norm_values', [1,4,10])
        self.node_feature = node_feature 
        self.edge_type = edge_type

        self.n_dim = 3  # 3D coordinates
        
        # Auto-detect model type
        if model_type == "auto":
            # Check the underlying backbone model class
            if hasattr(self.solver.model, 'model'):
                model_class = self.solver.model.model.__class__.__name__
            else:
                model_class = self.solver.model.__class__.__name__
            
            if model_class in ["eSEN_Backbone", "eSEN_Encoder"]:
                self.model_type = "esen"
            else:
                self.model_type = "egcn"
        else:
            self.model_type = model_type
        
        self._initialize_model_attributes()

    def _load_model(self, chkpt_path):
        """
        Loads a pre-trained model from a checkpoint file.
        
        Handles both old-format Engine checkpoints (.pkl) and new Lightning checkpoints (.ckpt).
        Returns a wrapper with a .model attribute that is the Task object (with .predict method).

        Args:
            chkpt_path (str): The path to the checkpoint file.

        Returns:
            object: An object with .model attribute containing the Task in evaluation mode.
        """
        import os
        import hydra
        from omegaconf import OmegaConf
        
        chkpt_path = os.path.expanduser(chkpt_path)
        
        class SolverWrapper:
            def __init__(self, task):
                self.model = task
        
        # Try Lightning checkpoint first if .ckpt extension
        if chkpt_path.endswith('.ckpt'):
            # Pre-check checkpoint for backward compatibility
            state = torch.load(chkpt_path, map_location="cpu", weights_only=False)
            hparams = state.get("hyper_parameters", {})
            model_config = hparams.get("model_config", {})
            
            # Check if old checkpoint (missing prediction_mlp_type)
            needs_legacy_handling = isinstance(model_config, dict) and 'prediction_mlp_type' not in model_config
            
            if not needs_legacy_handling:
                # New checkpoint with prediction_mlp_type saved - use Lightning loading
                try:
                    from MolecularDiffusion.core.engine_lightning import EngineLightning
                    wrapper_lightning = EngineLightning.load_from_checkpoint(chkpt_path, map_location="cpu", strict=False)
                    solver = SolverWrapper(wrapper_lightning.task)
                    solver.model.eval()
                    if torch.cuda.is_available():
                        solver.model = solver.model.cuda()
                    return solver
                except Exception as e:
                    print(f"EngineLightning load failed: {e}. Trying manual fallback.")
            else:
                print("Old checkpoint detected (no prediction_mlp_type): using manual loading with legacy MLP")
                # Fall through to manual loading below which handles legacy properly
        
        # Load checkpoint manually
        state = torch.load(chkpt_path, map_location="cpu", weights_only=False)
        
        # Check for Lightning checkpoint structure
        if "hyper_parameters" in state:
            hparams = state.get("hyper_parameters", {})
            model_config = hparams.get("model_config", None)
            
            if model_config is not None:
                if isinstance(model_config, dict):
                    model_config = OmegaConf.create(model_config)
                
                OmegaConf.set_struct(model_config, False)
                
                # Ensure atom_vocab is set
                if self.atom_vocab is not None:
                    if 'atom_vocab' not in model_config or model_config.atom_vocab is None:
                        model_config.atom_vocab = self.atom_vocab
                
                # BACKWARD COMPATIBILITY: Old checkpoints don't have prediction_mlp_type.
                # If not present, default to 'legacy' to match old common.MLP structure.
                if 'prediction_mlp_type' not in model_config:
                    print("Old checkpoint detected: injecting prediction_mlp_type='legacy' for backward compatibility")
                    model_config.prediction_mlp_type = 'legacy'
                    if 'mlp_batch_norm' not in model_config:
                        model_config.mlp_batch_norm = True
                
                task_factory = hydra.utils.instantiate(model_config)
                task = task_factory.build()
                
                # Load weights
                state_dict = state.get("state_dict", {})
                cleaned_state_dict = {k[5:] if k.startswith("task.") else k: v for k, v in state_dict.items()}
                task.load_state_dict(cleaned_state_dict, strict=False)
                
                solver = SolverWrapper(task)
                solver.model.eval()
                if torch.cuda.is_available():
                    solver.model = solver.model.cuda()
                return solver
        
        # Check for old-format Engine checkpoint (pickled task in hyperparameters)
        if "hyperparameters" in state and "task" in state["hyperparameters"]:
            task_obj = state["hyperparameters"]["task"]
            # Check if task object itself has predict (it should be the Task class instance)
            if hasattr(task_obj, "predict"):
                # Load state dict
                if "ema_model" in state:
                    task_obj.load_state_dict(state["ema_model"], strict=False)
                elif "model" in state and isinstance(state["model"], dict):
                    task_obj.load_state_dict(state["model"], strict=False)
                
                solver = SolverWrapper(task_obj)
                solver.model.eval()
                if torch.cuda.is_available():
                    solver.model = solver.model.cuda()
                return solver
        
        # Fall back to standard Engine loading for new-format checkpoints
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
            chkpt = torch.load(self.chkpt_directory, weights_only=False)
            device = next(self.solver.model.parameters()).device
            
            # Helper to find keys
            def get_val(key, default=None):
                if "model" in chkpt and isinstance(chkpt["model"], dict) and key in chkpt["model"]:
                    return chkpt["model"][key]
                if key in chkpt:
                    return chkpt[key]
                # Lightning structure often hides these or they are buffers
                if "state_dict" in chkpt:
                    sd = chkpt["state_dict"]
                    # Try exact match or with 'task.' prefix (common in EngineLightning)
                    if key in sd:
                        return sd[key]
                    if f"task.{key}" in sd:
                        return sd[f"task.{key}"]
                
                return None

            std = get_val("std")
            weight = get_val("weight")
            mean = get_val("mean")
            
            if std is not None:
                self.solver.model.std = std.to(device)
            if weight is not None:
                self.solver.model.weight = weight.to(device)
            if mean is not None:
                self.solver.model.mean = mean.to(device)
            
            if std is None:
                print("Warning: 'std', 'weight', or 'mean' not found in checkpoint. Using default values (std=1.0, mean=0.0, weight=1.0).")
                self.solver.model.std = torch.tensor(1.0, device=device)
                self.solver.model.mean = torch.tensor(0.0, device=device)
                self.solver.model.weight = torch.tensor(1.0, device=device)

        if not hasattr(self.solver.model, 'atom_vocab'):
            self.solver.model.atom_vocab = self.atom_vocab
        if not hasattr(self.solver.model, 'node_feature'):
            self.solver.model.node_feature = self.node_feature

    def __call__(self, xh, t):
        """
        Calculates the negative energy score for a given molecular configuration.
        This method makes the class instance callable.
        
        Args:
            xh: Tensor of shape [B, N, D] where B is batch size, N is number of nodes,
                D is features (3 coords + h_dim + 1 charge)
            t: Tensor of shape [B] or [B, 1] containing timesteps
            
        Returns:
            Tensor of shape [B] containing per-sample energy scores (for gradient computation)
        """
        bs, n_nodes, _ = xh.shape
        RADIUS = 4
        device = xh.device
      
        
        # Check if model has predict_dense (GuidanceModelPredictionPointCloud)
        if hasattr(self.solver.model, 'predict_dense'):
            return self._call_dense(xh, t)
        
        
        # Handle timestep tensor
        if t.dim() == 0:
            t_values = [t.item()] * bs
        elif t.dim() == 1:
            t_values = [t[i].item() for i in range(bs)]
        else:
            t_values = [t[i, 0].item() for i in range(bs)]
        
        # Create list of graphs for proper batching
        graph_list = []
        for i in range(bs):
            # Extract per-sample data
            # NOTE: Unnormalize by multiplying (since xh is normalized latent)
            coords_i = xh[i, :, :self.n_dim] * self.norm_factor[0]
            h_i = xh[i, :, self.n_dim:-1] * self.norm_factor[1]
            charge_i = xh[i, :, -1] * self.norm_factor[2]
            
            # print(charge_i, self.norm_factor[2])
          
            # Build graph for this molecule
            if self.edge_type == "distance":
                edge_index = radius_graph(coords_i, r=RADIUS)
            else:
                # Default to fully connected
                if n_nodes > 1:
                    row_ = torch.arange(n_nodes, device=device).repeat_interleave(n_nodes)
                    col_ = torch.arange(n_nodes, device=device).repeat(n_nodes)
                    edge_index = torch.stack([row_, col_], dim=0)
                    edge_index = edge_index[:, row_ != col_] # Remove self-loops
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            
            # Create time tensor for this sample
            times_i = torch.full((n_nodes, 1), t_values[i], dtype=torch.float32, device=device)
            tags_i = torch.zeros(n_nodes, dtype=torch.long, device=device)
            
            graph_data = Data(
                x=h_i,
                pos=coords_i,
                atomic_numbers=charge_i,
                natoms=n_nodes,
                smiles=None,
                xyz=None,
                edge_index=edge_index,
                tags=tags_i,
                times=times_i,
            )
            graph_list.append(graph_data)
        
        # Create batched graph
        batch_graph = Batch.from_data_list(graph_list)
        batch_graph = batch_graph.to(device)
        
        # Add eSEN-specific fields if needed
        if self.model_type == "esen":
            batch_graph.num_atoms = torch.tensor([g.natoms for g in graph_list], device=device)
            batch_graph.token_idx = torch.zeros(batch_graph.num_nodes, dtype=torch.long, device=device)
            # eSEN expects atomic_numbers as long tensor (integer indices)
            if hasattr(batch_graph, 'atomic_numbers'):
                batch_graph.atomic_numbers = batch_graph.atomic_numbers.long()
        
        mol = {"graph": batch_graph}
        
        # Get predictions - model returns tensor of shape [B, 2] with [T1, S1] columns
        preds = self.solver.model.predict(mol, evaluate=True)
        # Handle different prediction formats
        if isinstance(preds, torch.Tensor):
            # Direct tensor output from model.predict()
            if preds.dim() == 2 and preds.size(1) >= 2:
                # Shape [B, 2] - columns are [T1, S1]
                t1_pred = preds[:, 0]  # [B]
                s1_pred = preds[:, 1]  # [B]
                
                # Compute per-sample energies
                energies = torch.stack([
                    -energy_score(s1_pred[i], t1_pred[i]) for i in range(bs)
                ])
            elif preds.dim() == 1 and preds.numel() >= 2:
                # Single sample [2] - values are [T1, S1]
                t1_pred = preds[0]
                s1_pred = preds[1]
                
                energy_single = -energy_score(s1_pred, t1_pred)
                energies = energy_single.unsqueeze(0).expand(bs)
            else:
                raise ValueError(f"Unexpected tensor shape: {preds.shape}")
                
        elif isinstance(preds, (list, tuple)) and len(preds) > 0:
            pred_tuple = preds[0]  # Get first prediction tuple
            
            if isinstance(pred_tuple, (list, tuple)) and len(pred_tuple) >= 2:
                t1_pred, s1_pred = pred_tuple[0], pred_tuple[1]
                
                if isinstance(t1_pred, torch.Tensor) and t1_pred.numel() == bs:
                    energies = torch.stack([
                        -energy_score(s1_pred[i], t1_pred[i]) for i in range(bs)
                    ])
                elif isinstance(t1_pred, torch.Tensor) and t1_pred.numel() == 1:
                    energy_single = -energy_score(s1_pred.squeeze(), t1_pred.squeeze())
                    energies = energy_single.unsqueeze(0).expand(bs)
                else:
                    energy_single = -energy_score(s1_pred, t1_pred)
                    if energy_single.dim() == 0:
                        energies = energy_single.unsqueeze(0).expand(bs)
                    else:
                        energies = energy_single
            elif isinstance(pred_tuple, torch.Tensor):
                # preds is a list containing a tensor
                return self.__call__helper_tensor(pred_tuple, bs)
            else:
                t1_pred, s1_pred = preds[0], preds[1]
                energy_single = -energy_score(s1_pred, t1_pred)
                if energy_single.dim() == 0:
                    energies = energy_single.unsqueeze(0).expand(bs)
                else:
                    energies = energy_single
        else:
            raise ValueError(f"Unexpected prediction format: {type(preds)}")
        
        return energies

    def _call_dense(self, xh, t):
        """
        Dense tensor inference path for GuidanceModelPredictionPointCloud.
        
        Uses predict_dense method which handles dense tensors directly,
        avoiding PyG Data object creation.
        """
        bs, n_nodes, _ = xh.shape
        device = xh.device
        
        # Split xh into coords, features, charges
        # NOTE: Unnormalize by multiplying (since xh is normalized latent)
        coords = xh[:, :, :self.n_dim] * self.norm_factor[0]  # (B, N, 3)
        h = xh[:, :, self.n_dim:-1] * self.norm_factor[1]  # (B, N, F)
        charges = xh[:, :, -1:] * self.norm_factor[2]  # (B, N, 1)
        
        # Combine features and charges for predict_dense
        h_with_charges = torch.cat([h, charges], dim=-1)  # (B, N, F+1)
        
        # Create node mask (all valid for full molecules)
        node_mask = torch.ones(bs, n_nodes, 1, device=device)
        
        # Handle timestep tensor
        if t.dim() == 0:
            t_tensor = torch.full((bs, 1), t.item(), device=device)
        elif t.dim() == 1:
            t_tensor = t.unsqueeze(-1)
        else:
            t_tensor = t
        
        # Call predict_dense
        preds = self.solver.model.predict_dense(coords, h_with_charges, node_mask, t_tensor)
        # Process predictions (same as __call__)
        if preds.dim() == 2 and preds.size(1) >= 2:
            t1_pred = preds[:, 0]
            s1_pred = preds[:, 1]
            energies = torch.stack([
                -energy_score(s1_pred[i], t1_pred[i]) for i in range(bs)
            ])
        elif preds.dim() == 1 and preds.numel() >= 2:
            t1_pred = preds[0]
            s1_pred = preds[1]
            energy_single = -energy_score(s1_pred, t1_pred)
            energies = energy_single.unsqueeze(0).expand(bs)
        else:
            raise ValueError(f"Unexpected tensor shape from predict_dense: {preds.shape}")
        
        return energies


class PCEnergyScore:
    """
    PointCloud-optimized Energy Score for gradient guidance.
    
    Uses GuidanceModelPredictionPointCloud's predict_dense method for efficient
    dense tensor inference, avoiding PyG Data object creation overhead.
    
    Designed for use as target_function in en_diffusion.py gradient guidance.
    """
    
    def __init__(
        self,
        chkpt_directory: str,
        norm_factor: List[float] = [1.0, 4.0, 10.0],
        n_dim: int = 3,
    ):
        """
        Initialize the PointCloud energy score function.
        
        Args:
            chkpt_directory: Path to the guidance model checkpoint
            norm_factor: Normalization factors [pos, categorical, integer]
            n_dim: Number of spatial dimensions (typically 3)
        """
        self.chkpt_directory = chkpt_directory
        self.norm_factor = norm_factor
        self.n_dim = n_dim
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the guidance model from checkpoint."""
        chkpt_path = self.chkpt_directory
        
        if chkpt_path.endswith('.ckpt'):
            # Lightning checkpoint
            from MolecularDiffusion.core.engine_lightning import EngineLightning
            wrapper = EngineLightning.load_from_checkpoint(chkpt_path, map_location="cpu", strict=False)
            self.model = wrapper.task
        elif chkpt_path.endswith('.pkl') or chkpt_path.endswith('.pth'):
            # Standard checkpoint
            checkpoint = torch.load(chkpt_path, map_location="cpu", weights_only=False)
            
            # Handle different checkpoint formats
            from MolecularDiffusion.core import Engine
            if isinstance(checkpoint, Engine):
                self.model = checkpoint.task
            elif hasattr(checkpoint, 'task'):
                self.model = checkpoint.task
            else:
                raise ValueError(f"Cannot extract model from checkpoint: {chkpt_path}")
        else:
            raise ValueError(f"Unsupported checkpoint format: {chkpt_path}")
        
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # Verify model supports predict_dense
        if not hasattr(self.model, 'predict_dense'):
            raise ValueError(
                "Model does not support predict_dense. "
                "Ensure you trained with dense_mode=True or use SFEnergyScore instead."
            )
    
    def __call__(self, xh: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute energy scores from dense input tensors.
        
        Args:
            xh: Combined coordinates and features (B, N, 3+F)
            t: Timestep tensor (B,) or (B, 1)
            
        Returns:
            energies: Energy scores per sample (B,)
        """
        bs = xh.shape[0]
        device = xh.device
        
        # Split xh into coordinates and features
        x = xh[:, :, :self.n_dim]
        h = xh[:, :, self.n_dim:]
        
        # Infer node mask from features (assume padding is all zeros)
        node_mask = (h.abs().sum(dim=-1, keepdim=True) > 1e-8).float()
        
        with torch.no_grad():
            # Call the dense prediction method
            preds = self.model.predict_dense(x, h, node_mask, t)
            
        # preds shape: (B, num_targets) where targets are typically [T1, S1]
        if preds.shape[1] >= 2:
            t1_pred = preds[:, 0]
            s1_pred = preds[:, 1]
            
            # Compute energy scores
            energies = torch.stack([
                -energy_score(s1_pred[i], t1_pred[i]) for i in range(bs)
            ])
        else:
            # Single target - use as is
            energies = -preds.squeeze(-1)
        
        return energies


if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="Compute energy scores from a CSV file.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_csv", type=str, required=False, help="Path to save the output CSV file. If not specified, overrides input_csv.")
    parser.add_argument("--threshold", type=float, help="Threshold to count entries with score higher than this value.")
    args = parser.parse_args()

    output_csv = args.output_csv if args.output_csv else args.input_csv

    df = pd.read_csv(args.input_csv)

    # Identify columns
    s1_col = next((col for col in ["S1", "S1_exc", "s1"] if col in df.columns), None)
    t1_col = next((col for col in ["T1", "T1_exc", "t1"] if col in df.columns), None)
    
    if not s1_col or not t1_col:
        print(f"Error: Could not find S1 or T1 columns. Available columns: {df.columns}")
        sys.exit(1)
        
    energy_scores = []
    for index, row in df.iterrows():
        try:
            s1_val = float(row[s1_col])
            t1_val = float(row[t1_col])
            
            # energy_score takes torch tensors
            t1_tensor = torch.tensor(t1_val)
            s1_tensor = torch.tensor(s1_val)
            
            # energy_score(x, y) where x=t1, y=s1
            score = energy_score(t1_tensor, s1_tensor).item()
            energy_scores.append(score)
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            energy_scores.append(np.nan)

    df["energy_score"] = energy_scores
    
    # Save
    df.to_csv(output_csv, index=False)
    
    # Statistics
    valid_scores = [s for s in energy_scores if not np.isnan(s)]
    if valid_scores:
        print(f"Mean energy score: {np.mean(valid_scores)}")
        print(f"Max energy score: {np.max(valid_scores)}")
        print(f"Min energy score: {np.min(valid_scores)}")
        
        if args.threshold is not None:
            count_above = sum(1 for s in valid_scores if s > args.threshold)
            portion_above = count_above / len(valid_scores)
            print(f"Entries with score > {args.threshold}: {count_above} ({portion_above:.2%})")

        # Check for condition S1 - 2*T1 > -1
        try:
             # Ensure numeric
             s1_vals = pd.to_numeric(df[s1_col], errors='coerce')
             t1_vals = pd.to_numeric(df[t1_col], errors='coerce')
             valid_mask = s1_vals.notna() & t1_vals.notna()
             
             if valid_mask.any():
                 s1_valid = s1_vals[valid_mask]
                 t1_valid = t1_vals[valid_mask]
                 condition_met = (s1_valid - 2 * t1_valid) > -1
                 count_cond = condition_met.sum()
                 portion_cond = count_cond / len(s1_valid)
                 print(f"Entries with {s1_col} - 2*{t1_col} > -1: {count_cond} ({portion_cond:.2%})")
        except Exception as e:
            print(f"Error calculating specific condition stat: {e}")

        # Plotting
        output_dir = os.path.dirname(output_csv)
        if not output_dir:
            output_dir = "."
            
        print(f"Plotting distributions to {output_dir}")
        try:
            # Drop NaNs for plotting
            plot_series = df["energy_score"].dropna()
            
            plot_kde_distribution(
                plot_series, 
                "Energy Score", 
                os.path.join(output_dir, "energy_score_kde.png")
            )
            plot_histogram_distribution(
                plot_series, 
                "Energy Score", 
                os.path.join(output_dir, "energy_score_hist.png")
            )
        except Exception as e:
            print(f"Error plotting distributions: {e}")
    else:
        print("No valid energy scores computed.")
