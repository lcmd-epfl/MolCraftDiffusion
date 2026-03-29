"""
TABASCO model integration with MolecularDiffusion data pipeline.

This module provides adapters to convert between PointCloudDataset format
and TABASCO's TensorDict format, plus a task wrapper for training.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from collections import Counter


class TabascoNodeDistribution:
    """
    Node distribution sampler compatible with EDM's node_dist_model interface.
    Samples molecule sizes from a histogram of atom counts.
    """
    
    def __init__(self, data_stats: dict):
        self.histogram = data_stats.get("num_atoms_histogram", {})
        # Compatibility with EDM interface
        self.n_node_dist = self.histogram
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample molecule sizes from the histogram distribution.
        
        Args:
            n_samples: Number of sizes to sample
            
        Returns:
            Tensor of shape (n_samples,) with sampled molecule sizes
        """
        if not self.histogram:
            # Fallback: sample uniformly from 5 to 29 (QM9-like range)
            sizes = [random.randint(5, 29) for _ in range(n_samples)]
        else:
            sizes = random.choices(
                list(self.histogram.keys()),
                weights=list(self.histogram.values()),
                k=n_samples
            )
        return torch.tensor(sizes, dtype=torch.long)

try:
    from tensordict import TensorDict
except ImportError:
    raise ImportError(
        "TensorDict is required for TABASCO integration. "
        "Install with: pip install tensordict"
    )

from MolecularDiffusion.modules.models.tabasco.flow_model import FlowMatchingModel
from MolecularDiffusion.modules.layers.tabasco.transformer_module import TransformerModule
from MolecularDiffusion.modules.models.tabasco.flow.interpolate import SDEMetricInterpolant, DiscreteInterpolant


class PointCloudToTensorDictAdapter(nn.Module):
    """
    Lightweight converter from PointCloud dict format to TABASCO TensorDict.
    
    Converts:
    - coords: (B, N, 3) → coords: (B, N, 3) [unchanged]
    - node_mask: (B, N) with 1=real → padding_mask: (B, N) with 1=padded [inverted]
    - charges: (B, N) integers → atomics: (B, N, num_types) one-hot
    """
    
    def __init__(self, num_atom_types: int):
        super().__init__()
        self.num_atom_types = num_atom_types
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> TensorDict:
        """
        Convert PointCloud batch to TABASCO TensorDict.
        
        Args:
            batch: Dictionary with keys:
                - coords: (B, N, 3) padded coordinates
                - node_mask: (B, N) with 1=real atom, 0=padded
                - charges: (B, N) atomic numbers (integers)
                - natoms: (B,) number of real atoms per molecule
        
        Returns:
            TensorDict with keys:
                - coords: (B, N, 3)
                - atomics: (B, N, num_atom_types) one-hot encoded
                - padding_mask: (B, N) with 1=padded, 0=real
        """
        # Use pre-computed one-hot features if available (handles mapping correctly)
        if "node_feature" in batch:
             atomics = batch["node_feature"]
        elif "node_features" in batch:
            atomics = batch["node_features"]
        elif "x" in batch:
            atomics = batch["x"] 
        else:
            # Fallback (risky if charges are atomic numbers > num_atom_types)
            atomics = F.one_hot(
                batch["charges"].long(), 
                num_classes=self.num_atom_types
            ).float()
        
        # Invert mask: TABASCO uses padding_mask where 1=padded
        # PointCloud uses node_mask where 1=real
        padding_mask = (batch["node_mask"] == 0)
        
        return TensorDict({
            "coords": batch["coords"],
            "atomics": atomics,
            "padding_mask": padding_mask
        }, batch_size=batch["coords"].size(0))


class TensorDictToPointCloudAdapter(nn.Module):
    """
    Convert TABASCO TensorDict output back to PointCloud format.
    
    Converts:
    - coords: (B, N, 3) → coords: (B, N, 3) [unchanged]
    - atomics: (B, N, atom_dim) one-hot/logits → charges: (B, N) integers
    - padding_mask: (B, N) with 1=padded → node_mask: (B, N) with 1=real [inverted]
    """
    
    def forward(self, tensor_dict: TensorDict) -> Dict[str, torch.Tensor]:
        """
        Convert TABASCO output to PointCloud format.
        
        Args:
            tensor_dict: TensorDict with:
                - coords: (B, N, 3)
                - atomics: (B, N, atom_dim) one-hot or logits
                - padding_mask: (B, N) with 1=padded
        
        Returns:
            Dictionary with:
                - coords: (B, N, 3)
                - charges: (B, N) atomic numbers
                - node_mask: (B, N) with 1=real, 0=padded
                - natoms: (B,) count of real atoms
        """
        # Convert one-hot/logits back to atomic numbers
        charges = tensor_dict["atomics"].argmax(dim=-1)
        
        # Invert padding mask back to node mask
        node_mask = (~tensor_dict["padding_mask"]).int()
        
        # Count real atoms per molecule
        natoms = node_mask.sum(dim=1)
        
        return {
            "coords": tensor_dict["coords"],
            "charges": charges,
            "node_mask": node_mask,
            "natoms": natoms
        }



class ModelTaskFactory:
    """
    Factory to satisfy train.py instantiation pattern.
    Matches conventions from tasks_egcl.py and tasks_esen.py.
    """
    def __init__(
        self,
        task_type: str,
        transformer_config: dict,
        coords_interpolant_config: dict,
        atomics_interpolant_config: dict,
        flow_matching_config: dict,
        num_atom_types: int,
        dataset_stats: dict,
        atom_vocab: Optional[list] = None,
        train_set: Optional[torch.utils.data.Dataset] = None,
        **kwargs
    ):
        self.task_type = task_type
        # Configuration parameters
        self.transformer_config = transformer_config
        self.coords_interpolant_config = coords_interpolant_config
        self.atomics_interpolant_config = atomics_interpolant_config
        self.flow_matching_config = flow_matching_config
        self.num_atom_types = num_atom_types
        self.dataset_stats = dataset_stats
        self.atom_vocab = atom_vocab or kwargs.get("atom_vocab", None)
        self.train_set = train_set
        self.kwargs = kwargs

    def compute_dataset_stats(self, dataset):
        """Compute missing dataset statistics from the training set."""
        print(f"Computing dataset statistics from {len(dataset)} samples...")
        
        num_atoms_list = []
        smiles_list = []
        
        # Try to access internal lists for speed if available (assuming PointCloudDataset)
        if hasattr(dataset, 'smiles_list') and hasattr(dataset, 'n_atoms'):
            print("Using cached lists from dataset.")
            smiles_list = dataset.smiles_list
            num_atoms_list = dataset.n_atoms
        else:
            # Iterate (slower)
            for i in range(len(dataset)):
                item = dataset[i]
                
                # Get atom count
                if 'natoms' in item:
                    n = item['natoms']
                    if torch.is_tensor(n): n = n.item()
                    num_atoms_list.append(n)
                elif 'node_mask' in item:
                    n = item['node_mask'].sum().item()
                    num_atoms_list.append(n)
                elif hasattr(item, 'natoms'): # Data object
                     num_atoms_list.append(item.natoms)

                # Get SMILES
                if 'smiles' in item:
                    smiles_list.append(item['smiles'])
                elif hasattr(item, 'smiles'):
                    smiles_list.append(item.smiles)
        
        # Compute histogram
        # Convert keys to int (YAML compatibility) and values to probability or counts.
        # TabascoNodeDistribution expects counts or weights.
        counts = Counter(num_atoms_list)
        histogram = {int(k): int(v) for k, v in counts.items()}
        
        # Update stats
        if not self.dataset_stats.get("atom_count_histogram"):
             self.dataset_stats["atom_count_histogram"] = histogram
             print(f"Computed atom_count_histogram: found {len(histogram)} unique sizes.")
        
        if not self.dataset_stats.get("all_smiles"):
             # Filter None values
             valid_smiles = [s for s in smiles_list if s]
             self.dataset_stats["all_smiles"] = valid_smiles
             print(f"Collected {len(valid_smiles)} SMILES strings.")
             
        if not self.dataset_stats.get("max_atoms"):
             if num_atoms_list:
                 self.dataset_stats["max_atoms"] = max(num_atoms_list)
                 print(f"Computed max_atoms: {self.dataset_stats['max_atoms']}")

    def build(self):
        """Build and return the TabascoDiffusionTask."""
        
        # Check if we need to compute stats
        needs_stats = (
            not self.dataset_stats.get("atom_count_histogram") or 
            not self.dataset_stats.get("all_smiles") or
            not self.dataset_stats.get("max_atoms")
        )
        
        if needs_stats:
            if self.train_set is not None:
                self.compute_dataset_stats(self.train_set)
            else:
                 print("WARNING: Dataset stats missing and no train_set provided. Using defaults/placeholders. This may affect generation quality.")

        self.task = TabascoDiffusionTask(
            transformer_config=self.transformer_config,
            coords_interpolant_config=self.coords_interpolant_config,
            atomics_interpolant_config=self.atomics_interpolant_config,
            flow_matching_config=self.flow_matching_config,
            num_atom_types=self.num_atom_types,
            dataset_stats=self.dataset_stats,
            atom_vocab=self.atom_vocab
        )
        return self.task


class TabascoDiffusionTask(nn.Module):
    """
    TABASCO flow-matching diffusion model integrated with MolecularDiffusion.
    """
    
    def __init__(
        self,
        transformer_config: dict,
        coords_interpolant_config: dict,
        atomics_interpolant_config: dict,
        flow_matching_config: dict,
        num_atom_types: int,
        dataset_stats: dict,
        atom_vocab: Optional[list] = None
    ):
        super().__init__()
        
        # Data format adapters
        self.to_tensordict = PointCloudToTensorDictAdapter(num_atom_types)
        self.to_pointcloud = TensorDictToPointCloudAdapter()
        
        # Build TABASCO components
        transformer = TransformerModule(**transformer_config)
        coords_interpolant = SDEMetricInterpolant(**coords_interpolant_config)
        atomics_interpolant = DiscreteInterpolant(**atomics_interpolant_config)
        
        # Assemble flow matching model
        self.tabasco_model = FlowMatchingModel(
            net=transformer,
            coords_interpolant=coords_interpolant,
            atomics_interpolant=atomics_interpolant,
            **flow_matching_config
        )
        
        # Set dataset statistics for unconditional sampling
        self.tabasco_model.set_data_stats({
            'max_num_atoms': dataset_stats.get('max_atoms', 100),
            'num_atoms_histogram': dataset_stats.get('atom_count_histogram', {}),
            'spatial_dim': 3,
            'atom_dim': num_atom_types,
            'all_smiles': dataset_stats.get('all_smiles', [])
        })
        
        self.atom_vocab = atom_vocab
        self.num_atom_types = num_atom_types
        self.task_type = "diffusion_tabasco"
        self._dataset_stats = dataset_stats
        
        # EDM compatibility attributes
        self.prop_dist_model = None  # No conditional sampling by default
        self.max_n_nodes = dataset_stats.get('max_atoms', 100)
        self._node_dist_model = None  # Lazy initialized

    @property
    def model(self):
        """tasks_generate.py compatibility: exposes self as the model interface."""
        return self

    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        Training forward pass.
        
        Args:
            batch: PointCloud format batch from dataloader
        
        Returns:
            loss: Scalar training loss
            stats: Dictionary of training statistics
        """
        # Convert PointCloud dict → TensorDict
        tensor_batch = self.to_tensordict(batch)
        
        # Forward through TABASCO
        loss, stats = self.tabasco_model(tensor_batch, compute_stats=True)
        
        return loss, stats
    
    def predict_and_target(self, batch: Dict[str, torch.Tensor]):
        """
        Evaluation pass for Engine compatibility.
        
        Args:
            batch: PointCloud format batch from dataloader
            
        Returns:
            pred: Loss tensor (B,) or scalar
            target: Dummy tensor of same shape
        """
        loss, stats = self.forward(batch)
        
        # Ensure loss is an appropriate shape for concatenation in Engine.evaluate
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
            
        dummy_target = torch.zeros_like(loss)
        return loss, dummy_target

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute evaluation metrics from aggregated predictions.
        
        Args:
            pred: Concatenated losses from predict_and_target
            target: Dummy targets
            
        Returns:
            Dictionary of metrics
        """
        return {"val_loss": pred.mean()}
    
    def sample(
        self, 
        batch_size: Optional[int] = None, 
        nodesxsample: Optional[torch.Tensor] = None,  # EDM compatibility
        num_steps: int = 100, 
        batch: Optional[Dict[str, torch.Tensor]] = None,
        return_trajectories: bool = False,
        **kwargs  # Accept additional kwargs for compatibility
    ):
        """
        Generate molecules via sampling.
        
        Args:
            batch_size: Number of molecules to generate (if batch is None)
            nodesxsample: Tensor of molecule sizes (EDM compatibility, used to infer batch_size)
            num_steps: Number of denoising steps
            batch: Optional reference batch for conditional generation
            return_trajectories: If True, return intermediate states
        
        Returns:
            Tuple of (one_hot, charges, coords, node_mask) matching EDM interface
            - one_hot: (B, N, num_atom_types) one-hot encoded atom types
            - charges: (B, N) atomic numbers
            - coords: (B, N, 3) positions
            - node_mask: (B, N) mask (1=real atom, 0=padding)
        """
        # EDM compatibility: derive batch_size and padding_mask from nodesxsample if provided
        if nodesxsample is not None:
            batch_size = len(nodesxsample)
            max_atoms = nodesxsample.max().item()
            # Tabasco uses padding_mask where 1=padded, 0=real
            padding_mask = torch.arange(max_atoms, device=self.device)[None, :] >= nodesxsample[:, None].to(self.device)
            
            # Construct a dummy batch to guide the shapes in the underlying model
            batch = TensorDict({
                "padding_mask": padding_mask,
                "coords": torch.zeros(batch_size, max_atoms, 3, device=self.device),
                "atomics": torch.zeros(batch_size, max_atoms, self.num_atom_types, device=self.device)
            }, batch_size=batch_size)
        else:
            # Convert input batch if provided
            batch = self.to_tensordict(batch) if batch is not None else None

        
        # Sample from TABASCO
        if return_trajectories:
            samples, trajectories = self.tabasco_model.sample(
                batch=batch,
                batch_size=batch_size,
                num_steps=num_steps,
                return_trajectories=True
            )
            # Convert trajectories back to PointCloud format
            trajectories_pc = [self.to_pointcloud(traj) for traj in trajectories]
            pointcloud_result = self.to_pointcloud(samples)
        else:
            samples = self.tabasco_model.sample(
                batch=batch,
                batch_size=batch_size,
                num_steps=num_steps
            )
            # Convert TensorDict → PointCloud format
            pointcloud_result = self.to_pointcloud(samples)
        
        # Convert dictionary to tuple for EDM compatibility
        # EDM expects: (one_hot, charges, coords, node_mask)
        # We need to convert charges to one-hot encoding
        charges = pointcloud_result["charges"]
        one_hot = torch.nn.functional.one_hot(
            charges, num_classes=self.num_atom_types
        ).float()
        coords = pointcloud_result["coords"]
        node_mask = pointcloud_result["node_mask"]
        
        # Return in EDM format: (one_hot, charges, coords, node_mask)
        return one_hot, charges, coords, node_mask

    
    @property
    def node_dist_model(self):
        """Return a node distribution sampler (EDM compatibility)."""
        if self._node_dist_model is None:
            self._node_dist_model = TabascoNodeDistribution(
                self.tabasco_model.data_stats
            )
        return self._node_dist_model
    
    @property
    def n_node_dist(self):
        """Direct access to node distribution histogram (EDM compatibility).
        
        Returns the histogram dictionary mapping number of atoms to counts.
        This provides the standard interface expected by GenerativeFactory.
        """
        return self.node_dist_model.n_node_dist

    
    @property
    def device(self):
        """Get model device."""
        return next(self.parameters()).device
