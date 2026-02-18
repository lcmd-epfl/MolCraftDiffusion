# Data Module Guide

This guide explains the `DataModule` class and its supported data formats, feature types, and preprocessing workflow.

---

## 1. Supported Data Formats

The data module supports three input formats:

| Format | Method | Description |
|--------|--------|-------------|
| **CSV + XYZ** | `load_csv` | CSV file with metadata + directory of `.xyz` structure files |
| **CSV + NPY** | `load_csv_npy` | CSV file + NumPy arrays for coordinates and atom counts |
| **ASE Database** | `load_db` | ASE `.db` files with embedded structures and properties |

### CSV + XYZ Format
```yaml
data:
  filename: data/molecules.csv  # CSV with 'filename' and 'smiles' columns
  xyz_dir: data/xyz/            # Directory containing {filename}.xyz files
```

### ASE Database Format
```yaml
data:
  ase_db_path: data/molecules.db  # ASE database file
```

---

## 2. Data Types

Two molecular representations are supported:

| Type | Class | Description |
|------|-------|-------------|
| `pointcloud` | `PointCloudDataset` | Dense tensor format with padding, used by EDM-style models |
| `pyg` | `GraphDataset` | PyG graph format with edge indices, used by GNN-based models |

Set via config:
```yaml
data:
  data_type: pointcloud  # or "pyg"
```

---

## 3. Node Featurization

Node features are configured via `node_feature_choice`. The type determines which features are computed:

### Geometric Features (String)

For xyz/csv/npy loaders - computed from coordinates only:

| Name | Alias | Dim | Description |
|------|-------|-----|-------------|
| `topological` | `atom_topological` | 7 | Graph topology features |
| `geometric` | `atom_geom` | 6 | Geometric + SASA features |
| `geometric_full` | `atom_geom_v2` | 10 | Extended set with angles |
| `geometric_fast` | `atom_geom_v2_trun` | 8 | Fast subset (no SASA) |
| `geometric_hybrid` | `atom_geom_opt` | 3 | Minimal optimized set |

#### `topological` (7 features)
| # | Feature | Description |
|---|---------|-------------|
| 1 | `valence_electrons` | Number of valence electrons |
| 2 | `electronegativity` | Pauling electronegativity |
| 3 | `covalent_radius` | Covalent radius (Å) |
| 4 | `degree` | Number of bonded neighbors |
| 5 | `closeness_centrality` | Graph closeness centrality |
| 6 | `betweenness_centrality` | Graph betweenness centrality |
| 7 | `community_id` | Modularity community assignment |

#### `geometric` (6 features)
| # | Feature | Description |
|---|---------|-------------|
| 1 | `degree` | Number of bonded neighbors |
| 2 | `valence_electrons` | Number of valence electrons |
| 3 | `electronegativity` | Pauling electronegativity |
| 4 | `sasa_volume` | Solvent-accessible volume |
| 5 | `sasa_surface` | Solvent-accessible surface area |
| 6 | `covalent_radius` | Covalent radius (Å) |

#### `geometric_full` (10 features)
| # | Feature | Description |
|---|---------|-------------|
| 1 | `valence_electrons` | Number of valence electrons |
| 2 | `electronegativity` | Pauling electronegativity |
| 3 | `covalent_radius` | Covalent radius (Å) |
| 4 | `degree` | Number of bonded neighbors |
| 5 | `closeness_centrality` | Graph closeness centrality |
| 6 | `betweenness_centrality` | Graph betweenness centrality |
| 7 | `community_id` | Modularity community assignment |
| 8 | `sasa_volume` | Solvent-accessible volume |
| 9 | `sasa_surface` | Solvent-accessible surface area |
| 10 | `avg_bond_angle` | Average bond angle (degrees) |

#### `geometric_fast` (8 features)
| # | Feature | Description |
|---|---------|-------------|
| 1 | `valence_electrons` | Number of valence electrons |
| 2 | `electronegativity` | Pauling electronegativity |
| 3 | `covalent_radius` | Covalent radius (Å) |
| 4 | `degree` | Number of bonded neighbors |
| 5 | `closeness_centrality` | Graph closeness centrality |
| 6 | `betweenness_centrality` | Graph betweenness centrality |
| 7 | `community_id` | Modularity community assignment |
| 8 | `avg_bond_angle` | Average bond angle (degrees) |

#### `geometric_hybrid` (3 features)
| # | Feature | Description |
|---|---------|-------------|
| 1 | `valence_electrons` | Number of valence electrons |
| 2 | `degree` | Number of bonded neighbors |
| 3 | `hybridization` | Predicted hybridization (from angles) |

```yaml
data:
  node_feature_choice: geometric_fast
```

### RDKit Scalar Features (List)

For `load_db` only - requires mol_block in database:

| Feature | Description |
|---------|-------------|
| `degree` | Atom degree (number of bonds) |
| `formal_charge` | Formal charge |
| `hybridization` | sp/sp2/sp3/etc |
| `is_aromatic` | Aromaticity flag |
| `valence` | Total valence |

```yaml
data:
  ase_db_path: data/molecules.db
  node_feature_choice: ['degree', 'formal_charge', 'hybridization']
```

### One-Hot Encoding

Controlled separately via `use_ohe_feature`:
```yaml
data:
  use_ohe_feature: true  # One-hot encode atom types
  atom_vocab: [H, C, N, O, F]  # Vocabulary for encoding
```

---

## 4. Feature Computation Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    NodeFeaturizer                           │
├─────────────────────────────────────────────────────────────┤
│  Input: atom_symbols, charges (Z), coordinates              │
│                                                             │
│  ┌─────────────────┐   ┌─────────────────────────────────┐  │
│  │  compute_ohe()  │ + │  compute_geom()                 │  │
│  │  One-hot encode │   │  Distance/angle-based features  │  │
│  └────────┬────────┘   └────────────────┬────────────────┘  │
│           │                             │                   │
│           └──────────┬──────────────────┘                   │
│                      ▼                                      │
│              featurize_all()                                │
│              [OHE | Geometric Features]                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Preprocessing & Caching

Processed datasets are cached as `.pt` files:

```
data/processed_data_{dataset_name}.pt
```

To force reprocessing, delete the cached file:
```bash
rm data/processed_data_qm9.pt
```

---

## 6. Configuration Reference

```yaml
data:
  _target_: MolecularDiffusion.runmodes.train.DataModule
  
  # Data source (choose one)
  filename: data/molecules.csv
  xyz_dir: data/xyz/
  # OR
  ase_db_path: data/molecules.db
  
  # Data format
  data_type: pointcloud    # pointcloud or pyg
  dataset_name: qm9        # Used for cache filename
  
  # Molecule constraints
  max_atom: 29
  with_hydrogen: true
  forbidden_atom: []
  atom_vocab: [H, C, N, O, F]
  
  # Featurization
  use_ohe_feature: true
  node_feature_choice: null  # str for geom, list for RDKit
  
  # Training
  batch_size: 64
  train_ratio: 0.8
  data_efficient_collator: true
  
  # Graph-specific (pyg only)
  edge_type: fully_connected
  radius: 4.0
  n_neigh: 5
```

---

## 7. Loader Feature Restrictions

| Loader | String Features | List Features (RDKit) |
|--------|-----------------|----------------------|
| `load_xyz` | ✓ Geometric | ✗ Not supported |
| `load_csv` | ✓ Geometric | ✗ Not supported |
| `load_npy` | ✓ Geometric | ✗ Not supported |
| `load_db` | ✓ Geometric | ✓ Requires mol_block |

---

## 8. Example Configurations

### Minimal (OHE only)
```yaml
data:
  filename: data/qm9.csv
  xyz_dir: data/xyz/
  use_ohe_feature: true
  node_feature_choice: null
```

### With Geometric Features
```yaml
data:
  filename: data/qm9.csv
  xyz_dir: data/xyz/
  use_ohe_feature: true
  node_feature_choice: geometric_fast
```

### From ASE DB with RDKit Features
```yaml
data:
  ase_db_path: data/molecules.db
  use_ohe_feature: true
  node_feature_choice: ['degree', 'hybridization', 'is_aromatic']
```
