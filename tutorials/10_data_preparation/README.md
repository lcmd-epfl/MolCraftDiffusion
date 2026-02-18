# Tutorial 10: Data Preparation CLI - Dataset Management & Augmentation

This tutorial covers the **data** module, which provides utilities for compiling datasets, featurizing molecules, and performing data augmentation.

---

## Overview

The data module includes four main command groups:

| Command Group | Description |
|---------------|-------------|
| `prepare` | Compile data to ASE DBs, annotate, and generate molecular blocks |
| `featurize` | Convert 3D structures into Morgan or SOAP vectors |
| `augment` | Charge, coordinate distortion, and size balancing |
| `ase-ops` | ASE database operations: merge, split, sample, inspect |

Access the CLI with:
```bash
MolCraftDiff data --help
```

---

## Part 1: Data Preparation

Compile various formats into unified ASE databases and annotate them.

### Compilation

Compile XYZ files or NumPy arrays into an ASE database.

```bash
# From XYZ directory
MolCraftDiff data prepare compile -s xyz_dir/ -d dataset.db

# From NumPy arrays with metadata CSV
MolCraftDiff data prepare compile -s coords.npy -n natoms.npy -c metadata.csv -d dataset.db
```

### Annotation

Add custom metadata tags to existing ASE databases.

```bash
MolCraftDiff data prepare annotate -d dataset.db -t group -v training
```

### Molecular Blocks & SMILES

Generate RDKit-compatible molecular blocks and SMILES from structures.

```bash
MolCraftDiff data prepare generate-blocks -s dataset.db -sd output.sdf --method hybrid
```

---

## Part 2: Featurization

Convert 3D structures into fixed-length vectors for downstream tasks or guidance.

### Morgan Fingerprints

Convert structures to Morgan fingerprints (requires SMILES generation).

```bash
MolCraftDiff data featurize -m morgan -i dataset.db -o features/ --radius 2 --nbits 2048
```

### SOAP Descriptors

Compute Smooth Overlap of Atomic Positions (SOAP) descriptors.

```bash
MolCraftDiff data featurize -m soap -i xyz_dir/ -o features/ --rcut 5.0 --nmax 8 --lmax 6
```

---

## Part 3: Data Augmentation

Increase dataset diversity via structural and property transformations.

### Charge Augmentation

Randomly modify molecular charges by adding/removing hydrogens.

```bash
MolCraftDiff data augment charge -i dataset.db -o augmented.db --max-h 1 --db
```

### Coordinate Distortion

Apply random Gaussian noise to atomic coordinates.

```bash
MolCraftDiff data augment distortion -i xyz_dir/ -o noisy_xyz/ --sigma 0.1
```

### Size Balancing

Balance the distribution of molecule sizes in a dataset.

```bash
MolCraftDiff data augment size -i dataset.db -o balanced.db --s-start 60 --t-start 50000
```

---

## Part 4: ASE Database Operations

Utilities for managing and inspecting ASE datasets.

### Merging & Splitting

```bash
# Merge multiple DBs
MolCraftDiff data ase-ops merge -i db_dir/ -o merged.db

# Split a DB into N parts
MolCraftDiff data ase-ops split -d dataset.db -o splits/ -n 5
```

### Sampling

Create representative subsets of large databases.

```bash
# Sample 10% of entries
MolCraftDiff data ase-ops sample -i dataset.db -o subset.db --fraction 0.1

# Sample exactly 1000 entries
MolCraftDiff data ase-ops sample -i dataset.db -o subset.db --number 1000
```

### Inspection

Check database contents and plot property distributions.

```bash
MolCraftDiff data ase-ops inspect -d dataset.db --limit 10 --output plots/
```

---

## Example Workflow

A typical dataset preparation workflow:

```bash
# 1. Compile XYZ files to ASE DB
MolCraftDiff data prepare compile -s data/xyz/ -d data/raw.db

# 2. Add metadata and generate SMILES
MolCraftDiff data prepare annotate -d data/raw.db -t dataset -v my_project
MolCraftDiff data prepare generate-blocks -s data/raw.db -sd data/mols.sdf

# 3. Augment with coordinate noise
MolCraftDiff data augment distortion -i data/raw.db -o data/augmented.db --sigma 0.05

# 4. Inspect distributions
MolCraftDiff data ase-ops inspect -d data/augmented.db --output data/plots/

# 5. Featurize for guidance training
MolCraftDiff data featurize -m soap -i data/augmented.db -o data/features/
```
