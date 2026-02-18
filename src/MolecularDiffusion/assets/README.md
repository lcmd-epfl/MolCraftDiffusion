# MolecularDiffusion Assets

This directory stores auxiliary files required for certain data processing and analysis tasks.

## Required Files

### SA Score
- **File**: `fpscores.pkl.gz`
- **Purpose**: Used for calculating Synthetic Accessibility (SA) scores.
- **Source**: Standard RDKit SA score data file.

### SCScore
- **Directory**: `scscore/`
- **Files**:
    - `scscore/models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz` (Model weights)
    - `scscore/scscore.py` (Script, if not installed as a package)
- **Purpose**: Used for calculating Synthetic Complexity (SC) scores.
- **Source**: [SCScore GitHub Repository](https://github.com/connorcoley/scscore)

## Usage
The CLI tools will automatically look for these files in this directory. If they are missing, the corresponding score calculations will be skipped with a warning.

