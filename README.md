<p align="center">
  <img src="./images/logo.png" alt="MolCraftDiffusion" width="480"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/molcraftdiffusion/"><img src="https://img.shields.io/pypi/v/molcraftdiffusion" alt="PyPI"/></a>
  <a href="https://chemrxiv.org/engage/chemrxiv/article-details/6909e50fef936fb4a23df237"><img src="https://img.shields.io/badge/Paper-ChemRxiv-blue" alt="Paper"/></a>
  <a href="https://doi.org/10.5281/zenodo.19511401"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.19511401.svg" alt="DOI"/></a>
  <a href="https://huggingface.co/pregH/MolecularDiffusion"><img src="https://img.shields.io/badge/Weights-HuggingFace-yellow" alt="Weights"/></a>
  <a href="https://huggingface.co/pregH/MolecularDiffusion"><img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow" alt="Dataset"/></a>
  <a href="https://preghosh.github.io/MolCraftDiffusion/"><img src="https://img.shields.io/badge/Docs-blue" alt="Docs"/></a>
  <a href="https://huggingface.co/spaces/pregH/MolCraftDiffusion-demo"><img src="https://img.shields.io/badge/Demo-HuggingFace-green" alt="Demo"/></a>
</p>

---

Three-dimensional molecular generative models assign atoms explicit Cartesian coordinates, enabling generation to be conditioned on both geometric (steric, shape) and physicochemical constraints, providing a more physically meaningful route to molecular discovery than string- or graph-based representations. The field, however, remains fragmented: implementations are scattered across incompatible repositories and evaluation protocols, impeding reproducibility and controlled comparison across methods.

MolCraftDiffusion is a modular, extensible platform for building, deploying, and evaluating 3D molecular diffusion models in computational chemistry. Its layered architecture decouples core training logic from model definitions and task implementations, so new generative architectures, guidance strategies, and evaluation metrics integrate with minimal changes to the codebase. Efficient pre-training is achieved through **curriculum learning**: a progressive chemical complexity ordering applied to datasets compiled from multiple sources, circumventing the cost of full retraining in downstream applications. Guided generation is supported via structure-directed mechanisms (**inpainting** for systematic structural variant exploration, **outpainting** for fragment extension) and property-directed mechanisms (gradient-based and classifier-free guidance). The platform's extensibility is demonstrated by integrating three architecturally distinct models from the literature (TABASCO, ADiT, and ShEPhERD), each without modifications to the core codebase, supporting applications from virtual library construction to inverse molecular design.

<p align="center">
  <img src="./images/overview.png" alt="workflow" width="700"/>
</p>

## Features

| | |
|---|---|
| **3D-native generation** | Models trained directly in Cartesian space; geometric validity by construction, not augmentation |
| **Extensible architecture** | Multiple backbone families included; adding a new model is a single sub-package drop-in |
| **Steerable generation** | Guide outputs toward target properties or structural constraints without retraining |
| **End-to-end pipeline** | Raw data through training to post-generation analysis, with no glue scripts needed |
| **Unified CLI** | `train · generate · predict · analyze · data`, all from one `MolCraftDiff` entry point |
| **Built-in analysis suite** | Geometry optimization, validity metrics, quantum-chemical descriptors, and featurization |

## Installation

```bash
# Create environment
conda create -n molcraft python=3.11 -y
conda activate molcraft
```

**GPU / CUDA:**
```bash
pip install molcraftdiffusion[gpu] \
    --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

**CPU-only:**
```bash
pip install molcraftdiffusion[cpu] \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --find-links https://data.pyg.org/whl/torch-2.6.0+cpu.html
```

**Optional feature groups:**
```bash
pip install 'molcraftdiffusion[data]'     # data prep, augmentation, SOAP featurization
pip install 'molcraftdiffusion[analyze]'  # metrics, xyz2mol, xtb-electronic

# xTB — must be installed via conda, not pip
conda install -c conda-forge xtb==6.7.1 -y
conda install xtb-python -y
```

> Commands that require optional packages exit with an installation hint rather than crashing.

### UMA featurization backend (optional)

`MolCraftDiff analyze featurize --backend uma` uses a pretrained UMA model. fairchem is **not** a pip dependency; vendor it manually:

```bash
git clone https://github.com/pregHosh/fairchem fairchem
```

Download `uma-s-1p2.pt` from [Hugging Face](https://huggingface.co/pregH/MolecularDiffusion) and place it at `training_outputs/uma-s-1p2.pt` (or pass `--checkpoint /path/to/checkpoint.pt`). The default SOAP backend has no such requirement.

### Development install

```bash
git clone https://github.com/pregHosh/MolCraftDiffusion
cd MolCraftDiffusion
pip install -e .[gpu] --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html

pip install -e '.[data]'     # optional
pip install -e '.[analyze]'  # optional
```

## Usage

Pre-trained diffusion models are available on [Hugging Face](https://huggingface.co/pregHosh/MolecularDiffusion). Starting from a pretrained checkpoint is recommended for downstream tasks.

### CLI

Run all commands from the repo root. Every command accepts a YAML config name and supports Hydra-style key overrides:

```
MolCraftDiff [COMMAND] [CONFIG_NAME] [key=value ...]
```

| Command | Description |
|---|---|
| `train` | Train a diffusion, regression, or guidance model |
| `generate` | Sample molecules from a trained model |
| `predict` | Run property prediction |
| `eval-predict` | Evaluate prediction results |
| `analyze` | Post-process and evaluate generated molecules |
| `data` | Data preparation and augmentation utilities |

```bash
MolCraftDiff train   example_diffusion_config
MolCraftDiff generate my_generation_config
MolCraftDiff predict  my_prediction_config
MolCraftDiff data prepare compile -s data_dir/ -d dataset.db

MolCraftDiff --help         # all commands
MolCraftDiff train --help   # per-command help
```

### Analysis & Post-processing

```bash
MolCraftDiff analyze optimize  generated_molecules/                              # GFN-xTB geometry optimization
MolCraftDiff analyze metrics   generated_molecules/                              # validity and connectivity
MolCraftDiff analyze compare   generated_molecules/                              # RMSD, energy diff, bonds/angles
MolCraftDiff analyze xyz2mol   generated_molecules/                              # XYZ → SMILES + fingerprints
MolCraftDiff analyze featurize generated_molecules/                              # SOAP feature vectors (default)
MolCraftDiff analyze featurize generated_molecules/ --backend uma --device cuda  # UMA backbone embeddings
```

## Visualization

- [3DMolViewer](https://github.com/pregHosh/3DMolViewer): interactive 3D property visualization
- [V](https://github.com/briling/v): lightweight X11 molecular viewer

## Tutorials

Full tutorials are at **https://preghosh.github.io/MolCraftDiffusion/**

## Project Structure

```
├── .project-root
├── justfile
├── pyproject.toml
└── src/MolecularDiffusion/
    ├── cli/                    # Click entry points (train, generate, predict, analyze, data)
    ├── configs/                # Hydra config trees (tasks, data, trainer, logger, hydra, interference)
    ├── core/                   # Training engine (PyTorch Lightning wrapper, callbacks, logging)
    ├── data/                   # Dataset, dataloader, and featurization components
    ├── modules/
    │   ├── layers/             # Reusable equivariant building blocks (EGCL, Equiformer v2, …)
    │   │                       #   Add a new layer family here; wire it into a model below.
    │   ├── models/             # One sub-package per architecture:
    │   │   ├── en_diffusion/   #   EDM — E(n)-equivariant diffusion (default backbone)
    │   │   ├── ldm/            #   Latent diffusion model (Equiformer encoder/decoder + VAE)
    │   │   ├── tabasco/        #   TABASCO flow-matching architecture
    │   │   ├── shepherd_arch/  #   Shepherd — bundles its own equiformer_v2 variant
    │   │   └── <new_arch>/     #   Drop a new architecture here; register it in configs/tasks/
    │   └── tasks/              # Lightning modules that bind a model to a training objective
    │                           #   (diffusion, regression, guidance, pharmacophore, SSL, …)
    ├── runmodes/               # Run-mode logic (train, generate, analyze, data preparation)
    └── utils/                  # Geometry, diffusion math, graph utilities, I/O helpers
```

## License

MIT

## Citation

If you use MolCraftDiffusion in your research, please cite:

[MolecularDiffusion: A Unified Generative-AI Framework for 3D Molecular Design](https://chemrxiv.org/engage/chemrxiv/article-details/6909e50fef936fb4a23df237) (ChemRxiv)

```bibtex
@article{worakul_modular_2026,
	title = {Modular {Framework} for {3D} {Molecular} {Generation} in {Computational} {Chemistry} {Applications}},
	copyright = {https://creativecommons.org/licenses/by/4.0/},
	issn = {0002-7863, 1520-5126},
	url = {https://pubs.acs.org/doi/10.1021/jacs.5c19960},
	doi = {10.1021/jacs.5c19960},
	language = {en},
	urldate = {2026-06-24},
	journal = {Journal of the American Chemical Society},
	author = {Worakul, Thanapat and Azzouzi, Mohammed and Wodrich, Matthew D. and Corminboeuf, Clémence},
	month = jun,
	year = {2026},
	pages = {jacs.5c19960},
}
```
