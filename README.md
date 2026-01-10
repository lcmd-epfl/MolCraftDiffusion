MolecularDiffusion
==================

The unified generative‑AI framework that streamline training the 3D molecular diffusion models to their deployment in data-driven computational chemistry pipelines

![workflow](./images/overview.png)

## Key Features

*   **End-to-End 3D Molecular Generation Workflow:** Support training diffusion model, and preditive models, and utilize them for various molecular generation tasks, all within a unified framework.
*   **Curriculum learning:** Efficient way for training and fine-tuning 3D molecular diffusion models
*   **Guidance Tools:** Generate molecules with specific characteristics:
    *   **Property-Targeted Generation:** Generate molecules with a target physicochemical or electronic properties (e.g., excitation energy, dipole moment)
    *   **Inpainting:** Systematically explore structural variants around reference molecules
    *   **Outpainting:** Extend a molecule by generating new parts.
*   **Command-Line Interface:** A user-friendly CLI for training, generation, and prediction.


[![arXiv](https://img.shields.io/badge/PDF-arXiv-blue)](https://chemrxiv.org/engage/chemrxiv/article-details/6909e50fef936fb4a23df237)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18121166.svg)](https://zenodo.org/records/18121166)
[![Weights](https://img.shields.io/badge/Weights-HuggingFace-yellow)](https://huggingface.co/pregH/MolecularDiffusion)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/pregH/MolecularDiffusion)



Installation
-----------

### Detailed Installation Guide

For a more detailed installation, including setting up a conda environment and installing necessary packages, follow these steps:

    # create new python environment
    conda create -n moleculardiffusion python=3.11 -c defaults
    conda activate moleculardiffusion

    # install pytorch according to instructions (use CUDA version for your system)
    # https://pytorch.org/get-started/
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
    
    # install pytorch geometric (use CUDA version for your system)
    # https://pytorch-geometric.readthedocs.io/
    pip install torch_geometric

    # Optional dependencies:
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
    conda install conda-forge::openbabel
    conda install xtb==6.7.1
    # install other libraries
    pip install fire seaborn decorator numpy scipy rdkit-pypi posebusters==0.5.1 networkx matplotlib pandas scikit-learn tqdm pyyaml omegaconf ase morfeus-ml morfeus-ml wandb rmsd

    pip install hydra-core==1.* hydra-colorlog rootutils

    # install cell2mol
    git clone https://github.com/lcmd-epfl/cell2mol
    cd cell2mol
    python setup.py install
    cd ..
    rm -rf cell2mol

    # Install the package. Use editable mode (-e) to make the MolCraftDiff CLI tool available.
    pip install -e .

    # optional for some featurizer/metrics
    # this require numpy==1.24.*
    pip install cosymlib

Usage
-----

### Pre-trained Models

Pre-trained diffusion models are available at [Hugging Face](https://huggingface.co/pregHosh/MolecularDiffusion) or in the `models/edm_pretrained/` directory. We suggest to start from this model for downstream application.

There are two ways to run experiments: using the `MolCraftDiff` command-line tool (recommended) or by executing the Python scripts directly.

### 1. `MolCraftDiff` CLI (Recommended)

Make sure you have installed the package in editable mode as described above, and that you run the commands from the root of the project directory.

**Commands:**
*   `train`: Run a training job.
*   `generate`: Run a molecule generation job.
*   `predict`: Run prediction with a trained model.
*   `eval_predict`: Evaluate predictions.
*   `analyze`: Perform analysis and post-processing on generated molecules.

**Command Syntax:**

    MolCraftDiff [COMMAND] [CONFIG_NAME/ARGUMENTS]

*   `[COMMAND]`: One of `train`, `generate`, `predict`, `eval_predict`, or `analyze`.
*   `[CONFIG_NAME]`: The name of the configuration file from the `configs/` directory (e.g., `train`, `example_diffusion_config`).
*   `[ARGUMENTS]`: Additional command-line arguments to override configuration settings.

**Examples:**

    # Train a model using the 'example_diffusion_config.yaml' configuration
    MolCraftDiff train example_diffusion_config

    # Generate molecules using the 'my_generation_config.yaml' configuration
    MolCraftDiff generate my_generation_config

    # Predict properties using a trained model
    MolCraftDiff predict my_prediction_config


**Getting Help:**

To see the main help message and a list of all commands:

    MolCraftDiff --help

To get help for a specific command:

    MolCraftDiff train --help

### 2. Direct Script Execution

You can also execute the scripts in the `scripts/` directory directly.

**Training:**

    python scripts/train.py tasks=[TASK]

where TASK is one of the following: `diffusion`, `guidance`, `regression`.

**Generation:**

    python scripts/generate.py interference=[INTERFERENCE]

where INTERFERENCE is one of the following: `gen_cfg`, `gen_cfggg`, `gen_conditional`, `gen`.

**Prediction:**

    python scripts/predict.py


### 3. Analysis & Post-processing

The `analyze` command provides a suite of tools for processing and evaluating generated molecules.

**Subcommands:**
*   `optimize`: Optimize molecular geometries using GFN-xTB.
*   `metrics`: Compute validity and connectivity metrics.
*   `compare`: Calculate RMSD, energy differences, and geometric properties (bonds/angles) between generated and reference structures.
*   `xyz2mol`: Convert XYZ files to SMILES and extract fingerprints/scaffolds.

**Examples:**

    # Optimize geometries in a directory
    MolCraftDiff analyze optimize -i generated_molecules/

    # Compute validity metrics
    MolCraftDiff analyze metrics -i generated_molecules/

    # Compare generated structures with ground truth (requires optimized counterparts)
    MolCraftDiff analyze compare generated_molecules/ --bonds

    # Convert XYZ to SMILES
    MolCraftDiff analyze xyz2mol -x generated_molecules/


Visualization
-------------

Generated 3D molecules and their properties can be visualized using the [3DMolViewer](https://github.com/pregHosh/3DMolViewer) package.

We also recommend our in-house and lightweight X11 molecular viewer [V](https://github.com/briling/v) package.


Tutorials
---------

A comprehensive set of tutorials is available in the [`tutorials/`](./tutorials/) directory, covering topics from basic model training to advanced generation techniques.



Project Structure
-----------------

```
├── .project-root
├── justfile
├── pyproject.toml
├── README.md
├── setup.py
└── src
    └── MolecularDiffusion
       ├── __init__.py
       ├── _version.py
       ├── molcraftdiff.py
       ├── callbacks
       │   ├── __init__.py
       │   └── train_helper.py
       ├── cli
       │   ├── __init__.py
       │   ├── analyze.py
       │   ├── eval_predict.py
       │   ├── generate.py
       │   ├── main.py
       │   ├── predict.py
       │   └── train.py
       ├── configs
       │   ├── data
       │   ├── hydra
       │   ├── interference
       │   ├── logger
       │   ├── tasks
       │   └── trainer
       ├── core
       │   ├── __init__.py
       │   ├── core.py
       │   ├── engine.py
       │   ├── logger.py
       │   └── meter.py
       ├── data
       │   ├── __init__.py
       │   ├── dataloader.py
       │   ├── dataset.py
       │   └── component
       ├── modules
       │   ├── __init__.py
       │   ├── layers
       │   ├── models
       │   └── tasks
       ├── runmodes
       │   ├── __init__.py
       │   ├── analyze
       │   │   ├── __init__.py
       │   │   ├── compute_energy_rmsd.py
       │   │   ├── compute_metrics.py
       │   │   ├── compute_pair_geometry.py
       │   │   ├── xtb_optimization.py
       │   │   └── xyz2mol.py
       │   ├── generate
       │   └── train
       └── utils
           ├── __init__.py
           ├── comm.py
           ├── diffusion_utils.py
           ├── file.py
           ├── geom_analyzer.py
           ├── geom_constant.py
           ├── geom_constraint.py
           ├── geom_metrics.py
           ├── geom_utils.py
           ├── io.py
           ├── molgraph_utils.py
           ├── plot_function.py
           ├── pretty.py
           ├── sascore.py
           ├── smilify.py
           └── torch.py
```


License
-------

This project is licensed under the MIT License.


Citation
--------

If you use MolecularDiffusion in your research, please cite the following:

[ChemRxiv: MolecularDiffusion: A Unified Generative-AI Framework for 3D Molecular Design](https://chemrxiv.org/engage/chemrxiv/article-details/6909e50fef936fb4a23df237)

<!-- ```bibtex
@article{hosh2025moleculardiffusion,
to be filled
}
``` -->