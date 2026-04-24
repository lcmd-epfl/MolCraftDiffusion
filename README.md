MolCraftDiffusion
==================

A unified generative AI framework for 3D molecular generation using diffusion models, designed to streamline the entire workflow from model training to deployment in data-driven computational chemistry pipelines.

MolCraftDiffusion enables researchers to train 3D molecular diffusion models, develop predictive models, and perform guided molecular generation for applications such as catalyst discovery, drug design, and exploration of chemical space.

![workflow](./images/overview.png)

## Key Features

MolCraftDiffusion provides a complete pipeline for training/fine-tuning diffusion models, building predictive property models, and applying them to data-driven molecular generation tasks within a unified framework.

*   **End-to-End 3D Molecular Generation Workflow:** Support training diffusion model, and preditive models, and utilize them for various molecular generation tasks, all within a unified framework.
*   **Curriculum learning:** Efficient way for training and fine-tuning 3D molecular diffusion models
*   **Guidance Tools:** MolCraftDiffusion includes several guidance mechanisms that enable the generation of molecules with desired structural or physicochemical properties.
    *   **Property-Targeted Generation:** Generate molecules with a target physicochemical or electronic properties (e.g., excitation energy, dipole moment)
    *   **Inpainting:** Systematically explore structural variants around reference molecules
    *   **Outpainting:** Extend a molecule by generating new parts.
*   **Command-Line Interface:** A simple and flexible CLI interface enables users to perform training, generation, prediction, and analysis tasks directly from the command line.


[![PyPI](https://img.shields.io/pypi/v/molcraftdiffusion)](https://pypi.org/project/molcraftdiffusion/)
[![arXiv](https://img.shields.io/badge/PDF-arXiv-blue)](https://chemrxiv.org/engage/chemrxiv/article-details/6909e50fef936fb4a23df237)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19511401.svg)](https://doi.org/10.5281/zenodo.19511401)
[![Weights](https://img.shields.io/badge/Weights-HuggingFace-yellow)](https://huggingface.co/pregH/MolecularDiffusion)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/pregH/MolecularDiffusion)
[![Tutorials](https://img.shields.io/badge/Tutorials-Docs-blue)](https://preghosh.github.io/MolCraftDiffusion/)


Try our interactive demo for molecular generation: [MolCraftDiffusion-demo](https://huggingface.co/spaces/pregH/MolCraftDiffusion-demo)

Installation
-----------

    # 1. Create environment
    conda create -n molcraft python=3.11 -y
    conda activate molcraft

    # 2. Install MolCraftDiffusion with a compute backend

    # GPU/CUDA install:
    pip install molcraftdiffusion[gpu] \
        --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html

    # CPU-only install:
    pip install molcraftdiffusion[cpu] \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --find-links https://data.pyg.org/whl/torch-2.6.0+cpu.html

    # 3. Optional feature groups

    # Data preparation/augmentation/featurization utilities:
    pip install 'molcraftdiffusion[data]'

    # Analysis and post-processing utilities:
    pip install 'molcraftdiffusion[analyze]'

    # xTB is used by some analyze commands and is best installed from conda-forge:
    conda install -c conda-forge xtb==6.7.1 -y

The base install keeps data and analysis chemistry packages optional. If you call a `MolCraftDiff data ...` or `MolCraftDiff analyze ...` command without the required optional packages, the command exits with an installation hint instead of crashing.

### Development / editable install

    git clone https://github.com/pregHosh/MolCraftDiffusion
    cd MolCraftDiffusion
    pip install -e .[gpu] \
        --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html

    # Add optional feature groups in editable mode when needed:
    pip install -e '.[data]'
    pip install -e '.[analyze]'

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
*   `data`: Data processing utilities (preparation, augmentation, and dataset operations).

**Command Syntax:**

    MolCraftDiff [COMMAND] [CONFIG_NAME/ARGUMENTS]

*   `[COMMAND]`: One of `train`, `generate`, `predict`, `eval_predict`, `analyze`, or `data`.
*   `[CONFIG_NAME]`: The name of the configuration file from the `configs/` directory (e.g., `train`, `example_diffusion_config`).
*   `[ARGUMENTS]`: Additional command-line arguments to override configuration settings.

**Examples:**

    # Train a model using the 'example_diffusion_config.yaml' configuration
    MolCraftDiff train example_diffusion_config

    # Generate molecules using the 'my_generation_config.yaml' configuration
    MolCraftDiff generate my_generation_config

    # Predict properties using a trained model
    MolCraftDiff predict my_prediction_config

    # Compile molecular data into an ASE database
    MolCraftDiff data prepare compile -s data_dir/ -d dataset.db


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

Tutorials are now hosted in the docs site: https://preghosh.github.io/MolCraftDiffusion/

The local `tutorials/` directory is deprecated and will be removed in a future release.



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
