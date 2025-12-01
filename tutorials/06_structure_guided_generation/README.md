# Tutorial 6: Structure-Guided Generation

This tutorial explains how to guide molecule generation using structural constraints, such as filling in a missing piece (inpainting) or growing a molecule from a fragment (outpainting).

## Contents

1.  **Introduction**: The concept of guiding generation with a structural template.
2.  **Inpainting**: How to configure and run generation to fill in a missing portion of a molecule.
3.  **Outpainting**: How to grow a molecule from a given substructure.
4.  **3D Geometric Constraints**: How to tune the geometric constraints.

## 1. Introduction

**Important Note:** The configuration files for this tutorial must be placed in the `configs/` directory at the root of the project for the scripts to read the settings.

Structure-guided generation allows you to influence the output of the diffusion model by providing a starting molecular structure. This is useful for tasks like:

*   **Inpainting**: Completing a molecule where a part is missing.
*   **Outpainting**: Extending a molecule from a given fragment.

The process involves providing a reference structure in an XYZ file and specifying which parts of the structure to modify or keep fixed. **Note that all atom indices are 0-indexed.**

## 2. Inpainting

Inpainting is the process of filling in a missing part of a molecule. You provide a template molecule and specify which atoms to "mask". The diffusion model will then generate the missing atoms and connect them to the rest of the molecule.

### Key Inpainting Parameters

The `condition_configs` section for inpainting now uses a sub-dictionary called `inpaint_cfgs` to group all specific inpainting settings.

| Parameter | Location | Description |
| :--- | :--- | :--- |
| `mol_size` | `interference` (top-level) | The expected size of the final molecule. **This should be larger than or equal to the number of atoms in the reference structure.** |
| `reference_structure_path` | `condition_configs` | **CRITICAL:** Path to your own XYZ file containing the molecule you want to inpaint. |
| `condition_component` | `condition_configs` | Component to inpaint (available choice: x, h, xh). |
| `n_retrys` | `condition_configs` | Number of retry attempts in case of bad molecules. |
| `t_retry` | `condition_configs` | Timestep to start retrying. |
| `n_frames` | `condition_configs` | Number of frames to keep for trajectory visualization. |
| `inpaint_cfgs` | `condition_configs` | **CRITICAL:** A sub-dictionary containing all settings specific to the inpainting algorithm, including `mask_node_index`, `denoising_strength`, and geometric control parameters. |
| `mask_node_index` | `inpaint_cfgs` | **CRITICAL:** A list of **0-indexed** atom indices from your XYZ file that you want to remove and have the model regenerate. |
| `denoising_strength` | `inpaint_cfgs` | Controls how much noise is added to the masked region before generation. Higher values give the model more creative freedom. |


### Configuration

Here is an example of a complete configuration file for inpainting, which you can name `my_inpaint.yaml`:

```yaml
# This file represents the combined configuration for inpainting generation.
# In the actual project, this is composed from `configs/generate.yaml` and `configs/interference/my_inpaint.yaml`.

defaults:
  - tasks: diffusion
  - interference: my_inpaint
  - _self_

name: "akatsuki"
chkpt_directory: "models/edm_pretrained/"
atom_vocab: [H,B,C,N,O,F,Al,Si,P,S,Cl,As,Se,Br,I,Hg,Bi]
diffusion_steps: 600
seed: 9

interference:
  _target_: MolecularDiffusion.runmodes.generate.GenerativeFactory
  task_type: inpaint
  sampling_mode: "ddpm"
  num_generate: 50
  mol_size: [50, 60] # Target size of the generated molecule (top-level interference param)
  output_path: "results/my_inpainting_run"
  condition_configs:
    reference_structure_path: "assets/BINOLCpHHH.xyz"
    condition_component: xh
    n_frames: 0
    n_retrys: 0
    t_retry: 180
    inpaint_cfgs:
      # To vary the BINOL part of the molecule, we mask the following 0-indexed atoms:
      mask_node_index: [5, 30, 31, 6, 7, 45, 8, 32, 9, 10, 33, 11, 34, 12, 35, 13, 36, 14, 15, 16, 17, 18, 37, 19, 38, 20, 39, 21, 40, 22, 23, 41, 24, 44, 25, 26, 43, 42]
      denoising_strength: 0.8
```

### Running Inpainting

Use the `MolCraftDiff generate` command with your configuration file:

```bash
MolCraftDiff generate my_inpaint.yaml
```

## 3. Outpainting

Outpainting is the process of growing a molecule from a given fragment. You provide a starting fragment, and the model will add new atoms to it.

### Key Outpainting Parameters

The `condition_configs` section for outpainting now uses a sub-dictionary called `outpaint_cfgs` to group all specific outpainting settings.

| Parameter | Location | Description |
| :--- | :--- | :--- |
| `mol_size` | `interference` (top-level) | The expected size of the final molecule (fragment + generated part). |
| `reference_structure_path` | `condition_configs` | **CRITICAL:** Path to your own XYZ file containing the fragment you want to grow from. |
| `condition_component` | `condition_configs` | Component to outpaint (available choice: x, h, xh). |
| `n_retrys` | `condition_configs` | Number of retry attempts in case of bad molecules. |
| `t_retry` | `condition_configs` | Timestep to start retrying. |
| `n_frames` | `condition_configs` | Number of frames to keep for trajectory visualization. |
| `outpaint_cfgs` | `condition_configs` | **CRITICAL:** A sub-dictionary containing all settings specific to the outpainting algorithm, including `connector_dicts`, `t_start`, and geometric control parameters. |
| `connector_dicts` | `outpaint_cfgs` | **CRITICAL:** A dictionary where keys are the **0-indexed** indices of atoms in your fragment, and values are the number of new connections to grow from that atom. |
| `t_start` | `outpaint_cfgs` | Timestep to start the generation. |


### Configuration

Here is an example of a complete configuration file for outpainting, which you can name `my_outpaint.yaml`:

### Example `my_outpaint.yaml`

```yaml
# This file represents the combined configuration for outpainting generation.
# In the actual project, this is composed from `configs/generate.yaml` and `configs/interference/my_outpaint.yaml`.

defaults:
  - tasks: diffusion
  - interference: my_outpaint
  - _self_

name: "akatsuki"
chkpt_directory: "models/edm_pretrained/"
atom_vocab: [H,B,C,N,O,F,Al,Si,P,S,Cl,As,Se,Br,I,Hg,Bi]
diffusion_steps: 600
seed: 9

interference:
  _target_: MolecularDiffusion.runmodes.generate.GenerativeFactory
  task_type: outpaint
  sampling_mode: "ddpm"
  num_generate: 50
  mol_size: [30, 40] # Target size of the generated molecule (top-level interference param)
  output_path: "results/my_outpainting_run"
  condition_configs:
    reference_structure_path: "assets/BINOLCp.xyz"
    condition_component: xh
    n_frames: 0
    n_retrys: 3
    t_retry: 180
    outpaint_cfgs:
      # To decorate BINOL-Cp with substituents at 0-indexed atoms 1, 2, and 3, each with 3 bonds:
      connector_dicts:
        1: [3]
        2: [3]
        3: [3]
      t_start: 0.8
```

### Running Outpainting

Use the `MolCraftDiff generate` command with your configuration file:

```bash
MolCraftDiff generate my_outpaint.yaml
```

## 4. Geometric Control Settings for Inpainting and Outpainting

The generation process for both inpainting and outpainting is guided by a set of geometric control settings that can be tuned within the `inpaint_cfgs` and `outpaint_cfgs` sub-dictionaries of your `condition_configs`. These parameters influence how the model handles collisions, connectivity, and the overall shape of the generated molecule.

### Common Geometric Control Parameters

These parameters are applicable to both inpainting and outpainting, nested within their respective `*_cfgs` dictionaries.

| Parameter | Location | Description |
| :--- | :--- | :--- |
| `d_threshold_f` | `inpaint_cfgs` or `outpaint_cfgs` | The minimum allowed distance (in Angstroms) between a generated atom and the fixed (frozen) atoms (in inpainting) or initial fragment (in outpainting). If a generated atom is closer than this threshold, it will be pushed away. |
| `w_b` | `inpaint_cfgs` or `outpaint_cfgs` | A weight that controls a push/pull force on the generated atoms. A higher value results in a stronger push away from fixed atoms or the fragment. |
| `all_frozen` | `inpaint_cfgs` or `outpaint_cfgs` | If `True` (for inpainting), all atoms in the reference structure are considered fixed. If `False` (for outpainting), atoms in the initial fragment can be slightly adjusted. |
| `use_covalent_radii`| `inpaint_cfgs` or `outpaint_cfgs` | If `True`, the collision avoidance logic will use the covalent radii of the atoms to determine the minimum allowed distance, instead of the fixed `d_threshold_f`. |
| `scale_factor` | `inpaint_cfgs` or `outpaint_cfgs` | A scaling factor for the covalent radii when `use_covalent_radii` is `True`. A value greater than 1.0 increases the effective size of the atoms, creating more space between them. |
| `t_critical_1`, `t_critical_2` | `inpaint_cfgs` or `outpaint_cfgs` | These parameters control the timesteps during the diffusion process at which the geometric constraints are most strongly applied. |
| `noise_initial_mask` | `inpaint_cfgs` or `outpaint_cfgs` | If `True`, the initial masked region (in inpainting) or the region to be generated (in outpainting) is initialized with noise. |

### Additional Outpainting-Specific Geometric Control Parameters

| Parameter | Location | Description |
| :--- | :--- | :--- |
| `d_threshold_c` | `outpaint_cfgs` | The minimum allowed distance between a generated atom and the connector atom it is attached to. |

By tuning these parameters, you can gain fine-grained control over the geometry of the generated molecules.