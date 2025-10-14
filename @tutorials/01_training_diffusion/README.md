# Tutorial 1: Training a Diffusion Model

This tutorial covers how to train a diffusion model from scratch. We will walk through the configuration process, explaining how to set up your experiment, data, and model parameters.

## The Configuration System (Hydra)

This project uses the [Hydra](https://hydra.cc/) framework for configuration. This allows for a powerful and modular setup. The core idea is that you have one main config file for your experiment, which is "composed" from smaller, default configuration files. You can then easily override any setting in your main file.

---

## Level 1: Your Main Experiment File

For most training runs, you only need to create and modify **one main experiment file**.

A perfect starting point is `configs/example_diffusion_config.yaml`. Let's copy it:

```bash
cp configs/example_diffusion_config.yaml configs/my_first_run.yaml
```

Now, let's look at `my_first_run.yaml`.

### The `defaults` List
The `defaults` section is the most important part. It tells Hydra which smaller files to merge to create the full configuration.

```yaml
defaults:
  - data: mol_dataset
  - tasks: diffusion
  - logger: default
  - trainer: default
  - hydra: default
  - _self_
```
This builds a configuration from `configs/data/mol_dataset.yaml`, `configs/tasks/diffusion.yaml`, and so on.

### Overriding Key Parameters
You can change any parameter from the defaults by simply adding it to your main file. Here are the most common parameters you will want to change.

#### **Key Parameters to Configure in `my_first_run.yaml`**

| Parameter | Description |
| :--- | :--- |
| `trainer.output_path` | **The main output directory.** All logs and checkpoints will be saved here. **This is the most important path to set.** |
| `name` | A unique name for your experiment, used for logging (e.g., in `wandb`). |
| `trainer.num_epochs` | How many epochs to train for. |
| `trainer.lr` | The learning rate for the optimizer. |
| `data.batch_size` | The number of molecules per batch. |
| `seed` | The random seed for reproducibility. |

**Example `my_first_run.yaml`:**
```yaml
# Inherit from the defaults
defaults:
  - data: mol_dataset
  - tasks: diffusion
  - logger: wandb  # Let's use wandb for logging
  - trainer: default
  - hydra: default
  - _self_

# --- Override paths and hyperparameters ---
name: "my_first_diffusion_experiment"
seed: 42

trainer:
  output_path: "outputs/my_first_run" # All results will be here
  num_epochs: 100
  lr: 0.0001

data:
  batch_size: 32
```

---

## Level 2: The Component Files (The "Defaults")

When you need more control, you can modify the default component files or create your own.

### `configs/data/` (Your Dataset)
These files define your dataset and where to find it. To use a new dataset, copy `mol_dataset.yaml` to `my_data.yaml`, edit the paths, and change the `defaults` in your main config to `- data: my_data`.

#### **Key Data Parameters**
| Parameter | Description |
| :--- | :--- |
| **`root`** | **Path** to the root directory of your dataset. |
| **`filename`** | **Path** to the CSV file containing molecule information. |
| **`xyz_dir`** | **Path** to the directory with the `.xyz` coordinate files. |
| `atom_vocab`| List of atom types present in your dataset. |

### `configs/tasks/` (The Model & Task)
This is the heart of your model, defining its architecture and the logic for the diffusion task.

#### **Key Task Parameters**
| Parameter | Description |
| :--- | :--- |
| `hidden_size` | The main dimension of the model's hidden layers. |
| `num_layers` | The number of layers in the model. |
| `diffusion_steps` | How many steps are in the diffusion process. |
| `diffusion_noise_schedule` | The strategy for adding noise (e.g., `polynomial_2`). |
| `condition_names` | A list of property names to use for conditional training. Leave empty for unconditional training. |

### `configs/trainer/` (The Optimizer)
This defines the optimization algorithm, learning rate scheduler, and other training loop details.

#### **Key Trainer Parameters**
| Parameter | Description |
| :--- | :--- |
| `optimizer_choice` | The algorithm to use (e.g., `adamw`, `adam`). |
| `scheduler` | The learning rate scheduler (e.g., `cosineannealing`). |

---

## Level 3: A Practical Walkthrough

Let's put it all together.

**1. Create your main config file:**
   ```bash
   cp configs/example_diffusion_config.yaml configs/my_first_run.yaml
   ```

**2. Edit `configs/my_first_run.yaml`** to set the paths and hyperparameters:
   ```yaml
   defaults:
     - data: mol_dataset # Assuming we use the default dataset for now
     - tasks: diffusion
     - logger: default
     - trainer: default
     - hydra: default
     - _self_

   name: "my_first_run"
   seed: 123

   trainer:
     output_path: "training_outputs/my_first_run"
     num_epochs: 150
     lr: 0.0002

   data:
     batch_size: 64
   ```

**3. Run Training:**
   Launch the training from the project root directory with the following command:
   ```bash
   python scripts/train.py --config-name=my_first_run
   ```
   Hydra will automatically find your file in the `configs` directory, compose the full configuration, and start the training. All results will be saved in `training_outputs/my_first_run`.
