import fire
import subprocess
import os

def find_project_root(marker=".project-root"):
    """Find the project root directory by checking the current working directory."""
    path = os.getcwd()
    if not os.path.exists(os.path.join(path, marker)):
        raise FileNotFoundError(
            f"Could not find project root marker: {marker}. "
            f"Please run this command from the root directory of the project (your current directory is {path})."
        )
    return path

class MolCraftDiff:
    """MolCraftDiff: A command-line interface for the Molecular Diffusion project.

    This tool provides access to the main workflows of the project, such as training
    models, generating molecules, and running predictions.

    Available commands are:
      - train
      - generate
      - predict
      - eval_predict

    To get help for a specific command, run:
      MolCraftDiff <command> --help

    Example:
      MolCraftDiff train --help
    """

    def train(self, config_file: str):
        """Run the training script.

        This command launches a training run using the specified configuration file.

        Args:
            config_file: The name of the configuration file (e.g., 'train' or 'train.yaml')
                         located in the 'configs/' directory.
        
        Example:
            MolCraftDiff train example_diffusion_config
        """
        self._run_script("train.py", config_file)

    def generate(self, config_file: str):
        """Run the generation script.

        This command launches a generation run using the specified configuration file.

        Args:
            config_file: The name of the configuration file (e.g., 'generate' or 'generate.yaml')
                         located in the 'configs/' directory.
        
        Example:
            MolCraftDiff generate my_generation_config
        """
        self._run_script("generate.py", config_file)

    def predict(self, config_file: str):
        """Run the prediction script.

        This command launches a prediction run using the specified configuration file.

        Args:
            config_file: The name of the configuration file (e.g., 'predict' or 'predict.yaml')
                         located in the 'configs/' directory.
        
        Example:
            MolCraftDiff predict my_prediction_config
        """
        self._run_script("predict.py", config_file)

    def eval_predict(self, config_file: str):
        """Run the prediction evaluation script.

        This command launches a prediction evaluation run using the specified configuration file.

        Args:
            config_file: The name of the configuration file (e.g., 'eval_predict' or 'eval_predict.yaml')
                         located in the 'configs/' directory.
        
        Example:
            MolCraftDiff eval_predict my_eval_config
        """
        self._run_script("eval_predict.py", config_file)

    def _run_script(self, script_name: str, config_file: str):
        """Helper method to run a script with a given config file."""
        try:
            project_root = find_project_root()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

        script_path = os.path.join(project_root, "scripts", script_name)
        if not os.path.exists(script_path):
            print(f"Error: Script '{script_name}' not found at '{script_path}'")
            return

        # Hydra expects the config file name without the .yaml extension or parent dirs
        if "/" in config_file:
            config_file = os.path.basename(config_file)
        if config_file.endswith(".yaml"):
            config_file = config_file[:-5]

        command = [
            "python",
            script_path,
            f"--config-name={config_file}",
        ]

        print(f"Running command: {' '.join(command)}")
        # We run the command from the project root for hydra to work correctly
        subprocess.run(command, cwd=project_root)

def main():
    fire.Fire(MolCraftDiff)

if __name__ == "__main__":
    main()