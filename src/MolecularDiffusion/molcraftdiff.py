import fire
import subprocess
import os
import logging
import platform
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_system_info():
    """Logs detailed system information."""
    logger.info("-------------------- System Information --------------------")
    # OS
    logger.info(f"OS: {platform.system()} {platform.release()}")

    # CPU
    logger.info(f"CPU: {platform.processor()}, Cores: {os.cpu_count()}")

    # RAM
    ram = psutil.virtual_memory()
    logger.info(f"RAM: Total {ram.total / (1024**3):.2f} GB, Available {ram.available / (1024**3):.2f} GB")

    # Python
    logger.info(f"Python Version: {platform.python_version()}")

    # PyTorch and Torch CUDA
    try:
        import torch
        logger.info(f"PyTorch Version: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"PyTorch CUDA Version: {torch.version.cuda}")
        else:
            logger.info("PyTorch: CUDA not available.")
    except ImportError:
        logger.warning("PyTorch not found. Cannot log PyTorch and CUDA versions.")

    # GPU details from nvidia-smi
    try:
        # Check for GPU count first
        count_result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader'], capture_output=True, text=True, check=True)
        num_gpus = int(count_result.stdout.strip())

        if num_gpus == 0:
            logger.info("No NVIDIA GPUs detected by nvidia-smi.")
        else:
            # Query driver version
            try:
                driver_result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits', '-i', '0'],
                    capture_output=True, text=True, check=True
                )
                logger.info(f"NVIDIA Driver Version: {driver_result.stdout.strip()}")
            except (subprocess.CalledProcessError, ValueError) as e:
                logger.warning(f"Could not query NVIDIA Driver Version: {e}")

            # Per-GPU info
            gpus_result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            gpus = gpus_result.stdout.strip().splitlines()
            logger.info("GPU Information:")
            for i, gpu_line in enumerate(gpus):
                name, mem_total_mb, mem_free_mb = [x.strip() for x in gpu_line.split(',')]
                mem_total_gb = float(mem_total_mb) / 1024
                mem_free_gb = float(mem_free_mb) / 1024
                logger.info(f"  - GPU {i}: {name}, VRAM: Total {mem_total_gb:.2f} GB, Available {mem_free_gb:.2f} GB")

    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"Could not query NVIDIA GPU information via nvidia-smi: {e}")
    logger.info("----------------------------------------------------------")

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
        log_system_info()
        logger.info(f"Attempting to run script: {script_name} with config: {config_file}")
        try:
            project_root = find_project_root()
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            return

        script_path = os.path.join(project_root, "scripts", script_name)
        if not os.path.exists(script_path):
            logger.error(f"Error: Script '{script_name}' not found at '{script_path}'")
            return

        # Hydra expects the config file name without the .yaml extension or parent dirs
        if "/" in config_file:
            config_file = os.path.basename(config_file)
        if config_file.endswith(".yaml"):
            config_file = config_file[:-5]

        # Check for SLURM environment and multiple GPUs
        is_slurm = "SLURM_JOB_ID" in os.environ
        num_gpus = 0
        
        if is_slurm:
            logger.info("SLURM environment detected.")
            try:
                num_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", 0))
            except (ValueError, TypeError):
                num_gpus = 0
        else:
            # By default, we will just run with python, and let torch handle device placement
            pass

        if is_slurm and num_gpus > 1:
            command = [
                "srun",
                "python",
                script_path,
                f"--config-name={config_file}",
            ]
        else:
            command = [
                "python",
                script_path,
                f"--config-name={config_file}",
            ]

        logger.info(f"Running command: {' '.join(command)}")
        # We run the command from the project root for hydra to work correctly
        subprocess.run(command, cwd=project_root)

def main():
    fire.Fire(MolCraftDiff)

if __name__ == "__main__":
    main()