from MolecularDiffusion.callbacks.train_helper import SP_regularizer
from MolecularDiffusion.modules.tasks import ProperyPrediction, GuidanceModelPrediction, GeomMolecularGenerative
from MolecularDiffusion.modules.models import EGT_dynamics, GraphTransformer, NoiseModel, EnVariationalDiffusion # type: ignore
from MolecularDiffusion.utils import adjust_weights, adjust_bias
import torch
import logging

logger = logging.getLogger(__name__)


class ModelTaskFactory:
    """
    Factory to build models and tasks for diffusion, property prediction, or guidance.

        Parameters:
            task_type (str): "diffusion", "property", or "guidance".
            train_set: Training dataset, used to infer input node feature dimensions.
            atom_vocab (list): List of atom vocabulary used for encoding.
            task_names (list): List of conditional labels (e.g., properties for guidance).
            condition_names (list): List of condition names for conditional generation.
            model_class (str): The model class to use. Defaults to "GraphTransformer".
            num_layers (int): Number of transformer layers.
            hidden_mlp_dims (dict): Dictionary of hidden MLP dimensions for the model.
            hidden_dims (dict): Dictionary of hidden dimensions for the model.
            act_fn_in (torch.nn.Module): Activation function for input layers.
            act_fn_out (torch.nn.Module): Activation function for output layers.
            chkpt_path (str): Optional path to model checkpoint.
            **kwargs: task-specific keyword arguments.

        Diffusion kwargs:
            diffusion_steps (int): Number of timesteps.
            diffusion_noise_schedule (str)
            diffusion_noise_precision
            diffusion_loss_type (str)
            normalize_factors
            extra_norm_values
            augment_noise (bool)
            data_augmentation (bool)
            context_mask_rate (float)
            mask_value (float)
            normalize_condition (str)
            sp_regularizer_regularizer (str)
            sp_regularizer_lambda_ (float)
            sp_regularizer_lambda_2 (float)
            sp_regularizer_lambda_update_value (float)
            sp_regularizer_lambda_update_step (int)
            sp_regularizer_polynomial_p (float)
            sp_regularizer_warm_up_steps (int)

        Property-prediction kwargs:
            task_learn (str)
            criterion (str)
            metric (str)
            num_mlp_layer (int)
            mlp_dropout (float)

        Guidance kwargs:
            diffusion_steps (int)
            diffusion_noise_precision
            nu_arr
            mapping
            task_learn (str)
            metric (str)
            num_mlp_layer (int)
            mlp_dropout (float)
            weight_classes
            norm_values
            t_max
    """
    def __init__(
        self,
        task_type: str,
        train_set,
        atom_vocab,
        task_names,
        condition_names: list = [],
        # Common model arguments
        model_class: str = "GraphTransformer",
        num_layers: int = 6,
        hidden_mlp_dims: dict = {},
        hidden_dims: dict = {},
        act_fn_in: torch.nn.Module = torch.nn.SiLU(),
        act_fn_out: torch.nn.Module = torch.nn.SiLU(),
        chkpt_path: str = None,
        **kwargs
    ):
        self.task_type = task_type
        self.train_set = train_set
        self.atom_vocab = atom_vocab
        self.task_names = task_names
        self.condition_names = condition_names
        self.model_class = model_class
        # Common model hyperparameters
        self.num_layers = num_layers
        self.hidden_mlp_dims = hidden_mlp_dims
        self.hidden_dims = hidden_dims
        self.act_fn_in = act_fn_in
        self.act_fn_out = act_fn_out


        # Compute feature dimensions
        n_dim_extra = len(kwargs.get("extra_norm_values", []))
        self.in_node_nf = len(atom_vocab) + n_dim_extra + 1 # +1 for atomic number
        self.dynamics_in_node_nf = self.in_node_nf + 1 # +1 for time (always include time in dynamics)
        self.context_node_nf = len(self.task_names)
        
        self.chkpt_path = chkpt_path
        self.kwargs = kwargs

        # Some checks
        use_adaptor_module = self.kwargs.get("use_adapter_module", False)
        if use_adaptor_module and self.context_node_nf < 1:
            raise ValueError("Must specify the contexts to use the adapter module.")

    def build(self):
        """
        Build and return (model, task) based on task_type.

        """
        is_main_process = (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        # Construct shared EGT dynamics (skip for diffusion_pyg which uses EGT_dynamics_PyG)
        if self.task_type != "diffusion_pyg":
            dynamics_model = EGT_dynamics(
                in_node_nf=self.dynamics_in_node_nf,
                in_edge_nf=1,
                in_global_nf=1,
                n_layers=self.num_layers,
                hidden_mlp_dims=self.hidden_mlp_dims,
                hidden_dims=self.hidden_dims,
                context_node_nf=self.context_node_nf,
                n_dims=3,
                condition_time=True,
                model=self.model_class,
            )
        

        if self.task_type == "diffusion":
            model = EnVariationalDiffusion(
                dynamics=dynamics_model,
                in_node_nf=self.in_node_nf,
                n_dims=3,
                timesteps=self.kwargs["diffusion_steps"],
                noise_schedule=self.kwargs.get("diffusion_noise_schedule", "polynomial_2"),
                noise_precision=self.kwargs.get("diffusion_noise_precision", 1e-5),
                loss_type=self.kwargs.get("diffusion_loss_type", "l2"),
                norm_values=self.kwargs.get("normalize_factors", [1,4,10]),
                include_charges=True,
                extra_norm_values=self.kwargs.get("extra_norm_values", []),
                context_mask_rate=self.kwargs.get("context_mask_rate", 0.15),
                mask_value=self.kwargs.get("mask_value", None), # CFG
            )
            
            if self.kwargs.get("sp_regularizer_deploy", False):
                if is_main_process:
                    logging.info("SP regularizer is enabled for diffusion task.")
                sp_reg = SP_regularizer(
                    regularizer=self.kwargs.get("sp_regularizer_regularizer", "hard"),
                    lambda_=self.kwargs.get("sp_regularizer_lambda_", 0),
                    lambda_2=self.kwargs.get("sp_regularizer_lambda_2", 1000),
                    lambda_update_value=self.kwargs.get("sp_regularizer_lambda_update_value", 1),
                    lambda_update_step=self.kwargs.get("sp_regularizer_lambda_update_step", 100),
                    polynomial_p=self.kwargs.get("sp_regularizer_polynomial_p", 1.1),
                    warm_up_steps=self.kwargs.get("sp_regularizer_warm_up_steps", 100),
                )
            else:
                if is_main_process:
                    logging.info("SP regularizer is disabled for diffusion task.")
                sp_reg = None
            self.task = GeomMolecularGenerative(
                model,
                augment_noise=self.kwargs.get("augment_noise", False),
                data_augmentation=self.kwargs.get("data_augmentation", False),
                condition=self.task_names,
                sp_regularizer=sp_reg,
                normalize_condition=self.kwargs.get("normalize_condition", None),
                reference_indices=self.kwargs.get("reference_indices", None),
            )

        elif self.task_type == "regression":
            model = GraphTransformer(
                in_node_nf=self.in_node_nf,
                in_edge_nf=1,
                in_global_nf=1,
                n_layers=self.num_layers,
                hidden_mlp_dims=self.hidden_mlp_dims,
                hidden_dims=self.hidden_dims,
                context_node_nf=self.context_node_nf,
                dropout=self.dropout,
                act_fn_in=self.act_fn_in,
                act_fn_out=self.act_fn_out,
            )

            self.task = ProperyPrediction(
                model,
                task=self.kwargs.get("task_learn", ""),
                include_charge=True,
                criterion=self.kwargs.get("criterion", "mse"),
                metric=self.kwargs.get("metric", "mae"),
                num_mlp_layer=self.kwargs.get("num_mlp_layer", 2),
                mlp_batch_norm=self.normalization,
                mlp_dropout=self.kwargs.get("mlp_dropout", 0.0),
                normalization=self.normalization
            )

        elif self.task_type == "guidance":
            model = GraphTransformer(
                in_node_nf=self.in_node_nf,
                in_edge_nf=1,
                in_global_nf=1,
                n_layers=self.num_layers,
                hidden_mlp_dims=self.hidden_mlp_dims,
                hidden_dims=self.hidden_dims,
                context_node_nf=self.context_node_nf,
                dropout=self.dropout,
                act_fn_in=self.act_fn_in,
                act_fn_out=self.act_fn_out,
            )
            noise_model = NoiseModel(
                timestep=self.kwargs.get("diffusion_steps"),
                noise_precision=self.kwargs.get("diffusion_noise_precision"),
                nu_arr=self.kwargs.get("nu_arr"),
                mapping=self.kwargs.get("mapping"),
            )
            self.task = GuidanceModelPrediction(
                model,
                noise_model,
                task=self.kwargs.get("task_learn", ""),
                include_charge=True,
                metric=self.kwargs.get("metric", "mae"),
                num_mlp_layer=self.kwargs.get("num_mlp_layer", 2),
                mlp_batch_norm=True,
                mlp_dropout=self.kwargs.get("mlp_dropout", 0.0),
                normalization=True,
                weight_classes=self.kwargs.get("weight_classes"),
                norm_values=self.kwargs.get("normalize_factors"),
                t_max=self.kwargs.get("t_max"),
            )

        else:
            raise ValueError(f"Unknown task_type '{self.task_type}'. Choose 'diffusion', 'regression', or 'guidance'.")

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # type: ignore
        if is_main_process:
            logger.info(f"Number of parameters: {n_params}")
        
        if self.chkpt_path:    
            try:
                ckpt = torch.load(self.chkpt_path, weights_only=False)
                chk_point = ckpt.get("ema_model") or ckpt.get("model")
                if chk_point is None:
                    raise KeyError("Checkpoint missing both 'ema_model' and 'model'")
                
                if is_main_process:
                    logger.info(f"Loading checkpoint from {self.chkpt_path}")
                
                try:
                    load_result = self.task.load_state_dict(chk_point, strict=False)
                    if is_main_process and (load_result.missing_keys or load_result.unexpected_keys):
                        logger.warning(f"\033[93mCheckpoint loaded with mismatched keys.\033[0m")
                        if load_result.missing_keys:
                            logger.warning(f"\033[93mMissing keys ({len(load_result.missing_keys)}): {load_result.missing_keys}\033[0m")
                        if load_result.unexpected_keys:
                            logger.warning(f"\033[93mUnexpected keys ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys}\033[0m")
                
                except RuntimeError as e:
                    made_adjustment = False
                    
                    # Search for mlp_in_X key (could have prefixes like model.dynamics...)
                    mlp_in_X_keys = [k for k in chk_point.keys() if k.endswith("dynamics.egnn.mlp_in_X.0.weight")]
                    mlp_out_X_w_keys = [k for k in chk_point.keys() if k.endswith("dynamics.egnn.mlp_out_X.2.weight")]
                    mlp_out_X_b_keys = [k for k in chk_point.keys() if k.endswith("dynamics.egnn.mlp_out_X.2.bias")]
                    
                    if mlp_in_X_keys and mlp_out_X_w_keys and mlp_out_X_b_keys:
                        k_in_x = mlp_in_X_keys[0]
                        k_out_w = mlp_out_X_w_keys[0]
                        k_out_b = mlp_out_X_b_keys[0]
                        
                        n_dim_pretrain = chk_point[k_in_x].shape[1]
                        n_extra_dim = self.dynamics_in_node_nf + self.n_concat_context - n_dim_pretrain

                        if n_extra_dim > 0:
                            if is_main_process:
                                logger.info("Adding dimensions to the EGT node embedding...")
                            
                            hidden_mlp_x = chk_point[k_in_x].shape[0]
                            
                            chk_point[k_in_x] = adjust_weights(
                                chk_point[k_in_x], 
                                (hidden_mlp_x, n_dim_pretrain + n_extra_dim)
                            )

                            chk_point[k_out_w] = adjust_weights(
                                chk_point[k_out_w], 
                                (n_dim_pretrain + n_extra_dim, hidden_mlp_x)
                            )
        
                            chk_point[k_out_b] = adjust_bias(
                                chk_point[k_out_b], 
                                (n_dim_pretrain + n_extra_dim,)
                            )      
                            made_adjustment = True

                    mlp_in_y_keys = [k for k in chk_point.keys() if k.endswith("dynamics.egnn.mlp_in_y.0.weight")]
                    if mlp_in_y_keys:
                        k_in_y = mlp_in_y_keys[0]
                        n_global_pretrain = chk_point[k_in_y].shape[1]
                        
                        # In PyG model, global_nf is NOT concatenated with context. 
                        if "diffusion_pyg" in self.task_type:
                            n_global_extra_dim = 1 - n_global_pretrain # Only time is expected in global
                        else:
                            n_global_extra_dim = 1 + self.n_concat_context - n_global_pretrain
                            
                        if n_global_extra_dim > 0:
                            hidden_mlp_y = chk_point[k_in_y].shape[0]
                            chk_point[k_in_y] = adjust_weights(
                                chk_point[k_in_y], 
                                (hidden_mlp_y, n_global_pretrain + n_global_extra_dim)
                            )
                            made_adjustment = True

                    if self.n_adapter_context > 0:
                        emb_c_in_w_keys = [
                            k for k in chk_point.keys()
                            if k.endswith("dynamics.emb_c_in.weight")
                        ]
                        if emb_c_in_w_keys:
                            emb_c_in_w_key = emb_c_in_w_keys[0]
                            n_context_pretrain = chk_point[emb_c_in_w_key].shape[1]
                            n_context_extra = self.n_adapter_context - n_context_pretrain
                            if n_context_extra > 0:
                                if is_main_process:
                                    logger.info("Adding dimensions to the adapter context embedding...")
                                hidden_dx = chk_point[emb_c_in_w_key].shape[0]
                                chk_point[emb_c_in_w_key] = adjust_weights(
                                    chk_point[emb_c_in_w_key],
                                    (hidden_dx, n_context_pretrain + n_context_extra),
                                )
                                made_adjustment = True

                    if made_adjustment:
                        res = self.task.load_state_dict(chk_point, strict=False) 
                        if is_main_process and (res.missing_keys or res.unexpected_keys):
                            logger.warning(f"\033[93mCheckpoint loaded with mismatched keys after adjustment.\033[0m")
                            if res.missing_keys:
                                logger.warning(f"\033[93mMissing keys ({len(res.missing_keys)}): {res.missing_keys}\033[0m")
                            if res.unexpected_keys:
                                logger.warning(f"\033[93mUnexpected keys ({len(res.unexpected_keys)}): {res.unexpected_keys}\033[0m")
                    else:
                        raise RuntimeError(f"The specified model configuration does not match with the checkpoint. Original error: {e}")
                         
                if "mean" in chk_point and "std" in chk_point:
                    self.task.mean = chk_point["mean"]
                    self.task.std = chk_point["std"]                           
            except FileNotFoundError:
                if is_main_process:
                    logger.warning(f"Checkpoint not found at {self.chkpt_path}. Initializing model without loading.")      
                raise FileNotFoundError(f"Checkpoint not found at {self.chkpt_path}.")
        self.task.atom_vocab = self.atom_vocab
            
        return self.task

