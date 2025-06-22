import torch
from typing import Tuple




def translate_to_origine(coords, node_mask):
    centroid = coords.mean(dim=1, keepdim=True)  
    translation_vector = -centroid
    translated_coords = coords + translation_vector * node_mask
    return translated_coords

    
def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask, std=1.0):
    assert len(size) == 3
    x = torch.randn(size, device=device) * std

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def sample_gaussian_with_mask(size, device, node_mask, std=1.0):
    x = torch.randn(size, device=device) * std
    x_masked = x * node_mask
    return x_masked





def coord2cosine(x, edge_index, epsilon=1e-8):
    row, col = edge_index
    tensor1, tensor2 = x[row], x[col]
    dot_product = torch.sum(tensor1 * tensor2, dim=-1)
    magnitude1 = torch.sqrt(torch.sum(tensor1**2, dim=-1)) + epsilon
    magnitude2 = torch.sqrt(torch.sum(tensor2**2, dim=-1)) + epsilon

    cosine_sim = dot_product / (magnitude1 * magnitude2)

    return cosine_sim



def coord2diff(x: torch.Tensor, edge_index: torch.Tensor, norm_constant: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the radial distance and normalized coordinate difference between nodes connected by edges.

    Args:
        x (torch.Tensor): Node coordinates of shape (num_nodes, 3).
        edge_index (torch.Tensor): Edge indices of shape (2, num_edges).
        norm_constant (float, optional): Constant added to the normalization term for numerical stability. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Radial distances of shape (num_edges, 1) and normalized coordinate differences of shape (num_edges, 3).
    """
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, dim=1, keepdim=True)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def remove_mean(x: torch.Tensor) -> torch.Tensor:
    """
    Removes the mean from a tensor along dimension 1.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Mean-centered tensor.
    """
    mean = torch.mean(x, dim=1, keepdim=True)
    return x - mean


def remove_mean_with_mask(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    Removes the mean from a tensor along dimension 1, considering a node mask.

    Args:
        x (torch.Tensor): Input tensor.
        node_mask (torch.Tensor): Boolean mask indicating valid nodes.

    Returns:
        torch.Tensor: Mean-centered tensor.
    """
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f"Error {masked_max_abs_value} too high"
    N = node_mask.sum(1, keepdims=True)
    mean = torch.sum(x, dim=1, keepdim=True) / N
    return x - mean * node_mask


def remove_mean_with_mask_v2(pos: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    Removes the mean from a tensor along dimension 1, considering a node mask.

    Args:
        pos (torch.Tensor): Input tensor of shape (bs, n, 3).
        node_mask (torch.Tensor): Boolean mask of shape (bs, n) indicating valid nodes.

    Returns:
        torch.Tensor: Mean-centered tensor.
    """
    # assert node_mask.dtype == torch.bool, f"Wrong dtype for the mask: {node_mask.dtype}"
    N = node_mask.sum(1, keepdims=True)
    mean = torch.sum(pos, dim=1, keepdim=True) / N
    return pos - mean * node_mask



def assert_mean_zero(x: torch.Tensor) -> None:
    """
    Asserts that the mean of a tensor along dimension 1 is close to zero.

    Args:
        x (torch.Tensor): Input tensor.
    """
    mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f"Mean is not zero, relative_error {rel_error}"


def assert_correctly_masked(variable: torch.Tensor, node_mask: torch.Tensor) -> None:
    """
    Asserts that the masked values in the variable are close to zero.

    Args:
        variable (torch.Tensor): Input tensor.
        node_mask (torch.Tensor): Boolean mask indicating valid nodes.
    """
    assert (
        variable * (1 - node_mask)
    ).abs().max().item() < 1e-4, "Variables not masked properly."
    
def check_mask_correct(variables: list, node_mask: torch.Tensor) -> None:
    """
    Checks if variables are correctly masked using assert_correctly_masked.

    Args:
        variables (list): List of variables to check.
        node_mask (torch.Tensor): Node mask to apply.
    """
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


