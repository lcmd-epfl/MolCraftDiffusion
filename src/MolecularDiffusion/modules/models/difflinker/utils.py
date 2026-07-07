"""Small numerical helpers ported near-verbatim from DiffLinker's
``src/utils.py`` (only the subset actually used by ``edm.py``/``egnn.py`` in
this integration; visualization/logging/EMA helpers were dropped as unused).
"""

import torch


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.size(0), -1).sum(dim=-1)


def remove_mean_with_mask(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f"Error {masked_max_abs_value} too high"
    n = node_mask.sum(1, keepdims=True)
    mean = torch.sum(x, dim=1, keepdim=True) / n
    return x - mean * node_mask


def remove_partial_mean_with_mask(
    x: torch.Tensor, node_mask: torch.Tensor, center_of_mass_mask: torch.Tensor
) -> torch.Tensor:
    """Subtract center of mass of fragments from coordinates of all atoms."""
    x_masked = x * center_of_mass_mask
    n = center_of_mass_mask.sum(1, keepdims=True)
    mean = torch.sum(x_masked, dim=1, keepdim=True) / n
    return x - mean * node_mask


def assert_correctly_masked(variable: torch.Tensor, node_mask: torch.Tensor) -> None:
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, (
        "Variables not masked properly."
    )


def assert_mean_zero_with_mask(
    x: torch.Tensor, node_mask: torch.Tensor, eps: float = 1e-10
) -> None:
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f"Mean is not zero, relative_error {rel_error}"


def assert_partial_mean_zero_with_mask(
    x: torch.Tensor,
    node_mask: torch.Tensor,
    center_of_mass_mask: torch.Tensor,
    eps: float = 1e-10,
) -> None:
    assert_correctly_masked(x, node_mask)
    x_masked = x * center_of_mass_mask
    largest_value = x_masked.abs().max().item()
    error = torch.sum(x_masked, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f"Partial mean is not zero, relative_error {rel_error}"


def sample_gaussian_with_mask(size, device, node_mask) -> torch.Tensor:
    x = torch.randn(size, device=device)
    return x * node_mask


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask) -> torch.Tensor:
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return remove_mean_with_mask(x_masked, node_mask)


def split_features(z, n_dims, num_classes, include_charges):
    assert z.size(2) == n_dims + num_classes + include_charges
    x = z[:, :, 0:n_dims]
    h = {"categorical": z[:, :, n_dims : n_dims + num_classes]}
    if include_charges:
        h["integer"] = z[:, :, n_dims + num_classes : n_dims + num_classes + 1]
    return x, h


class FoundNaNException(Exception):
    def __init__(self, x, h):
        x_nan_idx = self.find_nan_idx(x)
        h_nan_idx = self.find_nan_idx(h)
        self.x_h_nan_idx = x_nan_idx & h_nan_idx
        self.only_x_nan_idx = x_nan_idx.difference(h_nan_idx)
        self.only_h_nan_idx = h_nan_idx.difference(x_nan_idx)

    @staticmethod
    def find_nan_idx(z):
        idx = set()
        for i in range(z.shape[0]):
            if torch.any(torch.isnan(z[i])):
                idx.add(i)
        return idx
