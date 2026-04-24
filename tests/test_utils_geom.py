"""Unit tests for MolecularDiffusion.utils.geom_utils."""

import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mask(batch_size, n_nodes, valid_nodes, device="cpu"):
    """Return a (B, N, 1) node mask with the first *valid_nodes* entries set."""
    mask = torch.zeros(batch_size, n_nodes, 1, device=device)
    mask[:, :valid_nodes, :] = 1.0
    return mask


# ---------------------------------------------------------------------------
# translate_to_origine
# ---------------------------------------------------------------------------

class TestTranslateToOrigine:
    def test_centroid_is_zero_after_translation(self):
        from MolecularDiffusion.utils.geom_utils import translate_to_origine

        coords = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]])  # (1, 2, 3)
        mask = torch.ones(1, 2, 1)
        out = translate_to_origine(coords, mask)
        assert out.mean(dim=1).abs().max().item() < 1e-5

    def test_masked_nodes_remain_zero(self):
        from MolecularDiffusion.utils.geom_utils import translate_to_origine

        coords = torch.zeros(2, 4, 3)
        coords[:, :2, :] = torch.randn(2, 2, 3)
        mask = _make_mask(2, 4, 2)
        out = translate_to_origine(coords, mask)
        # padded positions should still be zero (mask = 0 → no translation applied)
        assert out[:, 2:, :].abs().max().item() < 1e-6

    def test_single_node_translates_to_origin(self):
        from MolecularDiffusion.utils.geom_utils import translate_to_origine

        coords = torch.tensor([[[5.0, -3.0, 2.0]]])
        mask = torch.ones(1, 1, 1)
        out = translate_to_origine(coords, mask)
        assert out.abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# remove_mean / remove_mean_with_mask
# ---------------------------------------------------------------------------

class TestRemoveMean:
    def test_remove_mean_zero_mean(self):
        from MolecularDiffusion.utils.geom_utils import remove_mean

        x = torch.randn(3, 5, 3)
        out = remove_mean(x)
        assert out.mean(dim=1).abs().max().item() < 1e-5

    def test_remove_mean_preserves_shape(self):
        from MolecularDiffusion.utils.geom_utils import remove_mean

        x = torch.randn(4, 6, 3)
        assert remove_mean(x).shape == x.shape

    def test_remove_mean_with_mask_zero_mean_valid_nodes(self):
        from MolecularDiffusion.utils.geom_utils import remove_mean_with_mask

        B, N, D = 2, 6, 3
        mask = _make_mask(B, N, 4)  # first 4 valid
        x = torch.randn(B, N, D) * mask
        out = remove_mean_with_mask(x, mask)
        # mean over valid nodes should be ~0
        mean_valid = (out * mask).sum(dim=1) / mask.sum(dim=1)
        assert mean_valid.abs().max().item() < 1e-4

    def test_remove_mean_with_mask_masked_nodes_stay_zero(self):
        from MolecularDiffusion.utils.geom_utils import remove_mean_with_mask

        B, N, D = 2, 6, 3
        mask = _make_mask(B, N, 4)
        x = torch.randn(B, N, D) * mask
        out = remove_mean_with_mask(x, mask)
        assert out[:, 4:, :].abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# sample_gaussian_with_mask
# ---------------------------------------------------------------------------

class TestSampleGaussianWithMask:
    def test_output_shape(self):
        from MolecularDiffusion.utils.geom_utils import sample_gaussian_with_mask

        size = (2, 5, 3)
        mask = _make_mask(2, 5, 3)
        out = sample_gaussian_with_mask(size, "cpu", mask)
        assert out.shape == torch.Size(size)

    def test_masked_positions_are_zero(self):
        from MolecularDiffusion.utils.geom_utils import sample_gaussian_with_mask

        size = (3, 8, 3)
        mask = _make_mask(3, 8, 5)
        out = sample_gaussian_with_mask(size, "cpu", mask)
        assert out[:, 5:, :].abs().max().item() == 0.0

    def test_valid_positions_are_nonzero_in_expectation(self):
        from MolecularDiffusion.utils.geom_utils import sample_gaussian_with_mask

        torch.manual_seed(42)
        size = (1, 100, 3)
        mask = _make_mask(1, 100, 100)
        out = sample_gaussian_with_mask(size, "cpu", mask)
        assert out.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# sample_center_gravity_zero_gaussian_with_mask
# ---------------------------------------------------------------------------

class TestSampleCenterGravityZeroGaussian:
    def test_zero_center_of_mass(self):
        from MolecularDiffusion.utils.geom_utils import sample_center_gravity_zero_gaussian_with_mask

        torch.manual_seed(0)
        size = (4, 10, 3)
        mask = _make_mask(4, 10, 8)
        out = sample_center_gravity_zero_gaussian_with_mask(size, "cpu", mask)
        com = (out * mask).sum(dim=1) / mask.sum(dim=1)
        assert com.abs().max().item() < 1e-4

    def test_output_shape(self):
        from MolecularDiffusion.utils.geom_utils import sample_center_gravity_zero_gaussian_with_mask

        size = (2, 6, 3)
        mask = _make_mask(2, 6, 6)
        out = sample_center_gravity_zero_gaussian_with_mask(size, "cpu", mask)
        assert out.shape == torch.Size(size)


# ---------------------------------------------------------------------------
# random_rotation
# ---------------------------------------------------------------------------

class TestRandomRotation:
    def test_preserves_pairwise_distances(self):
        from MolecularDiffusion.utils.geom_utils import random_rotation

        torch.manual_seed(7)
        x = torch.randn(2, 5, 3)
        xr = random_rotation(x)
        # pairwise distances should be invariant under rotation
        def pdist(t):
            diff = t.unsqueeze(2) - t.unsqueeze(1)
            return (diff ** 2).sum(-1)

        assert torch.allclose(pdist(x), pdist(xr), atol=1e-4)

    def test_output_shape_unchanged(self):
        from MolecularDiffusion.utils.geom_utils import random_rotation

        x = torch.randn(3, 7, 3)
        assert random_rotation(x).shape == x.shape

    def test_2d_rotation(self):
        from MolecularDiffusion.utils.geom_utils import random_rotation

        x = torch.randn(2, 4, 2)
        xr = random_rotation(x)
        assert xr.shape == x.shape


# ---------------------------------------------------------------------------
# coord2diff
# ---------------------------------------------------------------------------

class TestCoord2Diff:
    def test_output_shapes(self):
        from MolecularDiffusion.utils.geom_utils import coord2diff

        x = torch.randn(4, 3)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        radial, diff = coord2diff(x, edge_index)
        assert radial.shape == (3, 1)
        assert diff.shape == (3, 3)

    def test_self_loop_radial_is_zero(self):
        from MolecularDiffusion.utils.geom_utils import coord2diff

        x = torch.randn(3, 3)
        edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])  # self-loops
        radial, _ = coord2diff(x, edge_index)
        assert radial.abs().max().item() < 1e-6

    def test_symmetry(self):
        from MolecularDiffusion.utils.geom_utils import coord2diff

        x = torch.randn(3, 3)
        fwd = torch.tensor([[0], [1]])
        bwd = torch.tensor([[1], [0]])
        r_fwd, d_fwd = coord2diff(x, fwd)
        r_bwd, d_bwd = coord2diff(x, bwd)
        assert torch.allclose(r_fwd, r_bwd, atol=1e-6)
        assert torch.allclose(d_fwd, -d_bwd, atol=1e-6)


# ---------------------------------------------------------------------------
# assert_mean_zero_with_mask
# ---------------------------------------------------------------------------

class TestAssertMeanZeroWithMask:
    def test_passes_for_zero_mean(self):
        from MolecularDiffusion.utils.geom_utils import assert_mean_zero_with_mask

        B, N, D = 2, 5, 3
        mask = _make_mask(B, N, 4)
        x = torch.zeros(B, N, D)
        assert_mean_zero_with_mask(x, mask)  # should not raise

    def test_raises_for_nonzero_mean(self):
        from MolecularDiffusion.utils.geom_utils import assert_mean_zero_with_mask

        B, N, D = 1, 4, 3
        mask = _make_mask(B, N, 4)
        x = torch.ones(B, N, D) * mask
        with pytest.raises(Exception):
            assert_mean_zero_with_mask(x, mask)
