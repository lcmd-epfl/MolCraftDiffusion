"""Unit tests for MolecularDiffusion.utils.diffusion_utils."""

import pytest
import torch


# ---------------------------------------------------------------------------
# compute_mean_mad_from_dataloader
# ---------------------------------------------------------------------------

class TestComputeMeanMad:
    def _run(self, values_list, names):
        from MolecularDiffusion.utils.diffusion_utils import compute_mean_mad_from_dataloader
        return compute_mean_mad_from_dataloader(values_list, names)

    def test_keys_match_task_names(self):
        names = ["energy", "gap"]
        props = [torch.randn(100), torch.randn(100)]
        norms = self._run(props, names)
        assert set(norms.keys()) == set(names)

    def test_mean_is_correct(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        norms = self._run([values], ["x"])
        assert abs(norms["x"]["mean"].item() - 3.0) < 1e-5

    def test_mad_is_nonnegative(self):
        values = torch.randn(200)
        norms = self._run([values], ["y"])
        assert norms["y"]["mad"].item() >= 0.0

    def test_constant_input_mad_is_zero(self):
        values = torch.ones(50) * 7.0
        norms = self._run([values], ["c"])
        assert norms["c"]["mad"].item() < 1e-6

    def test_max_min_bounds(self):
        values = torch.tensor([0.0, 1.0, 2.0, 3.0])
        norms = self._run([values], ["v"])
        assert abs(norms["v"]["max"].item() - 3.0) < 1e-5
        assert abs(norms["v"]["min"].item() - 0.0) < 1e-5

    def test_multiple_properties(self):
        names = ["a", "b", "c"]
        props = [torch.randn(50) for _ in names]
        norms = self._run(props, names)
        assert len(norms) == 3


# ---------------------------------------------------------------------------
# prepare_context
# ---------------------------------------------------------------------------

class TestPrepareContext:
    def _make_batch(self, B, N):
        return {
            "coords": torch.randn(B, N, 3),
            "node_mask": torch.ones(B, N),
            "energy": torch.randn(B),
            "gap": torch.randn(B),
        }

    def _make_norms(self, task_names, values):
        norms = {}
        for name, v in zip(task_names, values):
            norms[name] = {
                "mean": torch.mean(v),
                "mad": torch.mean(torch.abs(v - torch.mean(v))) + 1e-8,
                "max": torch.max(v),
                "min": torch.min(v),
            }
        return norms

    def test_output_shape_global_features(self):
        from MolecularDiffusion.utils.diffusion_utils import prepare_context

        B, N = 3, 8
        batch = self._make_batch(B, N)
        names = ["energy", "gap"]
        norms = self._make_norms(names, [batch[n] for n in names])
        ctx = prepare_context(names, batch, norms)
        assert ctx.shape == (B, N, len(names))

    def test_maxmin_normalization_range(self):
        from MolecularDiffusion.utils.diffusion_utils import prepare_context

        B, N = 4, 6
        batch = self._make_batch(B, N)
        names = ["energy"]
        # Extreme values so normalization is tight
        batch["energy"] = torch.tensor([0.0, 1.0, 2.0, 3.0])
        norms = self._make_norms(names, [batch["energy"]])
        ctx = prepare_context(names, batch, norms, normalization_method="maxmin")
        # All values should be in [-1, 1]
        assert ctx.abs().max().item() <= 1.0 + 1e-5

    def test_no_normalization(self):
        from MolecularDiffusion.utils.diffusion_utils import prepare_context

        B, N = 2, 5
        batch = self._make_batch(B, N)
        names = ["energy"]
        norms = self._make_norms(names, [batch["energy"]])
        ctx = prepare_context(names, batch, norms, normalization_method=None)
        assert ctx.shape == (B, N, 1)

    def test_mad_normalization(self):
        from MolecularDiffusion.utils.diffusion_utils import prepare_context

        B, N = 2, 4
        batch = self._make_batch(B, N)
        names = ["energy"]
        norms = self._make_norms(names, [batch["energy"]])
        ctx = prepare_context(names, batch, norms, normalization_method="mad")
        assert ctx.shape == (B, N, 1)

    def test_value_normalization(self):
        from MolecularDiffusion.utils.diffusion_utils import prepare_context

        B, N = 2, 4
        batch = self._make_batch(B, N)
        names = ["energy"]
        batch["energy"] = torch.ones(B) * 10.0
        norms = self._make_norms(names, [batch["energy"]])
        ctx = prepare_context(names, batch, norms, normalization_method="value_10")
        # 10 / 10 = 1.0 for each node
        assert torch.allclose(ctx[:, :, 0], torch.ones(B, N), atol=1e-5)

    def test_invalid_normalization_raises(self):
        from MolecularDiffusion.utils.diffusion_utils import prepare_context

        B, N = 2, 4
        batch = self._make_batch(B, N)
        names = ["energy"]
        norms = self._make_norms(names, [batch["energy"]])
        with pytest.raises(ValueError):
            prepare_context(names, batch, norms, normalization_method="unknown")

    def test_masked_context_is_zero(self):
        from MolecularDiffusion.utils.diffusion_utils import prepare_context

        B, N = 2, 6
        batch = self._make_batch(B, N)
        batch["node_mask"] = torch.zeros(B, N)
        batch["node_mask"][:, :3] = 1.0
        names = ["energy"]
        norms = self._make_norms(names, [batch["energy"]])
        ctx = prepare_context(names, batch, norms)
        assert ctx[:, 3:, :].abs().max().item() == 0.0
