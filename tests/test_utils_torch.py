"""Unit tests for MolecularDiffusion.utils.torch."""

import pytest
import torch
from torch import nn


# ---------------------------------------------------------------------------
# cpu() helper
# ---------------------------------------------------------------------------

class TestCpuHelper:
    def test_tensor(self):
        from MolecularDiffusion.utils.torch import cpu

        t = torch.randn(3, 4)
        assert cpu(t).device.type == "cpu"

    def test_dict_of_tensors(self):
        from MolecularDiffusion.utils.torch import cpu

        d = {"a": torch.randn(2), "b": torch.randn(3)}
        out = cpu(d)
        assert all(v.device.type == "cpu" for v in out.values())

    def test_list_of_tensors(self):
        from MolecularDiffusion.utils.torch import cpu

        lst = [torch.randn(2), torch.randn(3)]
        out = cpu(lst)
        assert all(v.device.type == "cpu" for v in out)

    def test_tuple_of_tensors(self):
        from MolecularDiffusion.utils.torch import cpu

        tup = (torch.randn(2), torch.randn(3))
        out = cpu(tup)
        assert isinstance(out, tuple)
        assert all(v.device.type == "cpu" for v in out)

    def test_string_passthrough(self):
        from MolecularDiffusion.utils.torch import cpu

        assert cpu("hello") == "hello"

    def test_bytes_passthrough(self):
        from MolecularDiffusion.utils.torch import cpu

        assert cpu(b"bytes") == b"bytes"

    def test_unsupported_type_raises(self):
        from MolecularDiffusion.utils.torch import cpu

        # Objects with no cpu() method and not a known scalar/container should raise
        class _Opaque:
            pass

        with pytest.raises(TypeError):
            cpu(_Opaque())

    def test_nested_dict(self):
        from MolecularDiffusion.utils.torch import cpu

        nested = {"outer": {"inner": torch.randn(2)}}
        out = cpu(nested)
        assert out["outer"]["inner"].device.type == "cpu"


# ---------------------------------------------------------------------------
# recursive_module_to_device
# ---------------------------------------------------------------------------

class TestRecursiveModuleToDevice:
    def test_linear_on_cpu(self):
        from MolecularDiffusion.utils.torch import recursive_module_to_device

        model = nn.Linear(4, 4)
        device = torch.device("cpu")
        recursive_module_to_device(model, device)
        assert next(model.parameters()).device.type == "cpu"

    def test_sequential_on_cpu(self):
        from MolecularDiffusion.utils.torch import recursive_module_to_device

        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        device = torch.device("cpu")
        recursive_module_to_device(model, device)
        for p in model.parameters():
            assert p.device.type == "cpu"

    def test_device_attribute_set(self):
        from MolecularDiffusion.utils.torch import recursive_module_to_device

        model = nn.Linear(2, 2)
        device = torch.device("cpu")
        recursive_module_to_device(model, device)
        assert model.device == device


# ---------------------------------------------------------------------------
# Package-level __init__ lazy loading
# ---------------------------------------------------------------------------

class TestPackageInit:
    def test_version_present(self):
        import MolecularDiffusion

        assert hasattr(MolecularDiffusion, "__version__")
        assert MolecularDiffusion.__version__ != ""

    def test_lazy_submodule_utils(self):
        from MolecularDiffusion import utils

        assert utils is not None

    def test_lazy_submodule_data(self):
        from MolecularDiffusion import data

        assert data is not None

    def test_lazy_submodule_core(self):
        pytest.importorskip("MolecularDiffusion.core",
                            reason="core requires optional native extensions")
        from MolecularDiffusion import core
        assert core is not None

    def test_lazy_submodule_callbacks(self):
        pytest.importorskip("MolecularDiffusion.callbacks",
                            reason="callbacks requires optional native extensions")
        from MolecularDiffusion import callbacks
        assert callbacks is not None

    def test_lazy_submodule_runmodes(self):
        pytest.importorskip("MolecularDiffusion.runmodes",
                            reason="runmodes requires optional native extensions")
        from MolecularDiffusion import runmodes
        assert runmodes is not None

    def test_invalid_submodule_raises(self):
        import MolecularDiffusion

        with pytest.raises(AttributeError):
            _ = MolecularDiffusion.nonexistent_module

    def test_submodule_cached_after_first_access(self):
        import MolecularDiffusion
        from MolecularDiffusion import utils as u1
        from MolecularDiffusion import utils as u2

        assert u1 is u2
