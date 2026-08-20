"""Caller-owned generate-config keys (cli/generate.py)."""

from omegaconf import OmegaConf

from MolecularDiffusion.cli.generate import _CALLER_OWNED_KEYS, _caller_overrides


class _PlainFactory:
    """A factory that declares nothing — the pre-existing behaviour."""


class _DeclaringFactory:
    generation_time_keys = ("sample_input",)


def _cfg(cls, **kw):
    target = f"{__name__}.{cls.__name__}"
    return OmegaConf.create({"_target_": target, **kw})


def test_undeclared_key_is_not_overridden():
    ckpt = _cfg(_PlainFactory, sample_input=None)
    caller = _cfg(_PlainFactory, sample_input="pool.sdf")
    assert _caller_overrides(ckpt, caller) == {}


def test_declared_key_is_overridden():
    ckpt = _cfg(_DeclaringFactory, sample_input=None)
    caller = _cfg(_DeclaringFactory, sample_input="pool.sdf")
    assert _caller_overrides(ckpt, caller) == {"sample_input": "pool.sdf"}


def test_base_keys_still_apply_and_none_never_overrides():
    ckpt = _cfg(_PlainFactory, chkpt_path="/cluster/old.ckpt", vae_ckpt="/cluster/vae.ckpt")
    caller = _cfg(_PlainFactory, chkpt_path="/local/new.ckpt", vae_ckpt=None)
    assert _caller_overrides(ckpt, caller, _CALLER_OWNED_KEYS) == {
        "chkpt_path": "/local/new.ckpt"
    }


def test_missing_or_unresolvable_configs_are_inert():
    assert _caller_overrides(None, _cfg(_DeclaringFactory)) == {}
    assert _caller_overrides(_cfg(_DeclaringFactory), None) == {}
    bad = OmegaConf.create({"_target_": "no.such.module.Factory"})
    assert _caller_overrides(bad, OmegaConf.create({"sample_input": "x"})) == {}
