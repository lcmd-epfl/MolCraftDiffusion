"""``ConformerFactory``: grouping, mapping, metrics, and the knobs it refuses.

Uses a fake task rather than a checkpoint -- the mode only needs the duck-typed
``conditioning_pool()`` / ``sample(mols=...)`` contract.
"""

import pandas as pd
import pytest
import torch

from MolecularDiffusion.modules.tasks.diffusion_loqi import load_conditioning_pool
from MolecularDiffusion.runmodes.generate.tasks_conformer import ConformerFactory

VOCAB = ["H", "C", "N", "O", "F", "S", "Cl"]


class FakeConformerTask:
    """Returns the input geometry plus a fixed jitter -- one conformer per mol."""

    atom_vocab = VOCAB
    sample_input = "fake.sdf"

    def __init__(self) -> None:
        self._pool = load_conditioning_pool(["CCO", "CCC"], VOCAB)

    def conditioning_pool(self) -> list:
        return self._pool

    def sample(self, mols=None, **_kwargs):
        n_max = max(int(m.n_nodes) for m in mols)
        bs = len(mols)
        one_hot = torch.zeros(bs, n_max, len(VOCAB))
        charges = torch.zeros(bs, n_max, dtype=torch.long)
        coords = torch.zeros(bs, n_max, 3)
        node_mask = torch.zeros(bs, n_max, dtype=torch.long)
        for i, m in enumerate(mols):
            n = int(m.n_nodes)
            one_hot[i, :n] = torch.nn.functional.one_hot(
                m.atom_idx.long(), len(VOCAB)
            ).float()
            coords[i, :n] = m.pos + 0.01 * (i + 1)
            node_mask[i, :n] = 1
        return one_hot, charges, coords, node_mask


def test_output_is_grouped_and_mapped(tmp_path) -> None:
    out = tmp_path / "conf"
    ConformerFactory(
        task=FakeConformerTask(),
        conformers_per_molecule=3,
        batch_size=2,
        output_path=str(out),
    ).run()

    for idx in (0, 1):
        mol_dir = out / f"mol_{idx:04d}"
        assert sorted(p.name for p in mol_dir.glob("conformer_*.xyz")) == [
            "conformer_000.xyz",
            "conformer_001.xyz",
            "conformer_002.xyz",
        ]
        # the bond channel .xyz does not have
        assert (mol_dir / "conformers.sdf").read_text().count("$$$$") == 3
        assert (mol_dir / "reference.sdf").exists()

    df = pd.read_csv(out / "conformers.csv")
    assert len(df) == 6
    assert set(df["mol_index"]) == {0, 1}
    assert df["smiles"].nunique() == 2
    # every conformer maps back to a file that exists
    assert all((out / path).exists() for path in df["xyz"])
    # jitter-only geometry => tiny RMSD to the reference
    assert df["rmsd"].notna().all()
    assert float(df["rmsd"].max()) < 0.1


@pytest.mark.parametrize("knob", ["mol_size", "num_generate", "max_mol_size"])
def test_denovo_knobs_are_refused_not_ignored(knob) -> None:
    with pytest.raises(ValueError, match="meaningless for conformer generation"):
        ConformerFactory(task=FakeConformerTask(), **{knob: 16})


def test_non_conformer_task_is_rejected() -> None:
    with pytest.raises(TypeError, match="not a conformer generator"):
        ConformerFactory(task=object())
