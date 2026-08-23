"""SynCoGen: synthesizable 3D molecule generation over a building-block vocabulary.

Ported from https://github.com/andreirekesh/SynCoGen (commit
``6b38eec26ccc687b32808c37aefdf75a4a30f1da``, MIT). See ``vocab.py`` for the one
rule that governs importing anything in here: **call ``ensure_vocabulary`` before
importing any other submodule.** This module deliberately re-exports only that
function, so a plain ``import ...models.syncogen`` cannot trip the ordering.
"""

from MolecularDiffusion.modules.models.syncogen.vocab import ensure_vocabulary

__all__ = ["ensure_vocabulary"]
