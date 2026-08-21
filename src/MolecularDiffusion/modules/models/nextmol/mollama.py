"""MoLlama -- NExT-Mol's 1D half, INFERENCE ONLY.

A ~970M-parameter ``LlamaForCausalLM`` over a 192-token **SELFIES** vocabulary
(``acharkq/MoLlama``). It is sampled from a bare BOS token with no conditioning
of any kind, so the published checkpoint works out of the box.

Ports ``sample_selfies`` (``model/llm_pl.py:446``), ``reencode_selfies``
(``llm_pl.py:508``) and the validity filter inside ``sample_molecules``
(``llm_pl.py:326-340``). Fine-tuning -- the LoRA/freeze branches of
``init_llm``, the SELFIES datamodules and the ``unseen_selfies_tokens.txt``
vocabulary extension -- is deliberately out of scope; ``peft`` is therefore not
needed. Without fine-tuning the sampled molecules follow general drug-like
chemistry rather than a specific training distribution, so Novelty/FCD against
QM9-2014 or GEOM-Drugs will not reproduce the paper. Validity and everything
DMT is responsible for are unaffected.

Two traps this module exists to get right:

1. **``eos_token_id``.** ``generation_config.json`` says 0 and so does
   ``config.json``; ``special_tokens_map.json`` maps BOTH ``bos_token`` and
   ``eos_token`` to ``<s>`` = 0. Upstream reads ``tokenizer.bos_token_id`` /
   ``tokenizer.eos_token_id``, which here are the same id, and that is correct:
   MoLlama was pretrained with ``<s>`` terminating a sequence. Substituting
   ``</s>`` = 2 (the id a generic Llama tokenizer would give) means generation
   never terminates.
2. **``dtype=``.** ``transformers`` 5.x renamed ``from_pretrained(torch_dtype=)``
   to ``dtype=``.

``transformers`` is imported lazily so the rest of MolecularDiffusion still
imports without it.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

__all__ = ["load_mollama", "reencode_selfies", "sample_smiles"]

#: upstream's per-dataset SELFIES length cap (``max_sf_tokens``). QM9's is far
#: smaller in practice; 30 is the default carried by every PL module signature.
DEFAULT_MAX_SF_TOKENS = 30


def load_mollama(model_id: str, device, dtype=torch.bfloat16):
    """Load the LM and its fast tokenizer. Returns ``(model, tokenizer)``."""
    try:
        from transformers import AutoTokenizer, LlamaForCausalLM
    except ImportError as exc:  # pragma: no cover - environment guard
        msg = (
            "The NExT-Mol de-novo pipeline needs `transformers` for MoLlama. "
            "See docs/model_integrations/nextmol/INTEGRATION_PLAN.md, "
            "'Environment Prerequisites', for the exact pinned install command "
            "(an UNPINNED `pip install transformers` upgrades huggingface_hub, "
            "click and hf-xet -- do not run it)."
        )
        raise ImportError(msg) from exc

    # `dtype=`, not `torch_dtype=`: renamed in transformers 5.x.
    model = LlamaForCausalLM.from_pretrained(model_id, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = model.to(device).eval()
    return model, tokenizer


def reencode_selfies(selfies: str) -> tuple[str, str, str]:
    """``(selfies, smiles_with_chirality, smiles_without_chirality)``.

    Empty strings on any RDKit/SELFIES failure -- that is upstream's validity
    signal (``llm_pl.py:508``). The **third** column is what the diffusion half
    consumes (``qm9_jodo_dm.py:438``): DMT places atoms in 3D and derives
    chirality from the geometry, so feeding it a chirality-tagged SMILES would
    over-specify the input.
    """
    import selfies as sf
    from rdkit import Chem

    try:
        decoded = sf.decoder(selfies)
        mol = Chem.MolFromSmiles(decoded)
        if mol is None:
            return "", "", ""
        smiles_with_chirality = Chem.MolToSmiles(mol, kekuleSmiles=True)
        reencoded = sf.encoder(smiles_with_chirality)
        smiles_without_chirality = Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:  # noqa: BLE001 - any failure means "invalid", as upstream
        return "", "", ""
    return reencoded, smiles_with_chirality, smiles_without_chirality


@torch.no_grad()
def sample_smiles(  # noqa: PLR0913
    model_id: str,
    n: int,
    *,
    device=None,
    temperature: float = 1.0,
    num_beams: int = 1,
    max_sf_tokens: int = DEFAULT_MAX_SF_TOKENS,
    batch_size: int = 200,
    max_loops: int = 200,
    dtype=torch.bfloat16,
) -> list[tuple[str, str, str]]:
    """Sample until ``n`` valid molecules accumulate.

    Returns ``(selfies, smiles_with_chirality, smiles_without_chirality)``
    triples, sorted, exactly as upstream writes its TSV.

    ``max_loops`` bounds what is an unbounded ``while True`` upstream. A LM that
    emits nothing valid would otherwise hang forever; hitting the cap returns
    whatever accumulated and logs a warning rather than pretending success.
    """
    import selfies as sf

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_mollama(model_id, device, dtype=dtype)
    try:
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
        vocab = tokenizer.get_vocab()

        out: list[tuple[str, str, str]] = []
        for loop in range(max_loops):
            if len(out) >= n:
                break
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=device
            )
            generated = model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=temperature,
                num_beams=num_beams,
                # -1 for the BOS token, which is already in input_ids.
                max_new_tokens=max_sf_tokens - 1,
                min_length=1,
                eos_token_id=eos_token_id,
                num_return_sequences=1,
            )
            texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for text in texts:
                selfies, smi_chiral, smi = reencode_selfies(text)
                if not selfies:
                    continue
                # Reject anything whose re-encoded SELFIES uses a token the LM
                # does not have -- upstream's own filter (llm_pl.py:329-336).
                if any(tok not in vocab for tok in sf.split_selfies(selfies)):
                    continue
                out.append((selfies, smi_chiral, smi))
            logger.info(
                "MoLlama loop %d: %d/%d valid molecules so far", loop + 1, len(out), n
            )
        if len(out) < n:
            logger.warning(
                "MoLlama produced only %d/%d valid molecules in %d loops; "
                "returning what it has.",
                len(out),
                n,
                max_loops,
            )
    finally:
        # Free the ~2 GB before the diffusion half starts.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return sorted(out)[:n]
