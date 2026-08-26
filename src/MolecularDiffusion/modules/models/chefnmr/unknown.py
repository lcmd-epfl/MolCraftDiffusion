"""Unknown-spectra input for ChefNMR: a spectrum and a formula, nothing else.

The converted benchmark corpus (``<prefix>.db`` plus four sidecars) can only
describe molecules whose 3D coordinates *and* SMILES you already have -- that
is, only the ones whose answer is known. It is the right input for
reproducing published numbers and the wrong one for the actual use case:
*here is a spectrum of an unknown, what is it?*

Generation needs exactly two things: the spectrum and the atom composition.
The formula stays mandatory on purpose -- in real NMR elucidation it comes
from high-resolution mass spectrometry, and upstream trains with
``known_atoms: True``. Coordinates are not needed here and are not asked for.

**Format: JSON.** One object per unknown, or a list of them::

    [
      {
        "name": "vanillin_unknown",
        "formula": "C8H8O3",
        "13C": {"peaks": [190.9, 151.7, 147.2, 129.9, 127.5,
                          114.4, 108.8, 56.1]},
        "1H":  {"peaks": [[9.82, 1.0], [7.44, 2.0], [7.04, 1.0],
                          [6.20, 1.0], [3.95, 3.0]]}
      }
    ]

CSV was the alternative and does not fit: a peak list is a variable-length
nested array and a pre-binned channel is 10,000 floats, so neither goes in a
cell without an encoding a human then has to remember.

Keys, per record:

``name``
    Required. Names the output directory, so it must be unique in the file.
``formula`` **or** ``atoms``
    Exactly one. ``formula`` is a plain molecular formula (``"C8H8O3"``);
    ``atoms`` is an explicit per-atom list (``["C", "C", "O", "H", ...]``),
    which is what an exporter writes when it wants to preserve a specific
    atom ordering.
``smiles``
    Optional, and only ever used for *scoring*. Supply a suspected structure
    and the run reports top-k against it; leave it out and the run is a
    genuine unknown -- :meth:`_reference` returns ``None``, and the shared
    seam then writes no ``metrics.json``.
``"1H"`` / ``"13C"``
    Optional (a missing channel is read by the model as absent, via its
    learned mask token). Each is an object holding **exactly one** of:

    ``peaks``
        A list of ppm shifts (``[190.9, 151.7]``) or of
        ``[ppm, intensity]`` pairs. Binned here onto the model's own grid.
    ``binned``
        A vector already on the model's grid, exactly ``n_bins`` long.

    ``peaks`` may be joined by ``linewidth`` (FWHM in ppm) on a non-binary
    channel; see :func:`bin_peaks`.

Grids, per upstream ``meta/grids/*.p`` (README lines 36-42). Each pickle was
opened and checked against ``np.linspace`` here: max abs difference 3e-14, so
the analytic form is the grid.

======  =======  ====================  ===========================
nuclei  n_bins   ppm range             semantics
======  =======  ====================  ===========================
1H      10 000   [-2, 10]              lineshape, peak scaled to 1
13C     10 000   [-20, 230]            lineshape, peak scaled to 1
13C     80       [3.423975, 231.3]     **binary** occupancy, {0, 1}
======  =======  ====================  ===========================

**The 80-bin 13C channel is an indicator, not intensities.** Measured over
both converted corpora: every value is exactly 0.0 or 1.0
(``c_all_binary: true`` in each ``_meta.json``), and the ``embed`` tokenizer
consumes it as ``x.long() * arange(1, L+1)`` into an
``nn.Embedding(L+1, D, padding_idx=0)``, whose index arithmetic is only
correct on ``{0, 1}``. Binning a peak list onto it therefore sets 1.0 and
ignores intensities. The 1H 10k channel is *not* binary -- mean non-zero
fraction 43.4% over 650,313 USPTO rows, continuous values down to 5e-6 -- so
it gets a real lineshape instead. Binary-ness is read off the loaded
checkpoint's tokenizer as well as this table, because the tokenizer is what
actually consumes the vector.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

#: ``(nuclei, n_bins) -> (first ppm, last ppm, binary)``. Verified against
#: upstream's ``meta/grids/{H10k,C10k,C80}.p``: ``np.linspace`` over these
#: endpoints reproduces every pickle to 3e-14.
_GRIDS: dict[tuple[str, int], tuple[float, float, bool]] = {
    ("1H", 10000): (-2.0, 10.0, False),
    ("13C", 10000): (-20.0, 230.0, False),
    ("13C", 80): (3.423975, 231.3, True),
}

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]*)")
_PEAK_PAIR = 2


@dataclass(frozen=True)
class Channel:
    """One measurement channel's slot in the concatenated condition vector.

    Attributes:
        name: the nuclei, as :attr:`ChefNMRElucidationGenerator
            .maskable_channels` names it (``"1H"``, ``"13C"``).
        n_bins: width of this channel's slice. ``0`` when the loaded
            checkpoint has no branch for it at all.
        trained: whether the checkpoint has this branch. A spectrum
            supplied for an untrained channel is refused rather than
            silently discarded.
        binary: whether the channel is a ``{0, 1}`` occupancy indicator.
            Read off the checkpoint's tokenizer, not guessed from the data.
    """

    name: str
    n_bins: int
    trained: bool
    binary: bool


def grid(channel: str, n_bins: int) -> np.ndarray:
    """The ppm axis a peak list is binned onto.

    Args:
        channel: ``"1H"`` or ``"13C"``.
        n_bins: the checkpoint's width for that channel.

    Returns:
        The ``(n_bins,)`` ppm axis. Grid points are bin *centres*, and both
        endpoints are on the grid.

    Raises:
        ValueError: when no grid is published for this pairing -- the ppm
            axis is then unknown and a peak list cannot be placed on it.
    """
    entry = _GRIDS.get((channel, int(n_bins)))
    if entry is None:
        published = sorted(f"{c} x {n}" for c, n in _GRIDS)
        msg = (
            f"no published ppm grid for a {n_bins}-bin {channel} channel, so "
            f"a peak list cannot be placed on it. Upstream ships {published} "
            "(meta/grids/*.p). Either load a checkpoint trained on one of "
            'those, or supply this channel as {"binned": [...]} already on '
            "your own grid."
        )
        raise ValueError(msg)
    first, last, _ = entry
    return np.linspace(first, last, int(n_bins))


def bin_peaks(
    peaks: Any,
    channel: Channel,
    linewidth: float | None = None,
    where: str = "",
) -> np.ndarray:
    """Place a peak list on ``channel``'s grid.

    A binary channel gets a nearest-bin 1.0 per peak and intensities are
    ignored, because that is what the channel holds (see the module
    docstring). Everything else gets a sum of Lorentzians scaled so the tallest
    point is 1.0 -- the normalisation the training data uses.

    Args:
        peaks: ppm shifts, or ``[ppm, intensity]`` pairs.
        channel: which slot of the condition vector this is.
        linewidth: FWHM in ppm, on a non-binary channel. ``None`` => one
            grid step, which for the 1H 10k grid is 0.0012 ppm. That is the
            **measured** line width: fitting a Lorentzian to isolated
            singlets in the USPTO corpus puts the best HWHM at 0.0006 ppm
            (18% max relative error, against 24% at twice that and 54% at
            four times), and the median singlet FWHM there is 2 grid bins.
            Raise it if your peaks came off an instrument with broader
            lines than the simulated training spectra.
        where: prefix for error messages, naming the record.

    Returns:
        A ``(channel.n_bins,)`` float32 vector.

    Raises:
        ValueError: on a malformed peak, a shift off the end of the grid, a
            non-positive intensity, or a non-positive ``linewidth``.
    """
    axis = grid(channel.name, channel.n_bins)
    shifts, heights = _peak_pairs(peaks, where)

    low, high = float(axis[0]), float(axis[-1])
    outside = shifts[(shifts < low) | (shifts > high)]
    if outside.size:
        msg = (
            f"{where}: {outside.tolist()} ppm is outside the {channel.name} "
            f"{channel.n_bins}-bin grid [{low:g}, {high:g}] this checkpoint "
            "was trained on. A shift the grid cannot hold would be silently "
            "dropped, so it is refused instead -- check the nuclei and the "
            "referencing."
        )
        raise ValueError(msg)

    if channel.binary:
        if not np.all(heights == 1.0):
            logger.warning(
                "[chefnmr] %s: the %s %d-bin channel is a BINARY occupancy "
                "indicator, so the intensities you gave are ignored; only "
                "the shifts are used.",
                where,
                channel.name,
                channel.n_bins,
            )
        out = np.zeros(channel.n_bins, dtype=np.float32)
        nearest = np.abs(axis[None, :] - shifts[:, None]).argmin(axis=1)
        out[nearest] = 1.0
        return out

    if np.any(heights <= 0.0):
        msg = (
            f"{where}: peak intensities must be positive on the "
            f"{channel.name} channel; got {heights.tolist()}."
        )
        raise ValueError(msg)
    step = float(axis[1] - axis[0])
    fwhm = step if linewidth is None else float(linewidth)
    if fwhm <= 0.0:
        msg = f"{where}: linewidth must be positive, got {linewidth!r}."
        raise ValueError(msg)
    offsets = (axis[None, :] - shifts[:, None]) / (fwhm / 2.0)
    out = (heights[:, None] / (1.0 + offsets**2)).sum(axis=0)
    return (out / out.max()).astype(np.float32)


def formula_to_symbols(
    formula: str, decoder: Sequence[str], where: str = ""
) -> list[str]:
    """``"C8H8O3"`` -> one symbol per atom, in ``decoder`` order.

    The order is arbitrary as far as the model is concerned -- the denoiser
    has no positional encoding on the atom axis -- but it is fixed here so
    two runs of the same formula draw the same noise for the same atom.

    Args:
        formula: a plain molecular formula. Parentheses, charges, dots and
            hydrates are not parsed; use ``atoms`` for anything else.
        decoder: the model's ``atom_decoder``.
        where: prefix for error messages, naming the record.

    Returns:
        One element symbol per atom.

    Raises:
        ValueError: on a formula this parser does not accept, an element
            outside ``decoder``, or a zero count.
    """
    text = str(formula).strip().replace(" ", "")
    counts: dict[str, int] = {}
    position = 0
    for match in _FORMULA_TOKEN.finditer(text):
        if match.start() != position:
            break
        position = match.end()
        element = match.group(1)
        counts[element] = counts.get(element, 0) + int(match.group(2) or 1)
    if position != len(text) or not counts:
        msg = (
            f"{where}: {formula!r} is not a plain molecular formula. Write "
            "element symbols with optional counts and nothing else, e.g. "
            "'C8H8O3'. Parentheses, charges, dots and hydrates are not "
            'parsed -- expand them yourself, or use "atoms" with one symbol '
            "per atom."
        )
        raise ValueError(msg)
    empty = sorted(e for e, n in counts.items() if n <= 0)
    if empty:
        msg = f"{where}: {formula!r} gives {empty} a count of zero."
        raise ValueError(msg)
    _check_vocab(counts, decoder, where)
    symbols: list[str] = []
    for element in decoder:
        symbols.extend([element] * counts.get(element, 0))
    return symbols


def read_unknown_spectra(
    path: str,
    channels: Sequence[Channel],
    decoder: Sequence[str],
    max_n_atoms: int,
) -> list[dict[str, Any]]:
    """Parse an unknown-spectra file into ready-to-use records.

    Args:
        path: the JSON file. See the module docstring for the format.
        channels: the condition vector's layout, read off the loaded
            checkpoint. Concatenated in this order.
        decoder: the model's ``atom_decoder``. Position-in-vocab is the
            one-hot column, so an element outside it has nowhere to go.
        max_n_atoms: the largest molecule these weights were trained on.

    Returns:
        One dict per record with ``name``, ``symbols``, ``smiles`` (possibly
        ``None``) and ``cond`` -- a ``(sum(n_bins),)`` float32 vector.

    Raises:
        ValueError: on anything malformed. Every check names the record.
    """
    try:
        raw = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        msg = f"{path} is not valid JSON: {exc}"
        raise ValueError(msg) from exc

    entries = [raw] if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        msg = (
            f"{path} must hold one unknown-spectra object or a non-empty "
            f"list of them; got {type(raw).__name__}."
        )
        raise ValueError(msg)

    allowed = {"name", "formula", "atoms", "smiles"}
    allowed |= {c.name for c in channels}
    records: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        records.append(
            _record(
                entry, index, path, allowed, channels, decoder, max_n_atoms
            )
        )

    names = [r["name"] for r in records]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        msg = (
            f"{path}: duplicate record name(s) {duplicates}. The name is the "
            "output directory, so duplicates would overwrite each other."
        )
        raise ValueError(msg)
    return records


def _record(  # noqa: PLR0913
    entry: Any,
    index: int,
    path: str,
    allowed: set[str],
    channels: Sequence[Channel],
    decoder: Sequence[str],
    max_n_atoms: int,
) -> dict[str, Any]:
    """One parsed record; see :func:`read_unknown_spectra`."""
    where = f"{path}[{index}]"
    if not isinstance(entry, dict):
        msg = f"{where}: expected an object, got {type(entry).__name__}."
        # One error type throughout this file on purpose, so a caller
        # can catch one thing and print it.
        raise ValueError(msg)  # noqa: TRY004
    extra = sorted(set(entry) - allowed)
    if extra:
        msg = (
            f"{where}: unrecognised key(s) {extra}. Accepted here: "
            f"{sorted(allowed)}. Silently ignoring a key would make the run "
            "look configured when it is not."
        )
        raise ValueError(msg)
    name = entry.get("name")
    if not name or not isinstance(name, str):
        msg = f"{where}: every record needs a non-empty string 'name'."
        raise ValueError(msg)
    where = f"{path}[{index}] {name!r}"

    symbols = _symbols(entry, decoder, max_n_atoms, where)
    cond = np.concatenate(
        [_channel_vector(entry, c, where) for c in channels]
    ).astype(np.float32)
    if not cond.any():
        wanted = [c.name for c in channels if c.trained]
        msg = (
            f"{where}: no spectrum at all. An all-zero condition is the "
            "classifier-free UNCONDITIONAL branch, so the model would emit "
            "a plausible molecule of the right formula that has nothing to "
            f"do with any measurement. Give it at least one of {wanted}."
        )
        raise ValueError(msg)
    smiles = entry.get("smiles")
    return {
        "name": name,
        "symbols": symbols,
        "smiles": str(smiles) if smiles else None,
        "cond": cond,
    }


def _symbols(
    entry: dict[str, Any],
    decoder: Sequence[str],
    max_n_atoms: int,
    where: str,
) -> list[str]:
    """The atom multiset, from ``formula`` or from ``atoms``."""
    has_formula = entry.get("formula") is not None
    has_atoms = entry.get("atoms") is not None
    if has_formula == has_atoms:
        msg = (
            f"{where}: give exactly one of 'formula' (e.g. \"C8H8O3\") or "
            '\'atoms\' (e.g. ["C", "C", "O", "H"]). The composition is '
            "an INPUT to this model -- in real elucidation it comes from "
            "high-resolution mass spec -- so it cannot be omitted."
        )
        raise ValueError(msg)

    if has_formula:
        symbols = formula_to_symbols(entry["formula"], decoder, where)
    else:
        atoms = entry["atoms"]
        if not isinstance(atoms, (list, tuple)) or not atoms:
            msg = f"{where}: 'atoms' must be a non-empty list of symbols."
            raise ValueError(msg)
        symbols = [str(a) for a in atoms]
        counts: dict[str, int] = {}
        for symbol in symbols:
            counts[symbol] = counts.get(symbol, 0) + 1
        _check_vocab(counts, decoder, where)

    if len(symbols) > max_n_atoms:
        msg = (
            f"{where}: {len(symbols)} atoms exceeds max_n_atoms="
            f"{max_n_atoms}, the largest molecule these weights were trained "
            "on. The molecule is NOT truncated to fit -- a partial formula "
            "would be a different compound. Use a checkpoint trained on "
            "larger molecules."
        )
        raise ValueError(msg)
    return symbols


def _check_vocab(
    counts: dict[str, int], decoder: Sequence[str], where: str
) -> None:
    """Refuse any element the model has no one-hot column for."""
    unknown = sorted(e for e in counts if e not in decoder)
    if unknown:
        msg = (
            f"{where}: {unknown} is outside the model's atom_decoder "
            f"{list(decoder)}. The one-hot column is position-in-vocab, so "
            "there is no column to put it in -- this checkpoint cannot "
            "elucidate that molecule."
        )
        raise ValueError(msg)


def _channel_vector(
    entry: dict[str, Any], channel: Channel, where: str
) -> np.ndarray:
    """One channel's slice of the condition vector."""
    spec = entry.get(channel.name)
    if spec is None:
        # Absent, not zeroed-on-purpose: an all-zero channel is exactly how
        # the embedder detects a missing modality and swaps in its learned
        # mask token. A 13C-only unknown is a normal, supported input.
        return np.zeros(max(channel.n_bins, 0), dtype=np.float32)
    if not channel.trained:
        msg = (
            f"{where}: this checkpoint has no {channel.name} branch (its "
            "condition_type excludes it), so that spectrum would never be "
            "read. Remove it, or load a checkpoint trained on "
            f"{channel.name}."
        )
        raise ValueError(msg)
    if not isinstance(spec, dict):
        msg = (
            f"{where}: {channel.name!r} must be an object holding exactly "
            'one of "peaks" or "binned".'
        )
        raise ValueError(msg)  # noqa: TRY004 - see _record

    keys = set(spec)
    if keys not in ({"peaks"}, {"peaks", "linewidth"}, {"binned"}):
        msg = (
            f"{where}: {channel.name!r} holds {sorted(keys)}; it takes "
            'exactly one of "peaks" (optionally with "linewidth") or '
            '"binned".'
        )
        raise ValueError(msg)

    if "binned" in spec:
        vector = np.asarray(spec["binned"], dtype=np.float32)
        if vector.shape != (channel.n_bins,):
            msg = (
                f"{where}: {channel.name!r} 'binned' has shape "
                f"{vector.shape}, but this checkpoint reads "
                f"{channel.n_bins} bins for that channel."
            )
            raise ValueError(msg)
        published = _GRIDS.get((channel.name, channel.n_bins))
        binary = channel.binary or bool(published and published[2])
        if binary and not np.isin(vector, (0.0, 1.0)).all():
            msg = (
                f"{where}: the {channel.name} {channel.n_bins}-bin channel "
                "is a BINARY occupancy indicator -- its tokenizer indexes an "
                "nn.Embedding with x.long(), so anything other than 0.0/1.0 "
                "is read as a bin number. Threshold your vector first."
            )
            raise ValueError(msg)
        return vector

    return bin_peaks(spec["peaks"], channel, spec.get("linewidth"), where)


def _peak_pairs(peaks: Any, where: str) -> tuple[np.ndarray, np.ndarray]:
    """``[ppm, ...]`` or ``[[ppm, height], ...]`` -> two aligned arrays."""
    if not isinstance(peaks, (list, tuple)) or not peaks:
        msg = f"{where}: 'peaks' must be a non-empty list."
        raise ValueError(msg)
    shifts: list[float] = []
    heights: list[float] = []
    for peak in peaks:
        if isinstance(peak, (int, float)) and not isinstance(peak, bool):
            shifts.append(float(peak))
            heights.append(1.0)
        elif isinstance(peak, (list, tuple)) and len(peak) == _PEAK_PAIR:
            shifts.append(float(peak[0]))
            heights.append(float(peak[1]))
        else:
            msg = (
                f"{where}: each peak is a ppm shift (7.45) or a "
                f"[ppm, intensity] pair ([7.45, 2.0]); got {peak!r}."
            )
            raise ValueError(msg)
    return np.asarray(shifts, dtype=float), np.asarray(heights, dtype=float)
