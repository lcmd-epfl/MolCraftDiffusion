"""Device selection shared by training, generation and analysis.

Order is CUDA -> Apple MPS -> CPU, so a CUDA machine picks exactly what the
old ``"cuda" if torch.cuda.is_available() else "cpu"`` idiom picked; MPS only
enters the picture on a Mac, where CUDA can never be available.

``MOLCRAFT_DEVICE`` (e.g. ``cpu``, ``mps``, ``cuda:1``) overrides the choice.
"""

import os

import torch


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return backend is not None and backend.is_available()


def get_device() -> torch.device:
    """Return the preferred torch device: $MOLCRAFT_DEVICE, else cuda, mps, cpu."""
    override = os.environ.get("MOLCRAFT_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(obj, device, *args, **kwargs):
    """Move any nested container of tensors to ``device`` (device-generic ``utils.cuda``)."""
    if hasattr(obj, "to"):
        return obj.to(device, *args, **kwargs)
    elif isinstance(obj, (str, bytes)):
        return obj
    elif isinstance(obj, dict):
        return type(obj)({k: to_device(v, device, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_device(x, device, *args, **kwargs) for x in obj)

    raise TypeError("Can't transfer object type `%s`" % type(obj))
