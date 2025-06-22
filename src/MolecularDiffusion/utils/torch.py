

import torch

from MolecularDiffusion import data # TODO


def cpu(obj, *args, **kwargs):
    """
    Transfer any nested container of tensors to CPU.
    """
    if hasattr(obj, "cpu"):
        return obj.cpu(*args, **kwargs)
    elif isinstance(obj, (str, bytes)):
        return obj
    elif isinstance(obj, dict):
        return type(obj)({k: cpu(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(cpu(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't transfer object type `%s`" % type(obj))


def cuda(obj, *args, **kwargs):
    """
    Transfer any nested container of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, (str, bytes)):
        return obj
    elif isinstance(obj, dict):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't transfer object type `%s`" % type(obj))


def detach(obj):
    """
    Detach tensors in any nested conatiner.
    """
    if hasattr(obj, "detach"):
        return obj.detach()
    elif isinstance(obj, dict):
        return type(obj)({k: detach(v) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(detach(x) for x in obj)

    raise TypeError("Can't perform detach over object type `%s`" % type(obj))


def clone(obj, *args, **kwargs):
    """
    Clone tensors in any nested conatiner.
    """
    if hasattr(obj, "clone"):
        return obj.clone(*args, **kwargs)
    elif isinstance(obj, dict):
        return type(obj)({k: clone(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(clone(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't perform detach over object type `%s`" % type(obj))


def mean(obj, *args, **kwargs):
    """
    Compute mean of tensors in any nested container.
    """
    if hasattr(obj, "mean"):
        return obj.mean(*args, **kwargs)
    elif isinstance(obj, dict):
        return type(obj)({k: mean(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(mean(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't perform mean over object type `%s`" % type(obj))


def cat(objs, *args, **kwargs):
    """
    Concatenate a list of nested containers with the same structure.
    """
    obj = objs[0]
    if isinstance(obj, torch.Tensor):
        return torch.cat(objs, *args, **kwargs)
    elif isinstance(obj, data.PackedGraph):
        return data.cat(objs)
    elif isinstance(obj, dict):
        return {k: cat([x[k] for x in objs], *args, **kwargs) for k in obj}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(cat(xs, *args, **kwargs) for xs in zip(*objs))

    raise TypeError("Can't perform concatenation over object type `%s`" % type(obj))


def stack(objs, *args, **kwargs):
    """
    Stack a list of nested containers with the same structure.
    """
    obj = objs[0]
    if isinstance(obj, torch.Tensor):
        return torch.stack(objs, *args, **kwargs)
    elif isinstance(obj, dict):
        return {k: stack([x[k] for x in objs], *args, **kwargs) for k in obj}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(stack(xs, *args, **kwargs) for xs in zip(*objs))

    raise TypeError("Can't perform stack over object type `%s`" % type(obj))


