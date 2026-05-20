__all__ = [
    "GenerativeFactory",
]


def __getattr__(name):
    if name == "GenerativeFactory":
        from .tasks_generate import GenerativeFactory

        return GenerativeFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
