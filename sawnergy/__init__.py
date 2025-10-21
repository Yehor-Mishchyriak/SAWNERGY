from .sawnergy_util import (
   ArrayStorage, elementwise_processor, files_from, batches_of, compose_steps
)

# expose subpackages on demand
def __getattr__(name):
    if name in {
        "sawnergy_util", "logging_util", "rin", "visual", "walks"
    }:
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)

__all__ = [
    # util
    "ArrayStorage", "elementwise_processor", "files_from", "batches_of", "compose_steps",
    # namespaces (lazy) (see above)
    "sawnergy_util", "logging_util", "rin", "visual", "walks"
]
