from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rnam")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .gru import minGRU
from .linear import Linear

__all__ = ["__version__", "Linear", "minGRU"]
