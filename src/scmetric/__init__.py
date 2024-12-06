from importlib.metadata import version

from . import external, pl, pp, tl

__all__ = ["pl", "pp", "tl", "external"]

__version__ = version("scmetric")
