"""Shared ELF-family primitives (T5 encoder, DiT layers, ACE).

Not a trainable model family: no ``CONFIG_CLS`` / ``build_model``.
Sibling packages re-export from here so public import paths stay unchanged.
"""

from . import ace, layers, t5_encoder

__all__ = ["ace", "layers", "t5_encoder"]
