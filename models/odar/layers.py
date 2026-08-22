"""ODAR layer primitives: shared ELF layers with ``ODARBlock`` alias."""

from models.elf_core.layers import *  # noqa: F403
from models.elf_core.layers import ELFBlock

# Historical name; body identical to ELFBlock (state_dict keys use self.blocks.i.*).
ODARBlock = ELFBlock
