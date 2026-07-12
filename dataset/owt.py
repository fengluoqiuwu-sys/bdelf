"""OpenWebText (OWT) dataset.

A concrete dataset built on top of the generic ``FL_Dataset`` template,
backed by the ``Skylion007/openwebtext`` repo on the HuggingFace Hub.
Obtain an instance via ``get_dataset("owt")``.
"""

from __future__ import annotations

from .dataset import FL_Dataset


class _OWT_Dataset(FL_Dataset):
    """OpenWebText dataset."""
