"""WikiText dataset.

A concrete dataset built on top of the generic ``FL_Dataset`` template,
backed by the ``Salesforce/wikitext`` repo (subset ``wikitext-103-v1``) on
the HuggingFace Hub. Obtain an instance via ``get_dataset("wikitext")``.
"""

from __future__ import annotations

from .dataset import FL_Dataset


class _WikiText_Dataset(FL_Dataset):
    """WikiText dataset."""
