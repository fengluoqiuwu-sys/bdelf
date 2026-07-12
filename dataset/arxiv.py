"""arXiv abstracts dataset.

A concrete dataset built on top of the generic ``FL_Dataset`` template,
backed by the ``gfissore/arxiv-abstracts-2021`` repo on the HuggingFace Hub.
Obtain an instance via ``get_dataset("arxiv")``.
"""

from __future__ import annotations

from .dataset import FL_Dataset


class _Arxiv_Dataset(FL_Dataset):
    """arXiv abstracts dataset."""
