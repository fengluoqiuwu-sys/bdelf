"""WikiText dataset.

A concrete dataset built on top of the generic ``FL_Dataset`` template,
backed by the ``Salesforce/wikitext`` repo (subset ``wikitext-103-v1``) on
the HuggingFace Hub. Obtain an instance via ``get_dataset("wikitext")``.
"""

from __future__ import annotations

from datasets import concatenate_datasets

from .dataset import FL_Dataset


class _WikiText_Dataset(FL_Dataset):
    """WikiText dataset."""

    def _build_split(self, split: str):
        if split == "train":
            return self._load_raw_split("train")
        if split == "eval":
            validation = self._load_raw_split("validation")
            test = self._load_raw_split("test")
            return concatenate_datasets([validation, test])
        raise ValueError(f"Unknown split '{split}'.")
