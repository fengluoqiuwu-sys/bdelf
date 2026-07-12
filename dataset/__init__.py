from .arxiv import _Arxiv_Dataset
from .dataset import (
    FL_Dataset,
    FL_DatasetConfig,
    get_dataset,
    list_datasets,
    register_dataset,
)
from .owt import _OWT_Dataset
from .wikitext import _WikiText_Dataset

# Registry: bind each dataset name to its concrete class.
register_dataset("owt", _OWT_Dataset)
register_dataset("wikitext", _WikiText_Dataset)
register_dataset("arxiv", _Arxiv_Dataset)

__all__ = [
    "FL_Dataset",
    "FL_DatasetConfig",
    "get_dataset",
    "list_datasets",
    "register_dataset",
]
