"""Held-out 评测 split 解析（``dev`` / ``eval`` 向前兼容）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from preprocess.preprocess import FL_PreprocessedDataset


def resolve_eval_split(
    preprocessed: FL_PreprocessedDataset,
    requested: str | None,
) -> str:
    """解析在线评测用的 held-out split。

    - ``requested`` 为 ``dev`` / ``eval`` 时须存在于缓存；
    - 省略时优先 ``dev``（owt 三向 holdout），否则 ``eval``（arxiv 等二向 holdout）。
    """
    available = set(preprocessed.get_splits())
    if requested is not None:
        if requested not in ("dev", "eval"):
            raise ValueError(
                f"eval_split must be 'dev' or 'eval', got {requested!r}"
            )
        if requested not in available:
            raise ValueError(
                f"eval_split={requested!r} not in preprocessed splits {sorted(available)}"
            )
        return requested
    if "dev" in available:
        return "dev"
    if "eval" in available:
        return "eval"
    raise ValueError(
        f"No held-out split (dev or eval) in preprocessed splits: {sorted(available)}"
    )


def require_train_and_holdout(splits: list[str]) -> None:
    """要求存在 train 与 dev/eval 之一。"""
    if "train" not in splits:
        raise ValueError(f"Dataset is missing train split; current splits: {splits}")
    if "dev" not in splits and "eval" not in splits:
        raise ValueError(
            f"Dataset is missing held-out split (dev or eval); "
            f"current splits: {splits}"
        )
