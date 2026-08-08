"""离线生成式评测包（TriFluency + Gen.PPL / unigram entropy）。

入口：``.venv/bin/python -m eval ...``（包名 ``eval``，勿与内建 ``eval()`` 混淆）。
Gen.PPL / entropy 权威实现：``eval.gen_ppl``；训练环胶水仍在 ``train.eval``。
"""

from __future__ import annotations

__all__ = ["gen_ppl"]
