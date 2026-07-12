"""HuggingFace pretrained model cache and loader for eval baselines."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import PreTrainedModel

CACHE_ROOT = Path("cache/models")


def _cache_name(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def resolve_hf_model_cache_path(repo_id: str, cache_path: str | os.PathLike | None = None) -> Path:
    """Return the local directory for ``repo_id`` under ``cache/models``."""
    if cache_path:
        return Path(cache_path)
    return CACHE_ROOT / _cache_name(repo_id)


def is_hf_model_cached(repo_id: str, cache_path: str | os.PathLike | None = None) -> bool:
    """Check whether a model snapshot exists locally."""
    path = resolve_hf_model_cache_path(repo_id, cache_path)
    if not path.is_dir():
        return False
    return (path / "config.json").exists()


def download_hf_model(
    repo_id: str,
    *,
    cache_path: str | os.PathLike | None = None,
    revision: str | None = None,
) -> Path:
    """Download ``repo_id`` into ``cache/models`` if not already present."""
    import hf_config  # noqa: F401
    from huggingface_hub import snapshot_download

    local_dir = resolve_hf_model_cache_path(repo_id, cache_path)
    if is_hf_model_cached(repo_id, local_dir):
        print("Already downloaded")
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
    )
    return local_dir


@dataclass
class FL_HFModelConfig:
    """Config for a cached HuggingFace eval model."""

    repo_id: str
    cache_path: Optional[str] = None
    revision: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cache_path:
            self.cache_path = str(resolve_hf_model_cache_path(self.repo_id))


class FL_HFModel:
    """Thin wrapper around a cached HuggingFace ``PreTrainedModel``."""

    def __init__(self, config: FL_HFModelConfig, model: PreTrainedModel) -> None:
        self.config = config
        self.model = model

    def is_downloaded(self) -> bool:
        return is_hf_model_cached(self.config.repo_id, self.config.cache_path)

    def download(self) -> Path:
        return download_hf_model(
            self.config.repo_id,
            cache_path=self.config.cache_path,
            revision=self.config.revision,
        )

    @classmethod
    def from_repo_id(
        cls,
        repo_id: str,
        *,
        cache_path: str | os.PathLike | None = None,
        revision: str | None = None,
        torch_dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
        **load_kwargs: Any,
    ) -> "FL_HFModel":
        config = FL_HFModelConfig(
            repo_id=repo_id,
            cache_path=str(cache_path) if cache_path is not None else None,
            revision=revision,
        )
        local_dir = download_hf_model(
            repo_id,
            cache_path=config.cache_path,
            revision=revision,
        )
        model = load_hf_model_from_cache(
            local_dir,
            torch_dtype=torch_dtype,
            device=device,
            **load_kwargs,
        )
        return cls(config, model)


def load_hf_model_from_cache(
    cache_path: str | os.PathLike,
    *,
    torch_dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    **load_kwargs: Any,
) -> PreTrainedModel:
    """Load a causal LM from a local snapshot directory."""
    import hf_config  # noqa: F401
    from transformers import AutoModelForCausalLM

    kwargs: Dict[str, Any] = dict(load_kwargs)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(str(cache_path), **kwargs)
    if device is not None:
        model = model.to(device)
    return model


def get_hf_model(
    repo_id: str,
    *,
    cache_path: str | os.PathLike | None = None,
    revision: str | None = None,
    torch_dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    **load_kwargs: Any,
) -> PreTrainedModel:
    """Download (if needed) and load a HuggingFace causal LM for eval."""
    wrapper = FL_HFModel.from_repo_id(
        repo_id,
        cache_path=cache_path,
        revision=revision,
        torch_dtype=torch_dtype,
        device=device,
        **load_kwargs,
    )
    return wrapper.model
