"""Model registry and factory."""

from .hf_model import (
    FL_HFModel,
    FL_HFModelConfig,
    download_hf_model,
    get_hf_model,
    is_hf_model_cached,
    resolve_hf_model_cache_path,
)
from .kinds import ModelKind, import_family_module, kind_of, list_kinds
from .latent.artifact_loader import (
    LatentArtifactLoad,
    artifacts_latent_root,
    list_artifact_tags,
    load_latent_artifact,
    resolve_artifact_checkpoint,
    save_latent_artifact,
)
from .model import (
    FL_PreTrainedModel,
    build_model,
    config_from_yaml,
    get_model,
    list_model_configs,
    list_models,
    load_model_yaml,
    resolve_full_sequence_training,
    resolve_model_config_path,
)

__all__ = [
    "FL_HFModel",
    "FL_HFModelConfig",
    "FL_PreTrainedModel",
    "LatentArtifactLoad",
    "ModelKind",
    "artifacts_latent_root",
    "build_model",
    "config_from_yaml",
    "download_hf_model",
    "get_hf_model",
    "get_model",
    "import_family_module",
    "is_hf_model_cached",
    "kind_of",
    "list_artifact_tags",
    "list_kinds",
    "load_latent_artifact",
    "resolve_artifact_checkpoint",
    "save_latent_artifact",
    "list_model_configs",
    "list_models",
    "load_model_yaml",
    "resolve_full_sequence_training",
    "resolve_hf_model_cache_path",
    "resolve_model_config_path",
]
