"""Latent representation models trained in-repo (e.g. Cola VAE)."""

from .artifact_export import default_artifact_tag, export_latent_artifact
from .artifact_loader import (
    LatentArtifactLoad,
    artifacts_latent_root,
    assert_not_artifacts_latent_path,
    list_artifact_tags,
    load_latent_artifact,
    resolve_artifact_checkpoint,
    resolve_artifact_dir,
    save_latent_artifact,
)

__all__ = [
    "LatentArtifactLoad",
    "artifacts_latent_root",
    "assert_not_artifacts_latent_path",
    "default_artifact_tag",
    "export_latent_artifact",
    "list_artifact_tags",
    "load_latent_artifact",
    "resolve_artifact_checkpoint",
    "resolve_artifact_dir",
    "save_latent_artifact",
]
