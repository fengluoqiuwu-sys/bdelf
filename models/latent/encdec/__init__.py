"""共享 T5-small 维数 encoder / block / 瓶颈读出（latent_t5、latent_vae）。"""

from .encoder import LatentEncoder
from .layers import TransformerBlock
from .readout import (
    PosteriorBReadout,
    PosteriorEReadout,
    SIGMA_TAG_SUFFIX,
    ensure_sigma_tag,
    gaussian_log_q,
    kl_gaussian,
    parse_kl_entropy,
    drop_off_kl_entropy,
    posterior_regularizer,
    sample_posterior,
)

__all__ = [
    "LatentEncoder",
    "TransformerBlock",
    "PosteriorBReadout",
    "PosteriorEReadout",
    "SIGMA_TAG_SUFFIX",
    "ensure_sigma_tag",
    "gaussian_log_q",
    "kl_gaussian",
    "parse_kl_entropy",
    "drop_off_kl_entropy",
    "posterior_regularizer",
    "sample_posterior",
]
