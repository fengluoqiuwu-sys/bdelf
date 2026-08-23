"""共享 T5-small 维数 encoder / block / 瓶颈读出（latent_t5、latent_vae）。"""

from .encoder import LatentEncoder
from .layers import TransformerBlock
from .readout import (
    PosteriorBReadout,
    PosteriorEReadout,
    kl_gaussian,
    sample_posterior,
)

__all__ = [
    "LatentEncoder",
    "TransformerBlock",
    "PosteriorBReadout",
    "PosteriorEReadout",
    "kl_gaussian",
    "sample_posterior",
]
