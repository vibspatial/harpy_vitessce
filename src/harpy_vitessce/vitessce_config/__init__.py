"""Vitessce config."""

from ._proteomics import macsima
from ._seqbased_transcriptomics import single_channel_image, visium_hd

__all__ = [
    "macsima",
    "visium_hd",
    "single_channel_image",
]
