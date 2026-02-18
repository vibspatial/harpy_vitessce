"""Vitessce config."""

from ._proteomics import proteomics
from ._seqbased_transcriptomics import single_channel_image, visium_hd

__all__ = [
    "proteomics",
    "visium_hd",
    "single_channel_image",
]
