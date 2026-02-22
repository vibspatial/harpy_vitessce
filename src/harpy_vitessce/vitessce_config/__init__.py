"""Vitessce config."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._proteomics import proteomics_from_spatialdata, proteomics_from_split_sources
    from ._seqbased_transcriptomics import (
        seq_based_from_spatialdata,
        seq_based_from_split_sources,
        single_channel_image,
    )

__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submod_attrs={
        "_proteomics": [
            "proteomics_from_spatialdata",
            "proteomics_from_split_sources",
        ],
        "_seqbased_transcriptomics": [
            "seq_based_from_spatialdata",
            "seq_based_from_split_sources",
            "single_channel_image",
        ],
    },
)

__all__ = [
    "proteomics_from_split_sources",
    "proteomics_from_spatialdata",
    "seq_based_from_spatialdata",
    "seq_based_from_split_sources",
    "single_channel_image",
]
