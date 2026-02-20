"""harpy_vitessce package."""

from . import data_utils, vitessce_config
from .vitessce_config import (
    proteomics_from_spatialdata,
    proteomics_from_split_sources,
    seq_based_from_spatialdata,
    seq_based_from_split_sources,
    single_channel_image,
)

# Convenience alias on the top-level package namespace.
proteomics = proteomics_from_spatialdata

__all__ = [
    "data_utils",
    "vitessce_config",
    "proteomics",
    "proteomics_from_split_sources",
    "proteomics_from_spatialdata",
    "seq_based_from_spatialdata",
    "seq_based_from_split_sources",
    "single_channel_image",
]
