"""harpy_vitessce package."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from . import data_utils, vitessce_config
    from .vitessce_config import (
        proteomics_from_spatialdata,
        proteomics_from_split_sources,
        seq_based_from_spatialdata,
        seq_based_from_split_sources,
        single_channel_image,
    )

    proteomics = proteomics_from_spatialdata

_lazy_getattr, _lazy_dir, _ = lazy.attach(
    __name__,
    submodules=["data_utils", "vitessce_config"],
    submod_attrs={
        "vitessce_config": [
            "proteomics_from_spatialdata",
            "proteomics_from_split_sources",
            "seq_based_from_spatialdata",
            "seq_based_from_split_sources",
            "single_channel_image",
        ],
    },
)


def __getattr__(name: str):
    if name == "proteomics":
        return _lazy_getattr("proteomics_from_spatialdata")
    return _lazy_getattr(name)


def __dir__():
    return sorted(set(list(_lazy_dir()) + ["proteomics"]))

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
