"""Shared data utilities."""

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from ._adata import copy_annotations, downcast_int64_to_int32, normalize_array
    from ._ome import array_to_ome_zarr, xarray_to_ome_zarr
    from ._visium_hd import example_visium_hd_processing

__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submod_attrs={
        "_ome": ["xarray_to_ome_zarr", "array_to_ome_zarr"],
        "_adata": [
            "copy_annotations",
            "downcast_int64_to_int32",
            "normalize_array",
        ],
        "_visium_hd": ["example_visium_hd_processing"],
    },
)

__all__ = [
    "xarray_to_ome_zarr",
    "array_to_ome_zarr",
    "copy_annotations",
    "example_visium_hd_processing",
    "downcast_int64_to_int32",
    "normalize_array",
]
