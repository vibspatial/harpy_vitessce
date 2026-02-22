from pathlib import Path
from typing import Mapping, Sequence

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from loguru import logger
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image, write_multiscale


def xarray_to_ome_zarr(
    tree_or_da: xr.DataArray | xr.DataTree,
    output_path: str | Path,
    channel_names: Sequence[str] | str,
    channel_colors: Mapping[str, str] | None = None,
    chunks: Sequence[int] = (1, 256, 256),
    coords_in_microns: bool = False,
    microns_per_pixel: float = 1.0,  # ignored if coords_in_microns is True (i.e. if True, we fetch the scale factor from the coords)
    scale_factors: Sequence[int]
    | None = None,  # None -> no pyramid, ignored if tree_or_da is tree
    zarr_format: int = 3,
) -> None:
    """
    Write an OME-Zarr image from either a single-resolution DataArray or a
    multi-resolution DataTree.

    For DataTree input, the function writes all `scale*` levels as a multiscale
    pyramid and derives per-level coordinate transforms from the x/y coords.
    For DataArray input, it writes a single scale when `scale_factors` is None,
    or builds a uniform pyramid using `ome_zarr.scale.Scaler` when
    `scale_factors` is provided.

    Parameters
    ----------
    tree_or_da : xarray.DataTree | xarray.DataArray
        Either a DataTree with `scale0`, `scale1`, ... groups containing an
        `image` DataArray, or a single DataArray with dims (c, y, x) or (y, x).
    output_path : str
        Output path to the OME-Zarr store.
    channel_names : Sequence[str] | str
        Channel labels to store in the OMERO metadata. A single string is
        treated as one channel name.
    channel_colors : dict[str, str] | None
        Mapping channel name -> hex color string. If None, defaults to white.
    chunks : Sequence[int]
        Zarr chunk sizes. Use 3 values for ``(c, y, x)`` data and 2 values for
        ``(y, x)`` data.
    coords_in_microns : bool
        If True, x/y coords are assumed to be micrometers already; otherwise
        they are treated as pixels and scaled by `microns_per_pixel`.
    microns_per_pixel : float
        Physical pixel size in micrometers (used when coords_in_microns is False).
    scale_factors : list[int] | None
        Per-level downscale factors for building a pyramid from a DataArray.
        Must be uniform (e.g., [2, 2, 2]). Ignored for DataTree input.
    zarr_format
        Zarr format to write. Ignored if zarr.__version__ < 3.
    """

    channel_names = _normalize_channel_names(channel_names)

    def _spacing_from_coords(da):
        dy = (
            float(da.coords["y"][1] - da.coords["y"][0])
            if "y" in da.coords and da.sizes["y"] > 1
            else 1.0
        )
        dx = (
            float(da.coords["x"][1] - da.coords["x"][0])
            if "x" in da.coords and da.sizes["x"] > 1
            else 1.0
        )
        return dy, dx

    try:
        # only supported for zarr version > 2
        z_root = zarr.open_group(output_path, mode="w", zarr_format=zarr_format)
    except TypeError:
        z_root = zarr.open_group(output_path, mode="w")

    def _normalize_dataarray_and_axes(
        arr: xr.DataArray,
    ) -> tuple[xr.DataArray, tuple[str, ...], list[dict[str, str]]]:
        dims = tuple(arr.dims)
        if len(dims) == 3 and set(dims) == {"c", "y", "x"}:
            axes = ("c", "y", "x")
            arr = arr.transpose(*axes)
        elif len(dims) == 2 and set(dims) == {"y", "x"}:
            axes = ("y", "x")
            arr = arr.transpose(*axes)
        else:
            raise ValueError(
                "Expected DataArray dims to be either ('c','y','x') or ('y','x'), "
                f"got {dims}."
            )
        axes_meta = []
        for name in axes:
            if name == "c":
                axes_meta.append({"name": "c", "type": "channel"})
            else:
                axes_meta.append({"name": name, "type": "space", "unit": "micrometer"})
        return arr, axes, axes_meta

    def _normalize_chunks(chunks: Sequence[int], n_dims: int) -> tuple[int, ...]:
        chunks_tuple = tuple(chunks)
        if len(chunks_tuple) == n_dims:
            return chunks_tuple
        raise ValueError(
            f"chunks must have {n_dims} entries for this data shape; got {chunks_tuple}."
        )

    da_for_dtype: xr.DataArray | None = None

    if isinstance(tree_or_da, xr.DataTree):
        tree = tree_or_da
        level_keys = sorted(
            [k for k in tree.keys() if k.startswith("scale")],
            key=lambda k: int(k.replace("scale", "")),
        )
        if not level_keys:
            raise ValueError("DataTree must contain at least one scale* node.")

        pyramid = []
        coord_tfm = []
        axes_meta: list[dict[str, str]] | None = None
        axes_names: tuple[str, ...] | None = None

        # If scale_factors provided, ignore them
        if scale_factors is not None:
            logger.warning("scale factors ignored if DataTree")

        for _, key in enumerate(level_keys):
            level_da, level_axes, level_axes_meta = _normalize_dataarray_and_axes(
                tree[key]["image"]
            )
            if axes_names is None:
                axes_names = level_axes
                axes_meta = level_axes_meta
                if "c" in axes_names:
                    n_channels = int(level_da.sizes["c"])
                    if len(channel_names) != n_channels:
                        raise ValueError(
                            "channel_names length does not match channel axis size: "
                            f"{len(channel_names)} != {n_channels}."
                        )
                elif len(channel_names) != 1:
                    raise ValueError(
                        "No channel axis found in data; channel_names must have length 1."
                    )
            elif level_axes != axes_names:
                raise ValueError(
                    "All DataTree levels must use the same dims pattern "
                    f"(got {level_axes} and {axes_names})."
                )
            pyramid.append(level_da.data)
            da_for_dtype = level_da

            dy, dx = _spacing_from_coords(level_da)
            if not coords_in_microns:
                dy *= microns_per_pixel
                dx *= microns_per_pixel
            if axes_names == ("c", "y", "x"):
                scale = [1.0, float(dy), float(dx)]
            else:
                scale = [float(dy), float(dx)]
            coord_tfm.append([{"type": "scale", "scale": scale}])

        write_multiscale(
            pyramid=pyramid,
            group=z_root,
            axes=axes_meta,
            coordinate_transformations=coord_tfm,
            # Use a single dict so ome-zarr copies options per level internally.
            # A repeated-list form shares the same dict object and can cause
            # only level 0 to receive explicit chunks.
            storage_options={"chunks": _normalize_chunks(chunks, len(axes_names))},
        )

    elif isinstance(tree_or_da, xr.DataArray):
        da, axes_names, axes_meta = _normalize_dataarray_and_axes(tree_or_da)
        da_for_dtype = da
        if "c" in axes_names:
            n_channels = int(da.sizes["c"])
            if len(channel_names) != n_channels:
                raise ValueError(
                    "channel_names length does not match channel axis size: "
                    f"{len(channel_names)} != {n_channels}."
                )
        elif len(channel_names) != 1:
            raise ValueError(
                "No channel axis found in data; channel_names must have length 1."
            )

        dy, dx = _spacing_from_coords(da)
        if not coords_in_microns:
            dy *= microns_per_pixel
            dx *= microns_per_pixel

        if axes_names == ("c", "y", "x"):
            scale_base = [1.0, float(dy), float(dx)]
        else:
            scale_base = [float(dy), float(dx)]

        if scale_factors is None:
            coord_tfm = [[{"type": "scale", "scale": scale_base}]]
            scaler = None
        else:
            if len(set(scale_factors)) != 1:
                raise ValueError(
                    "Non-uniform scale_factors require precomputed multiscale (DataTree). "
                    "ome_zarr.Scaler only supports a constant downscale."
                )
            downscale = scale_factors[0]
            max_layer = len(scale_factors)
            scaler = Scaler(max_layer=max_layer, downscale=downscale)

            cum = [1.0] + _cum_factors(scale_factors)
            if axes_names == ("c", "y", "x"):
                coord_tfm = [
                    [{"type": "scale", "scale": [1.0, float(dy * c), float(dx * c)]}]
                    for c in cum
                ]
            else:
                coord_tfm = [
                    [{"type": "scale", "scale": [float(dy * c), float(dx * c)]}]
                    for c in cum
                ]

        write_image(
            image=da.data,
            group=z_root,
            axes=axes_meta,
            coordinate_transformations=coord_tfm,
            storage_options={"chunks": _normalize_chunks(chunks, len(axes_names))},
            scaler=scaler,
        )

    else:
        raise ValueError(
            "tree_or_da must be an xarray.DataTree or xarray.DataArray; "
            f"got {type(tree_or_da).__name__}."
        )

    if channel_colors is None:
        channel_colors = {name: "FFFFFF" for name in channel_names}

    if da_for_dtype is None:
        raise RuntimeError("No image data was written; cannot infer dtype metadata.")

    dtype_info = (
        np.iinfo(da_for_dtype.dtype)
        if da_for_dtype.dtype.kind in ("u", "i")
        else np.finfo(da_for_dtype.dtype)
    )

    default_window = {
        "start": 0,
        "min": 0,
        "max": dtype_info.max,
        "end": dtype_info.max,
    }

    z_root.attrs["omero"] = {
        "name": "Image",
        "version": "0.3",
        "rdefs": _default_omero_rdefs_model(len(channel_names)),
        "channels": [
            {
                "label": ch,
                "color": channel_colors.get(ch, "FFFFFF"),
                "window": default_window,
            }
            for ch in channel_names
        ],
    }


def array_to_ome_zarr(
    img_arr: np.ndarray | da.Array,
    output_path: str | Path,
    channel_names: Sequence[str] | str,
    channel_colors: Mapping[str, str] | None = None,
    chunks: tuple[int, int, int] = (1, 256, 256),
    microns_per_pixel: float = 1.0,
    microns_per_z: float | None = None,
    scale_factors: Sequence[int] | None = None,  # None -> no pyramid
    axes: Sequence[str] = ("c", "y", "x"),
    zarr_format: int = 2,
) -> None:
    """
    Write an OME-Zarr image from a numpy/dask array with explicit axes.

    Parameters
    ----------
    img_arr : numpy.ndarray | dask.array.Array
        Image array matching the provided axes.
    output_path : str | Path
        Output path to the OME-Zarr store.
    channel_names : Sequence[str] | str
        Channel labels to store in the OMERO metadata. A single string is
        treated as one channel name.
    channel_colors : dict[str, str] | None
        Mapping channel name -> hex color string. If None, defaults to white.
    chunks : tuple[int, int, int]
        Zarr chunk sizes in (c, y, x) order.
    microns_per_pixel : float
        Physical pixel size in micrometers for x/y.
    microns_per_z : float | None
        Physical pixel size in micrometers for z. Defaults to microns_per_pixel.
    scale_factors : list[int] | None
        Per-level downscale factors for building a pyramid. Must be uniform
        (e.g., [2, 2, 2]) if provided.
    axes : Sequence[str] | str
        Axis order as a sequence (e.g., ["c", "y", "x"]) or a string
        (e.g., "cyx" or "czyx").
    zarr_format : int
        Zarr format version to write. Supported values are ``2`` and ``3``.
    """
    channel_names = _normalize_channel_names(channel_names)

    if microns_per_z is None:
        microns_per_z = microns_per_pixel

    if isinstance(axes, str):
        axes = list(axes)

    if zarr_format not in {2, 3}:
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}.")

    try:
        z_root = zarr.open_group(output_path, mode="w", zarr_format=zarr_format)
    except TypeError as e:
        if zarr_format != 2:
            raise ValueError(
                "Requested zarr_format=3, but installed zarr does not support "
                "the zarr_format argument. Upgrade zarr to v3+ or use zarr_format=2."
            ) from e
        z_root = zarr.open_group(output_path, mode="w")

    axes_meta = []
    for name in axes:
        if name == "c":
            axes_meta.append({"name": "c", "type": "channel"})
        elif name in {"x", "y", "z"}:
            axes_meta.append({"name": name, "type": "space", "unit": "micrometer"})
        elif name == "t":
            axes_meta.append({"name": "t", "type": "time"})
        else:
            axes_meta.append({"name": name})

    base_scale = []
    for name in axes:
        if name in {"c", "t"}:
            base_scale.append(1.0)
        elif name == "z":
            base_scale.append(float(microns_per_z))
        elif name in {"x", "y"}:
            base_scale.append(float(microns_per_pixel))
        else:
            base_scale.append(1.0)

    if "c" in axes:
        channel_axis = axes.index("c")
        n_channels = img_arr.shape[channel_axis]
        if len(channel_names) != n_channels:
            raise ValueError(
                "channel_names length does not match channel axis size: "
                f"{len(channel_names)} != {n_channels}."
            )
    else:
        if len(channel_names) != 1:
            raise ValueError(
                "No channel axis found in axes; channel_names must have length 1."
            )

    if scale_factors is None:
        coord_tfm = [[{"type": "scale", "scale": base_scale}]]
        scaler = None
    else:
        if len(set(scale_factors)) != 1:
            raise ValueError(
                "Non-uniform scale_factors require precomputed multiscale. "
                "ome_zarr.Scaler only supports a constant downscale."
            )
        downscale = scale_factors[0]
        max_layer = len(scale_factors)
        scaler = Scaler(max_layer=max_layer, downscale=downscale)
        cum = [1.0] + _cum_factors(scale_factors)
        coord_tfm = []
        for c in cum:
            scale = []
            for name, base in zip(axes, base_scale):
                if name in {"x", "y"}:
                    scale.append(float(base * c))
                else:
                    scale.append(float(base))
            coord_tfm.append([{"type": "scale", "scale": scale}])

    write_image(
        image=img_arr,
        group=z_root,
        axes=axes_meta,
        coordinate_transformations=coord_tfm,
        storage_options={"chunks": chunks},
        scaler=scaler,
    )

    if channel_colors is None:
        channel_colors = {name: "FFFFFF" for name in channel_names}

    dtype = np.dtype(img_arr.dtype)
    dtype_info = np.iinfo(dtype) if dtype.kind in ("u", "i") else np.finfo(dtype)

    default_window = {
        "start": 0,
        "min": 0,
        "max": dtype_info.max,
        "end": dtype_info.max,
    }

    z_root.attrs["omero"] = {
        "name": "Image",
        "version": "0.3",
        "rdefs": _default_omero_rdefs_model(len(channel_names)),
        "channels": [
            {
                "label": ch,
                "color": channel_colors.get(ch, "FFFFFF"),
                "window": default_window,
            }
            for ch in channel_names
        ],
    }


def _cum_factors(scale_factors: Sequence[int]) -> list[float]:
    out: list[float] = []
    c = 1.0
    for f in scale_factors:
        c *= f
        out.append(c)
    return out


def _normalize_channel_names(channel_names: Sequence[str] | str) -> list[str]:
    if isinstance(channel_names, str):
        return [channel_names]
    return list(channel_names)


def _default_omero_rdefs_model(n_channels: int) -> dict[str, str]:
    # Vitessce treats model='color' as RGB; for non-RGB channel counts we
    # keep an empty rdefs mapping so data is handled as multiplex/grayscale.
    if n_channels in {3, 4}:
        return {"model": "color"}
    return {}
