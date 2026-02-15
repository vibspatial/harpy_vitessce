import uuid
from collections.abc import Sequence
from pathlib import Path

import zarr
from vitessce import (
    CoordinationLevel as CL,
)
from vitessce import (
    CoordinationType as ct,
)
from vitessce import ImageOmeZarrWrapper, VitessceConfig, hconcat

from harpy_vitessce.vitessce_config._utils import _normalize_path_or_url

# Vitessce component identifiers used by this config.
SPATIAL_VIEW = "spatialBeta"
LAYER_CONTROLLER_VIEW = "layerControllerBeta"
MAX_INITIAL_CHANNELS = 6  # Viv only supports 6 simultanously.


class _MacsimaImageLayerConfigBuilder:
    def __init__(
        self,
        img_source: str,
        is_img_remote: bool,
        base_dir: str | Path | None,
        layer_opacity: float,
        channel_opacity: float,
    ):
        self.img_source = img_source
        self.is_img_remote = is_img_remote
        self.base_dir = base_dir
        self.layer_opacity = layer_opacity
        self.channel_opacity = channel_opacity

    def _resolve_local_img_path(self) -> Path | None:
        if self.is_img_remote:
            return None
        path = Path(self.img_source)
        if path.is_absolute():
            return path
        if self.base_dir is not None:
            return Path(self.base_dir) / path
        return path

    def _read_channel_metadata(self) -> tuple[int | None, list[str] | None]:
        """
        Return channel count and optional channel labels from local OME-Zarr metadata.
        """
        img_path = self._resolve_local_img_path()
        if img_path is None or not img_path.exists():
            return None, None

        try:
            z_root = zarr.open_group(str(img_path), mode="r")
        except Exception:
            return None, None

        channel_names: list[str] | None = None
        n_channels: int | None = None

        omero = z_root.attrs.get("omero")
        if isinstance(omero, dict):
            channels = omero.get("channels")
            if isinstance(channels, list) and len(channels) > 0:
                n_channels = len(channels)
                names: list[str] = []
                for i, ch in enumerate(channels):
                    if isinstance(ch, dict) and isinstance(ch.get("label"), str):
                        names.append(ch["label"])
                    else:
                        names.append(f"channel_{i}")
                channel_names = names

        if n_channels is None:
            multiscales = z_root.attrs.get("multiscales")
            if isinstance(multiscales, list) and len(multiscales) > 0:
                first = multiscales[0]
                axes = first.get("axes")
                datasets = first.get("datasets")
                if (
                    isinstance(axes, list)
                    and isinstance(datasets, list)
                    and len(datasets) > 0
                ):
                    axis_names: list[str | None] = []
                    for axis in axes:
                        if isinstance(axis, dict):
                            axis_names.append(axis.get("name"))
                        elif isinstance(axis, str):
                            axis_names.append(axis)
                        else:
                            axis_names.append(None)
                    if "c" in axis_names:
                        c_axis = axis_names.index("c")
                        first_dataset = datasets[0]
                        if isinstance(first_dataset, dict):
                            dataset_path = first_dataset.get("path")
                            if isinstance(dataset_path, str):
                                try:
                                    arr = z_root[dataset_path]
                                    n_channels = int(arr.shape[c_axis])
                                except Exception:
                                    pass

        return n_channels, channel_names

    @staticmethod
    def _channel_color(index: int) -> list[int]:
        palette = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 165, 0],
            [165, 255, 0],
        ]
        return palette[index % len(palette)]

    @staticmethod
    def _dedupe_preserve_order(values: Sequence[int]) -> list[int]:
        seen: set[int] = set()
        out: list[int] = []
        for v in values:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def _resolve_requested_channels(
        self, channels: Sequence[int | str] | None
    ) -> list[int]:
        n_channels, channel_names = self._read_channel_metadata()

        if channels is None:
            if n_channels is None:
                return [0]
            # so currently we fall back to showing channel 0.
            # TODO: should we not fall back to showing the first 5-6 channels.
            return list(range(min(MAX_INITIAL_CHANNELS, n_channels)))

        if len(channels) == 0:
            raise ValueError("channels must not be empty.")

        name_to_index: dict[str, int] = {}
        if channel_names is not None:
            for i, name in enumerate(channel_names):
                if name not in name_to_index:
                    name_to_index[name] = i

        resolved: list[int] = []
        for channel in channels:
            if isinstance(channel, bool):
                raise TypeError("channels entries must be int or str, not bool.")
            if isinstance(channel, int):
                idx = channel
            elif isinstance(channel, str):
                if len(name_to_index) == 0:
                    raise ValueError(
                        "channels contains names, but channel labels are unavailable "
                        "for this image source."
                    )
                if channel not in name_to_index:
                    raise ValueError(
                        f"Unknown channel name '{channel}'. Available names: "
                        f"{list(name_to_index.keys())}"
                    )
                idx = name_to_index[channel]
            else:
                raise TypeError(
                    "channels entries must be int (channel index) or str (channel name)."
                )

            if idx < 0:
                raise ValueError(f"Channel index must be >= 0, got {idx}.")
            resolved.append(idx)

        resolved = self._dedupe_preserve_order(resolved)
        if len(resolved) > MAX_INITIAL_CHANNELS:
            raise ValueError(
                f"Vitessce spatialBeta can render at most {MAX_INITIAL_CHANNELS} "
                f"channels at once. Got {len(resolved)} initial channels."
            )

        if n_channels is not None:
            out_of_bounds = [idx for idx in resolved if idx >= n_channels]
            if len(out_of_bounds) > 0:
                raise ValueError(
                    f"Channel indices out of bounds for image with {n_channels} channels: "
                    f"{out_of_bounds}"
                )

        return resolved

    def build(
        self,
        file_uid: str,
        channels: Sequence[int | str] | None,
    ) -> tuple[dict[str, object], bool]:
        selected_channels = self._resolve_requested_channels(channels)
        image_layer: dict[str, object] = {
            "fileUid": file_uid,
            "spatialLayerVisible": True,
            "spatialLayerOpacity": self.layer_opacity,
            "photometricInterpretation": "BlackIsZero",
            "imageChannel": CL(
                [
                    {
                        "spatialTargetC": i,
                        "spatialChannelColor": self._channel_color(i),
                        "spatialChannelVisible": True,
                        "spatialChannelOpacity": self.channel_opacity,
                        "spatialChannelWindow": None,
                    }
                    for i in selected_channels
                ]
            ),
        }
        return image_layer, False


def macsima(
    img_source: str | Path,  # local path relative to base_dir or remote URL
    name: str = "MACSima",
    description: str = "MACSima image-only view",
    schema_version: str = "1.0.18",
    base_dir: str | Path | None = None,
    center: tuple[float, float] | None = None,
    zoom: float | None = -4,
    layer_opacity: float = 1.0,
    channel_opacity: float = 1.0,
    channels: Sequence[int | str] | None = None,
) -> VitessceConfig:
    """
    Build a Vitessce configuration for MACSima image-only visualization.

    Parameters
    ----------
    img_source
        Path/URL to the OME-Zarr image. Local paths are relative to ``base_dir``
        when provided.
    name
        Dataset name shown in Vitessce.
    description
        Configuration description.
    schema_version
        Vitessce schema version.
    base_dir
        Optional base directory for relative local paths in the config.
        Ignored when ``img_source`` is a remote URL.
    center
        Initial spatial target as ``(x, y)`` camera center coordinates.
        Use ``None`` to keep Vitessce defaults.
    zoom
        Initial spatial zoom level. Use ``None`` to keep Vitessce defaults.
    layer_opacity
        Opacity of the image layer in ``[0, 1]``.
    channel_opacity
        Default opacity applied to each image channel in ``[0, 1]``.
    channels
        Initial channels rendered by spatialBeta (max 6 at once).
        Entries can be integer channel indices or channel names.
        If ``None``, the first 6 channels are used (or `[0]` if channel count
        cannot be inferred).

    Returns
    -------
    VitessceConfig
        A configured Vitessce configuration object with image-only views.

    Raises
    ------
    ValueError
        If ``center`` is provided but is not a 2-item tuple.
    """
    img_source, is_img_remote = _normalize_path_or_url(img_source, "img_source")
    if center is not None and len(center) != 2:
        raise ValueError("center must be a tuple of two floats: (x, y).")
    if not 0.0 <= layer_opacity <= 1.0:
        raise ValueError("layer_opacity must be between 0.0 and 1.0.")
    if not 0.0 <= channel_opacity <= 1.0:
        raise ValueError("channel_opacity must be between 0.0 and 1.0.")

    vc = VitessceConfig(
        schema_version=schema_version,
        description=description,
        # base_dir only applies to local *_path entries.
        base_dir=(
            None if is_img_remote else (str(base_dir) if base_dir is not None else None)
        ),
    )

    spatial_coordination_scopes = []
    if zoom is not None:
        (spatial_zoom,) = vc.add_coordination(ct.SPATIAL_ZOOM)
        spatial_zoom.set_value(zoom)
        spatial_coordination_scopes.append(spatial_zoom)
    if center is not None:
        spatial_target_x, spatial_target_y = vc.add_coordination(
            ct.SPATIAL_TARGET_X, ct.SPATIAL_TARGET_Y
        )
        spatial_target_x.set_value(center[0])
        spatial_target_y.set_value(center[1])
        spatial_coordination_scopes.extend([spatial_target_x, spatial_target_y])

    file_uuid = f"img_macsima_{uuid.uuid4()}"  # can be set to any value
    img_wrapper_kwargs: dict[str, object] = {
        "coordination_values": {"fileUid": file_uuid},
    }
    img_wrapper_kwargs["img_url" if is_img_remote else "img_path"] = img_source
    dataset = vc.add_dataset(name=name).add_object(
        ImageOmeZarrWrapper(**img_wrapper_kwargs)
    )

    spatial_plot = vc.add_view(SPATIAL_VIEW, dataset=dataset)
    layer_controller = vc.add_view(LAYER_CONTROLLER_VIEW, dataset=dataset)

    if spatial_coordination_scopes:
        spatial_plot.use_coordination(*spatial_coordination_scopes)

    image_layer, can_render_as_rgb = _MacsimaImageLayerConfigBuilder(
        img_source=img_source,
        is_img_remote=is_img_remote,
        base_dir=base_dir,
        layer_opacity=layer_opacity,
        channel_opacity=channel_opacity,
    ).build(file_uid=file_uuid, channels=channels)

    vc.link_views_by_dict(
        [spatial_plot, layer_controller],
        {"imageLayer": CL([image_layer])},
    )
    layer_controller.set_props(disableChannelsIfRgbDetected=can_render_as_rgb)
    vc.layout(hconcat(spatial_plot, layer_controller, split=[8, 4]))

    return vc
