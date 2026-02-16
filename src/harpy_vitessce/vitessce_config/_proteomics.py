import uuid
from collections.abc import Sequence
from pathlib import Path

from loguru import logger
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


def _channel_color(index: int) -> list[int]:
    palette = [
        [255, 0, 0],  # "#FF0000"  Red
        [0, 255, 0],  # "#00FF00"  Green
        [0, 0, 255],  # "#0000FF"  Blue
        [255, 255, 0],  # "#FFFF00"  Yellow
        [128, 0, 128],  # "#800080"  Purple
        [255, 165, 0],  # "#FFA500"  Orange
        [255, 192, 203],  # "#FFC0CB"  Pink
        [139, 69, 19],  # "#8B4513"  Brown
    ]
    return palette[index % len(palette)]


def _build_macsima_image_layer(
    file_uid: str,
    channels: Sequence[int | str] | None,
    layer_opacity: float,
    channel_opacity: float,
) -> tuple[dict[str, object], bool]:
    # maybe let user specify how much, else default to 1?
    # maybe add an assert
    # allow user to pass a palette.
    if channels is None:
        selected_channels = [0]  # only render the first if none specified
        logger.info(
            "No channels were provided; rendering only channel at index 0. "
            "Additional channels can be enabled in the Vitessce UI."
        )
    else:
        selected_channels = channels
    if len(selected_channels) > MAX_INITIAL_CHANNELS:
        logger.warning(
            "Vitessce {} supports at most {} simultaneously visible channels; "
            "got {}. Will only render the first {} channels.",
            SPATIAL_VIEW,
            MAX_INITIAL_CHANNELS,
            len(selected_channels),
            MAX_INITIAL_CHANNELS,
        )
        selected_channels = selected_channels[:MAX_INITIAL_CHANNELS]
    image_layer: dict[str, object] = {
        "fileUid": file_uid,
        "spatialLayerVisible": True,
        "spatialLayerOpacity": layer_opacity,
        "photometricInterpretation": "BlackIsZero",
        "imageChannel": CL(
            [
                {
                    "spatialTargetC": channel,
                    "spatialChannelColor": _channel_color(pos),
                    # TODO: yes -> override colors?
                    "spatialChannelVisible": True,
                    "spatialChannelOpacity": channel_opacity,  # TODO: this seems to be ignored.
                    "spatialChannelWindow": None,
                }
                for pos, channel in enumerate(selected_channels)
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
    channel_opacity: float = 1.0,  # channel opacity seems to be ignored.
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
        Initial channels rendered by spatialBeta.
        Entries can be integer channel indices or channel names.
        If more than 6 channels are provided, only the first 6 are used.
        If ``None``, only channel at index 0 is shown.
        Channel colors are assigned from an internal palette in the order
        of this list (position-based, not value-based).

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

    image_layer, can_render_as_rgb = _build_macsima_image_layer(
        file_uid=file_uuid,
        channels=channels,
        layer_opacity=layer_opacity,
        channel_opacity=channel_opacity,  # TODO this seems to be ignored.
    )

    vc.link_views_by_dict(
        [spatial_plot, layer_controller],
        {"imageLayer": CL([image_layer])},
    )
    layer_controller.set_props(disableChannelsIfRgbDetected=can_render_as_rgb)
    vc.layout(hconcat(spatial_plot, layer_controller, split=[8, 4]))

    return vc
