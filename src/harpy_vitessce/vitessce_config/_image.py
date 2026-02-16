from collections.abc import Sequence

from loguru import logger
from vitessce import (
    CoordinationLevel as CL,
)

from harpy_vitessce.vitessce_config._constants import (
    MAX_INITIAL_CHANNELS,
    SPATIAL_VIEW,
)


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
    # if only one channel, we should fall back to grey.
    return palette[index % len(palette)]


def build_image_layer_config(
    file_uid: str,
    channels: Sequence[int | str] | None,
    visualize_as_rgb: bool = True,
    layer_opacity: float = 1.0,
) -> dict[str, object]:
    """
    Build the Vitessce `imageLayer` coordination entry for the image.

    Parameters
    ----------
    file_uid
        File identifier used to link this layer to the image file.
    channels
        Initial channels rendered in the image layer.
    visualize_as_rgb
        If ``True``, configure image rendering as RGB and ignore ``channels``.
    layer_opacity
        Opacity of the image layer in ``[0, 1]``.

    Returns
    -------
    dict[str, object]
        Layer config for Vitessce `imageLayer`.
    """
    image_layer: dict[str, object] = {
        "fileUid": file_uid,
        "spatialLayerVisible": True,
        "spatialLayerOpacity": layer_opacity,
    }
    # if channels is None -> if visualize_as_rgb -> try rendering as rgb
    if channels is None:
        if visualize_as_rgb:
            # channels ignored in rgb mode.
            selected_channels = None
        else:
            logger.info(
                "No channels were provided, and visualize as rgb set to False; rendering only channel at index 0. "
                "Additional channels can be enabled in the Vitessce UI."
            )
            selected_channels = [0]
    else:
        if visualize_as_rgb:
            logger.warning(
                "Received {} channel selection(s), but visualize_as_rgb=True; "
                "the `channels` parameter is ignored in RGB mode.",
                len(channels),
            )
        selected_channels = channels

    if visualize_as_rgb:
        image_layer["photometricInterpretation"] = "RGB"
        return image_layer

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

    # ignored when photometricInterpretation is RGB (handled by early return).
    image_layer["imageChannel"] = CL(
        [
            {
                "spatialTargetC": channel,
                "spatialChannelColor": _channel_color(pos)
                if len(selected_channels) > 1
                else [255, 255, 255],  # we choose to override colors.
                "spatialChannelVisible": True,
                "spatialChannelWindow": None,
            }
            for pos, channel in enumerate(selected_channels)
        ]
    )
    image_layer["photometricInterpretation"] = "BlackIsZero"
    return image_layer
