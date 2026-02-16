from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np
from loguru import logger
from vitessce import (
    CoordinationLevel as CL,
)

from harpy_vitessce.vitessce_config._constants import (
    MAX_INITIAL_CHANNELS,
    SPATIAL_VIEW,
)


# neon
def _default_palette() -> list[list[int]]:
    return [
        [0, 255, 255],  # #00FFFF  Cyan
        [255, 0, 255],  # #FF00FF  Magenta
        [255, 165, 0],  # #FFA500  Orange
        [173, 255, 47],  # #ADFF2F  Green-yellow
        [255, 80, 80],  # #FF5050  Light red
        [135, 206, 255],  # #87CEFF  Light blue
    ]


def _hex_to_rgb(color: str) -> list[int]:
    if not isinstance(color, str) or len(color) != 7 or not color.startswith("#"):
        raise ValueError(f"Invalid hex color '{color}'. Expected format '#RRGGBB'.")
    try:
        return [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
    except ValueError as e:
        raise ValueError(
            f"Invalid hex color '{color}'. Expected format '#RRGGBB'."
        ) from e


def _resolve_palette(palette: Sequence[str] | None) -> list[list[int]]:
    if palette is None or len(palette) == 0:
        return _default_palette()
    return [_hex_to_rgb(color) for color in palette]


def _rgb_to_hex(color: Sequence[int]) -> str:
    return f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"


def _channel_color(index: int, palette_rgb: Sequence[list[int]]) -> list[int]:
    return palette_rgb[index % len(palette_rgb)]


def build_image_layer_config(
    file_uid: str,
    channels: Sequence[int | str] | None,
    visualize_as_rgb: bool = True,
    layer_opacity: float = 1.0,
    palette: Sequence[str] | None = None,
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
    palette
        Optional list of channel colors in hex format (``"#RRGGBB"``).
        Colors are assigned by channel position and repeated cyclically when
        fewer colors than channels are provided.

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
        if palette is not None and len(palette) > 0:
            logger.warning(
                "Received {} palette color(s), but visualize_as_rgb=True; "
                "the `palette` parameter is ignored in RGB mode.",
                len(palette),
            )
        image_layer["photometricInterpretation"] = "RGB"
        return image_layer

    if len(selected_channels) > MAX_INITIAL_CHANNELS:
        logger.warning(
            "Vitessce {} supports at most {} simultaneously visible channels; "
            "got {}. Will only render the first {} channels. "
            "You can switch channels in the Vitessce layer controller UI.",
            SPATIAL_VIEW,
            MAX_INITIAL_CHANNELS,
            len(selected_channels),
            MAX_INITIAL_CHANNELS,
        )
        selected_channels = selected_channels[:MAX_INITIAL_CHANNELS]

    palette_rgb = _resolve_palette(palette)
    no_palette_provided = palette is None or len(palette) == 0
    if no_palette_provided:
        if len(selected_channels) == 1:
            logger.info(
                "No palette provided and one channel selected; rendering channel in white (#FFFFFF)."
            )
        else:
            logger.info(
                "No palette provided and {} channels selected; rendering with the default channel palette.",
                len(selected_channels),
            )

    # ignored when photometricInterpretation is RGB (handled by early return).
    image_layer["imageChannel"] = CL(
        [
            {
                "spatialTargetC": channel,
                "spatialChannelColor": (
                    [255, 255, 255]
                    if no_palette_provided and len(selected_channels) == 1
                    else _channel_color(pos, palette_rgb)
                ),
                "spatialChannelVisible": True,
                "spatialChannelWindow": None,
            }
            for pos, channel in enumerate(selected_channels)
        ]
    )
    image_layer["photometricInterpretation"] = "BlackIsZero"
    return image_layer


def _resolve_image_coordinate_transformations(
    *,
    coordinate_transformations: Sequence[Mapping[str, object]] | None,
    microns_per_pixel: float | tuple[float, float] | None,
) -> list[dict[str, object]] | None:
    """
    Resolve optional file-level OME-NGFF coordinate transformations.

    Notes
    -----
    Vitessce applies file-level ``coordinateTransformations`` *after* the
    OME-NGFF metadata transforms from the Zarr store.
    """
    if coordinate_transformations is not None and microns_per_pixel is not None:
        raise ValueError(
            "Provide either coordinate_transformations or microns_per_pixel, not both."
        )

    if coordinate_transformations is not None:
        return [dict(transform) for transform in coordinate_transformations]

    if microns_per_pixel is None:
        return None

    if isinstance(microns_per_pixel, Real):
        mpp_y = float(microns_per_pixel)
        mpp_x = float(microns_per_pixel)
    else:
        if len(microns_per_pixel) != 2:
            raise ValueError(
                "microns_per_pixel must be a positive float or a (y, x) tuple."
            )
        mpp_y = float(microns_per_pixel[0])
        mpp_x = float(microns_per_pixel[1])

    if mpp_y <= 0 or mpp_x <= 0:
        raise ValueError("microns_per_pixel values must be > 0.")

    logger.warning(
        "Applying microns_per_pixel={} as additional file-level scale multiplier. "
        "This is composed after OME-NGFF metadata transforms.",
        (mpp_y, mpp_x),
    )
    return [{"type": "scale", "scale": [1.0, mpp_y, mpp_x]}]


def affine_matrix_to_ngff_coordinate_transformations(
    affine: Sequence[Sequence[float]] | np.ndarray,
    *,
    atol: float = 1e-8,
    enforce_c_identity: bool = True,
) -> list[dict[str, Any]]:
    """
    Convert an affine transform (c,y,x) to OME-NGFF coordinateTransformations.

    Accepted shapes:
    - (4, 4): homogeneous matrix for (c, y, x, 1)
    - (3, 4): [linear | translation]
    - (3, 3): linear only (translation assumed zero)

    Assumes column-vector convention: x' = A x + t.
    For 4x4, translation is expected in the last column.
    """
    A = np.asarray(affine, dtype=float)

    if A.shape == (4, 4):
        if not np.allclose(A[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
            raise ValueError("For 4x4 input, last row must be [0, 0, 0, 1].")
        linear = A[:3, :3]
        translation = A[:3, 3]
    elif A.shape == (3, 4):
        linear = A[:, :3]
        translation = A[:, 3]
    elif A.shape == (3, 3):
        linear = A
        translation = np.zeros(3, dtype=float)
    else:
        raise ValueError("Affine must have shape (4,4), (3,4), or (3,3).")

    if not np.all(np.isfinite(linear)) or not np.all(np.isfinite(translation)):
        raise ValueError("Affine contains non-finite values.")

    # NGFF-compatible here means: scale (+ optional translation), no rotation/shear.
    diag_only = np.diag(np.diag(linear))
    if not np.allclose(linear, diag_only, atol=atol):
        raise ValueError(
            "Not NGFF scale/translation-only: off-diagonal terms found (rotation/shear)."
        )

    sc, sy, sx = np.diag(linear).astype(float)
    tc, ty, tx = translation.astype(float)

    if (
        np.isclose(sc, 0.0, atol=atol)
        or np.isclose(sy, 0.0, atol=atol)
        or np.isclose(sx, 0.0, atol=atol)
    ):
        raise ValueError("Scale contains zero, which is invalid.")

    if enforce_c_identity:
        if not np.isclose(sc, 1.0, atol=atol) or not np.isclose(tc, 0.0, atol=atol):
            raise ValueError(
                "Expected channel axis unchanged: c-scale=1 and c-translation=0."
            )
        sc, tc = 1.0, 0.0

    # OME-NGFF v0.4 order: scale first, optional translation second.
    coordinate_transformations: list[dict[str, Any]] = [
        {"type": "scale", "scale": [float(sc), float(sy), float(sx)]}
    ]

    if not np.allclose([tc, ty, tx], [0.0, 0.0, 0.0], atol=atol):
        coordinate_transformations.append(
            {"type": "translation", "translation": [float(tc), float(ty), float(tx)]}
        )

    return coordinate_transformations
