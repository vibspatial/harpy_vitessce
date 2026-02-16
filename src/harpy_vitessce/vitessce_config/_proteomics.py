import uuid
from collections.abc import Sequence
from pathlib import Path

from vitessce import (
    CoordinationLevel as CL,
)
from vitessce import (
    CoordinationType as ct,
)
from vitessce import ImageOmeZarrWrapper, VitessceConfig, hconcat

from harpy_vitessce.vitessce_config._constants import (
    LAYER_CONTROLLER_VIEW,
    SPATIAL_VIEW,
)
from harpy_vitessce.vitessce_config._image import build_image_layer_config
from harpy_vitessce.vitessce_config._utils import _normalize_path_or_url


def macsima(
    img_source: str | Path,  # local path relative to base_dir or remote URL
    name: str = "MACSima",
    description: str = "MACSima image-only view",
    schema_version: str = "1.0.18",
    base_dir: str | Path | None = None,
    center: tuple[float, float] | None = None,
    zoom: float | None = -4,
    channels: Sequence[int | str] | None = None,
    palette: Sequence[str] | None = None,
    layer_opacity: float = 1.0,
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
    channels
        Initial channels rendered by spatialBeta.
        Entries can be integer channel indices or channel names.
        If more than 6 channels are provided, only the first 6 are used.
        If ``None``, only channel at index 0 is shown.
        Channel colors are assigned from an internal palette in the order
        of this list (position-based, not value-based).
    palette
        Optional list of channel colors in hex format (``"#RRGGBB"``) used
        by position for selected channels.
    layer_opacity
        Opacity of the image layer in ``[0, 1]``.

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

    image_layer = build_image_layer_config(
        file_uid=file_uuid,
        channels=channels,
        palette=palette,
        visualize_as_rgb=False,
        layer_opacity=layer_opacity,
    )

    vc.link_views_by_dict(
        [spatial_plot, layer_controller],
        {"imageLayer": CL([image_layer])},
    )
    layer_controller.set_props(disableChannelsIfRgbDetected=False)
    vc.layout(hconcat(spatial_plot, layer_controller, split=[8, 4]))

    return vc
