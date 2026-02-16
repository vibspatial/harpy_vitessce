import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from loguru import logger
from spatialdata import SpatialData
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
from harpy_vitessce.vitessce_config._image import (
    _resolve_image_coordinate_transformations,
    _spatialdata_transformation_to_ngff,
    build_image_layer_config,
)
from harpy_vitessce.vitessce_config._utils import _normalize_path_or_url


def macsima(
    sdata: SpatialData | None = None,
    img_layer: str | None = None,
    img_source: str
    | Path
    | None = None,  # local path relative to base_dir or remote URL
    base_dir: str | Path | None = None,
    name: str = "MACSima",
    description: str = "MACSima",
    schema_version: str = "1.0.18",
    center: tuple[float, float] | None = None,
    zoom: float | None = -4,
    channels: Sequence[int | str] | None = None,
    palette: Sequence[str] | None = None,
    layer_opacity: float = 1.0,
    microns_per_pixel: float | tuple[float, float] | None = None,
    coordinate_transformations: Sequence[Mapping[str, object]] | None = None,
    to_coordinate_system: str = "global",
) -> VitessceConfig:
    """
    Build a Vitessce configuration for MACSima image-only visualization.

    Parameters
    ----------
    sdata
        ``SpatialData`` object. When provided, image source is resolved
        as ``sdata.path / "images" / img_layer``.
    img_layer
        Image layer name under ``images`` in ``sdata``. Required when ``sdata``
        is provided.
        Ignored when ``sdata`` is not provided.
    img_source
        Path/URL to an OME-Zarr image. Local paths are relative to ``base_dir``
        when provided.
        Ignored when ``sdata`` is provided.
    base_dir
        Optional base directory for relative local paths in the config.
        Ignored when ``img_source`` is a remote URL.
        Ignored when ``sdata`` is provided.
    name
        Dataset name shown in Vitessce.
    description
        Configuration description.
    schema_version
        Vitessce schema version.
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
    microns_per_pixel
        Convenience option to add a file-level scale transform on ``(y, x)``.
        A scalar applies isotropically.
        Values are multiplicative scale factors (for absolute override, use
        ``desired_pixel_size / source_pixel_size``).
        This transform is composed *after* OME-NGFF metadata transforms.
    coordinate_transformations
        Raw file-level OME-NGFF coordinate transformations passed to
        ``ImageOmeZarrWrapper``.
        Mutually exclusive with ``microns_per_pixel``.
    to_coordinate_system
        Coordinate-system key used only when ``sdata`` is provided and both
        ``microns_per_pixel`` and ``coordinate_transformations`` are ``None``.
        In that case, the transform is read from ``sdata.images[img_layer]``,
        converted to an affine matrix on ``("c", "y", "x")``, and then mapped
        to OME-NGFF ``coordinateTransformations``.
        Typically this is the micron coordinate system.
        Ignored otherwise.

    Returns
    -------
    VitessceConfig
        A configured Vitessce configuration object with image-only views.

    Raises
    ------
    ValueError
        If ``center`` is provided but is not a 2-item tuple.
        If ``sdata`` is provided but ``img_layer`` is missing.
        If neither ``img_source`` nor ``sdata`` is provided.
        If ``sdata.path`` is ``None``.
    """
    if sdata is not None:
        if img_layer is None:
            raise ValueError("img_layer is required when sdata is provided.")
        if img_source is not None:
            logger.warning(
                "Both sdata and img_source were provided; img_source is ignored and "
                "image source is resolved from sdata.path/images/{}.",
                img_layer,
            )
        if base_dir is not None:
            logger.warning(
                "Both sdata and base_dir were provided; base_dir is ignored because "
                "image source is resolved from sdata.path."
            )
        if sdata.path is None:
            raise ValueError(
                "sdata.path is None. Provide a backed SpatialData object or pass img_source directly."
            )
        img_source = Path(sdata.path) / "images" / img_layer
        base_dir = None
    elif img_source is None:
        raise ValueError("Either img_source or sdata must be provided.")

    img_source, is_img_remote = _normalize_path_or_url(img_source, "img_source")

    # resolve the transformation:
    if sdata is not None:
        if coordinate_transformations is None and microns_per_pixel is None:
            logger.info(
                "Both coordinate_transformations and microns_per_pixel is None. "
                "Fetching coordinate transformation from the SpatialData object."
            )
            coordinate_transformations = _spatialdata_transformation_to_ngff(
                sdata,
                layer=img_layer,
                to_coordinate_system=to_coordinate_system,
            )

    image_coordinate_transformations = _resolve_image_coordinate_transformations(
        coordinate_transformations=coordinate_transformations,
        microns_per_pixel=microns_per_pixel,
    )
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
    if image_coordinate_transformations is not None:
        img_wrapper_kwargs["coordinate_transformations"] = (
            image_coordinate_transformations
        )
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
