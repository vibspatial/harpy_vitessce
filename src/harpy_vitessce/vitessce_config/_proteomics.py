import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from loguru import logger
from spatialdata import SpatialData
from vitessce import (
    AnnDataWrapper,
    ImageOmeZarrWrapper,
    ObsSegmentationsOmeZarrWrapper,
    VitessceConfig,
    get_initial_coordination_scope_prefix,
    hconcat,
    vconcat,
)
from vitessce import (
    Component as cm,
)
from vitessce import (
    CoordinationLevel as CL,
)
from vitessce import (
    CoordinationType as ct,
)

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


def macsima(  # maybe we should rename this to proteomics
    sdata: SpatialData | None = None,
    img_layer: str | None = None,
    labels_layer: str | None = None,
    table_layer: str | None = None,
    img_source: str
    | Path
    | None = None,  # local path relative to base_dir or remote URL
    labels_source: str | Path | None = None,
    adata_source: str | Path | None = None,
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
    visualize_feature_matrix: bool = False,
    visualize_heatmap: bool = False,
    spatial_key: str = "spatial",
    labels_key: str = "cell_ID",
    labels_key_display_name: str = "cell ID",
    cluster_key: str | None = None,
    cluster_key_display_name: str = "Clusters",
    embedding_key: str | None = None,
    embedding_display_name: str = "UMAP",
) -> VitessceConfig:
    """
    Build a Vitessce configuration for MACSima image/segmentation visualization.

    Parameters
    ----------
    sdata
        ``SpatialData`` object. When provided, image source is resolved
        as ``sdata.path / "images" / img_layer``.
    img_layer
        Image layer name under ``images`` in ``sdata``. Required when ``sdata``
        is provided.
        Ignored when ``sdata`` is not provided.
    labels_layer
        Labels layer name under ``labels`` in ``sdata``.
        When provided, segmentation boundaries can be rendered in spatial view.
        Ignored when ``sdata`` is not provided.
    table_layer
        Table layer name under ``tables`` in ``sdata``. When provided together
        with ``labels_layer``, enables feature-matrix and/or cluster coloring
        and/or embedding visualization (depending on ``visualize_feature_matrix``,
        ``visualize_heatmap``, ``cluster_key`` and ``embedding_key``).
        Ignored when ``sdata`` is not provided.
    img_source
        Path/URL to an OME-Zarr image. Local paths are relative to ``base_dir``
        when provided.
        Ignored when ``sdata`` is provided.
    labels_source
        Path/URL to an OME-Zarr labels segmentation (``obsSegmentations``).
        Ignored when ``sdata`` is provided.
    adata_source
        Path/URL to an AnnData ``.zarr``/``.h5ad`` source.
        Required when ``visualize_feature_matrix=True``,
        ``visualize_heatmap=True`` and/or ``cluster_key``/``embedding_key`` is
        provided.
        ``X`` must be available when either ``visualize_feature_matrix=True``
        or ``visualize_heatmap=True``.
        Observation indices must match segmentation label IDs when used with
        ``labels_source``/``labels_layer``.
        Ignored when ``sdata`` is provided.
    visualize_feature_matrix
        If ``True``, expose the AnnData ``X`` matrix in a feature list and
        enable ``geneSelection``-based coloring.
    visualize_heatmap
        If ``True``, expose a heatmap view driven by the AnnData ``X`` matrix.
        This is independent from ``visualize_feature_matrix``.
    spatial_key
        Key under ``obsm`` used for cell coordinates,
        e.g. ``"spatial"`` -> ``"obsm/spatial"``.
    labels_key
        Key under ``obs`` used for cell labels,
        e.g. ``"cell_ID"`` -> ``"obs/cell_ID"``.
        These keys should map to the provided segmentations mask via ``labels_source`` or labels_layer. # TODO: currently ignored by vitessce
    labels_key_display_name
        Display label for ``labels_key`` in the Vitessce UI.  # TODO: currently ignored by vitessce
    cluster_key
        Optional key under ``obs`` used for categorical cell-set annotations,
        e.g. ``"kronos"`` -> ``"obs/kronos"``.
        Set to ``None`` to disable cluster/cell-set views and color encoding.
    cluster_key_display_name
        Display label for the cluster annotation in the Vitessce UI.
    embedding_key
        Optional key under ``obsm`` used for embedding coordinates,
        e.g. ``"X_umap"`` -> ``"obsm/X_umap"``.
        Set to ``None`` to disable the UMAP scatterplot view.
    embedding_display_name
        Display label for the embedding in the Vitessce UI and scatterplot mapping.
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
        A configured Vitessce configuration object with image-only views, and
        optional segmentation/feature/obs-set/embedding/heatmap views depending
        on the selected AnnData visualization options.

    Raises
    ------
    ValueError
        If ``cluster_key`` is provided as an empty string.
        If ``cluster_key_display_name`` is empty when ``cluster_key`` is provided.
        If ``embedding_key`` is provided as an empty string.
        If ``embedding_display_name`` is empty when ``embedding_key`` is provided.
        If ``spatial_key`` is empty.
        If AnnData-based visualization is requested and ``labels_key`` is empty.
        If AnnData-based visualization is requested and ``labels_key_display_name`` is empty.
        If ``center`` is provided but is not a 2-item tuple.
        If ``sdata`` is provided but ``img_layer`` is missing.
        If neither ``img_source`` nor ``sdata`` is provided.
        If ``sdata.path`` is ``None``.
        If AnnData-based visualization is requested but ``table_layer``/``adata_source``
        is missing.
        If AnnData-based visualization is requested without
        ``labels_source``/``labels_layer``.
    """
    if cluster_key is not None and not cluster_key:
        raise ValueError("cluster_key must be a non-empty string when provided.")
    if cluster_key is not None and not cluster_key_display_name:
        raise ValueError(
            "cluster_key_display_name must be non-empty when cluster_key is provided."
        )
    if embedding_key is not None and not embedding_key:
        raise ValueError("embedding_key must be a non-empty string when provided.")
    if embedding_key is not None and not embedding_display_name:
        raise ValueError(
            "embedding_display_name must be non-empty when embedding_key is provided."
        )
    if not spatial_key:
        raise ValueError("spatial_key must be a non-empty string.")

    has_feature_matrix = visualize_feature_matrix
    has_heatmap = visualize_heatmap
    has_matrix_data = has_feature_matrix or has_heatmap
    has_clusters = cluster_key is not None
    has_embedding = embedding_key is not None
    needs_adata = has_matrix_data or has_clusters or has_embedding
    if needs_adata and not labels_key:
        raise ValueError(
            "labels_key must be a non-empty string when AnnData-based visualization is requested."
        )
    if needs_adata and not labels_key_display_name:
        raise ValueError(
            "labels_key_display_name must be non-empty when AnnData-based visualization is requested."
        )

    if sdata is not None:
        if img_layer is None:
            raise ValueError("img_layer is required when sdata is provided.")
        if needs_adata and table_layer is None:
            raise ValueError(
                "table_layer is required when sdata is provided and "
                "visualize_feature_matrix=True or visualize_heatmap=True or "
                "cluster_key/embedding_key is provided."
            )
        if img_source is not None:
            logger.warning(
                "Both sdata and img_source were provided; img_source is ignored and "
                "image source is resolved from sdata.path/images/{}.",
                img_layer,
            )
        if labels_source is not None:
            logger.warning(
                "Both sdata and labels_source were provided; labels_source is ignored and "
                "labels source is resolved from sdata.path/labels/{}.",
                labels_layer,
            )
        if adata_source is not None:
            logger.warning(
                "Both sdata and adata_source were provided; adata_source is ignored and "
                "table source is resolved from sdata.path/tables/{}.",
                table_layer,
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
        labels_source = (
            Path(sdata.path) / "labels" / labels_layer
            if labels_layer is not None
            else None
        )
        if table_layer is not None:
            if needs_adata:
                adata_source = Path(sdata.path) / "tables" / table_layer
            else:
                logger.warning(
                    "table_layer was provided, but both visualize_feature_matrix=False "
                    "and visualize_heatmap=False and cluster_key/embedding_key are None; "
                    "table data is ignored."
                )
                adata_source = None
        else:
            adata_source = None
        base_dir = None
    elif img_source is None:
        raise ValueError("Either img_source or sdata must be provided.")

    if not needs_adata and adata_source is not None:
        logger.warning(
            "adata_source was provided, but visualize_feature_matrix=False, "
            "visualize_heatmap=False and cluster_key/embedding_key are None; "
            "AnnData is ignored."
        )
        adata_source = None

    img_source, is_img_remote = _normalize_path_or_url(img_source, "img_source")
    if labels_source is not None:
        labels_source, is_labels_remote = _normalize_path_or_url(
            labels_source, "labels_source"
        )
    else:
        is_labels_remote = False
    if adata_source is not None:
        adata_source, is_adata_remote = _normalize_path_or_url(
            adata_source, "adata_source"
        )
    else:
        is_adata_remote = False

    if needs_adata and adata_source is None:
        raise ValueError(
            "adata_source/table_layer is required when visualize_feature_matrix=True, "
            "visualize_heatmap=True or cluster_key/embedding_key is provided."
        )
    if needs_adata and labels_source is None:
        raise ValueError(
            "labels_source/labels_layer is required when visualize_feature_matrix=True, "
            "visualize_heatmap=True or cluster_key/embedding_key is provided."
        )

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
    # TODO -> check that same transformation defined on labels layer if sdata is provided and fetched from labels layer. If not raise ValueError.

    if center is not None and len(center) != 2:
        raise ValueError("center must be a tuple of two floats: (x, y).")
    if not 0.0 <= layer_opacity <= 1.0:
        raise ValueError("layer_opacity must be between 0.0 and 1.0.")

    all_sources_remote = (
        is_img_remote
        and (labels_source is None or is_labels_remote)
        and (adata_source is None or is_adata_remote)
    )
    vc = VitessceConfig(
        schema_version=schema_version,
        description=description,
        # base_dir only applies to local *_path entries.
        base_dir=(
            None
            if all_sources_remote
            else (str(base_dir) if base_dir is not None else None)
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

    labels_file_uuid: str | None = None
    if labels_source is not None:
        labels_file_uuid = f"seg_macsima_{uuid.uuid4()}"
        seg_wrapper_kwargs: dict[str, object] = {
            "coordination_values": {"fileUid": labels_file_uuid},
        }
        if image_coordinate_transformations is not None:
            seg_wrapper_kwargs["coordinate_transformations"] = (
                image_coordinate_transformations
            )
        seg_wrapper_kwargs["img_url" if is_labels_remote else "img_path"] = (
            labels_source
        )
        dataset.add_object(ObsSegmentationsOmeZarrWrapper(**seg_wrapper_kwargs))

    if needs_adata:
        assert adata_source is not None
        adata_wrapper_kwargs: dict[str, object] = {
            "obs_locations_path": f"obsm/{spatial_key}",
            "obs_labels_paths": f"obs/{labels_key}",
            "obs_labels_names": labels_key_display_name,
            "obs_feature_matrix_path": "X" if has_matrix_data else None,
            "coordination_values": {"obsType": "cell"},
        }
        if has_matrix_data:
            adata_wrapper_kwargs["coordination_values"].update(
                {"featureType": "marker", "featureValueType": "intensity"}
            )
        if has_clusters:
            assert cluster_key is not None
            adata_wrapper_kwargs["obs_set_paths"] = [f"obs/{cluster_key}"]
            adata_wrapper_kwargs["obs_set_names"] = [cluster_key_display_name]
        if has_embedding:
            assert embedding_key is not None
            adata_wrapper_kwargs["obs_embedding_paths"] = [f"obsm/{embedding_key}"]
            adata_wrapper_kwargs["obs_embedding_names"] = [embedding_display_name]
        adata_wrapper_kwargs["adata_url" if is_adata_remote else "adata_path"] = (
            adata_source
        )
        dataset.add_object(AnnDataWrapper(**adata_wrapper_kwargs))

    spatial_plot = vc.add_view(SPATIAL_VIEW, dataset=dataset)
    layer_controller = vc.add_view(LAYER_CONTROLLER_VIEW, dataset=dataset)
    feature_list = (
        vc.add_view(cm.FEATURE_LIST, dataset=dataset) if has_feature_matrix else None
    )
    heatmap = vc.add_view(cm.HEATMAP, dataset=dataset) if has_heatmap else None
    obs_sets = vc.add_view(cm.OBS_SETS, dataset=dataset) if has_clusters else None
    umap = (
        vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping=embedding_display_name)
        if has_embedding
        else None
    )

    if spatial_coordination_scopes:
        spatial_plot.use_coordination(*spatial_coordination_scopes)

    obs_color = None
    if has_matrix_data and has_clusters:
        (
            obs_type,
            feat_type,
            feat_val_type,
            obs_color,
            feat_sel,
            obs_set_sel,
        ) = vc.add_coordination(  # coordinate them all, because we want to switch between feature matrix and cluster key.
            ct.OBS_TYPE,
            ct.FEATURE_TYPE,
            ct.FEATURE_VALUE_TYPE,
            ct.OBS_COLOR_ENCODING,
            ct.FEATURE_SELECTION,
            ct.OBS_SET_SELECTION,
        )
        obs_type.set_value("cell")
        feat_type.set_value("marker")
        feat_val_type.set_value("intensity")
        obs_color.set_value(
            "cellSetSelection"
        )  # we default to cluster key coloring if we can choose between features and clusters.
        feat_sel.set_value(None)
        obs_set_sel.set_value(None)

        spatial_plot.use_coordination(
            obs_type, feat_type, feat_val_type, obs_color, feat_sel, obs_set_sel
        )
        if feature_list is not None:
            feature_list.use_coordination(
                obs_type, obs_color, feat_sel, feat_type, feat_val_type
            )
        if obs_sets is not None:
            obs_sets.use_coordination(obs_type, obs_set_sel, obs_color)
        if heatmap is not None:
            heatmap.use_coordination(
                obs_type, feat_type, feat_val_type, feat_sel, obs_set_sel
            )
        if umap is not None:
            umap.use_coordination(
                obs_type, feat_type, feat_val_type, obs_color, feat_sel, obs_set_sel
            )
    elif has_matrix_data:
        obs_type, feat_type, feat_val_type, obs_color, feat_sel = vc.add_coordination(
            ct.OBS_TYPE,
            ct.FEATURE_TYPE,
            ct.FEATURE_VALUE_TYPE,
            ct.OBS_COLOR_ENCODING,
            ct.FEATURE_SELECTION,
        )
        obs_type.set_value("cell")
        feat_type.set_value("marker")
        feat_val_type.set_value("intensity")
        obs_color.set_value("geneSelection")
        feat_sel.set_value(None)

        spatial_plot.use_coordination(
            obs_type, feat_type, feat_val_type, obs_color, feat_sel
        )
        if feature_list is not None:
            feature_list.use_coordination(
                obs_type, obs_color, feat_sel, feat_type, feat_val_type
            )
        if heatmap is not None:
            heatmap.use_coordination(obs_type, feat_type, feat_val_type, feat_sel)
        if umap is not None:
            umap.use_coordination(
                obs_type, feat_type, feat_val_type, obs_color, feat_sel
            )
    elif has_clusters:
        obs_type, obs_color, obs_set_sel = vc.add_coordination(
            ct.OBS_TYPE,
            ct.OBS_COLOR_ENCODING,
            ct.OBS_SET_SELECTION,
        )
        obs_type.set_value("cell")
        obs_color.set_value("cellSetSelection")
        obs_set_sel.set_value(None)

        spatial_plot.use_coordination(obs_type, obs_color, obs_set_sel)
        if obs_sets is not None:
            obs_sets.use_coordination(obs_type, obs_set_sel, obs_color)
        if umap is not None:
            umap.use_coordination(obs_type, obs_color, obs_set_sel)
    elif has_embedding:
        (obs_type,) = vc.add_coordination(ct.OBS_TYPE)
        obs_type.set_value("cell")
        spatial_plot.use_coordination(obs_type)
        if umap is not None:
            umap.use_coordination(obs_type)

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
        scope_prefix=get_initial_coordination_scope_prefix("A", "image"),
    )

    if labels_file_uuid is not None:
        segmentation_channel: dict[str, object] = {
            "spatialTargetC": 0,
            "spatialChannelOpacity": 0.75,
        }
        if obs_color is not None:
            segmentation_channel["obsColorEncoding"] = obs_color
        if has_matrix_data:
            segmentation_channel["featureValueColormapRange"] = [0, 1]
        vc.link_views_by_dict(
            [spatial_plot, layer_controller],
            {
                "segmentationLayer": CL(
                    [
                        {
                            "fileUid": labels_file_uuid,
                            "segmentationChannel": CL([segmentation_channel]),
                        }
                    ]
                )
            },
            scope_prefix=get_initial_coordination_scope_prefix("A", "obsSegmentations"),
        )

    """
    # lasso broken on segmentationlayer.
    if has_clusters or has_feature_matrix:
        vc.link_views_by_dict(
            [spatial_plot, layer_controller],
            {
                "spotLayer": CL(
                    [
                        {
                            "obsType": obs_type,
                            "spatialLayerVisible": True,
                            "spatialLayerOpacity": 0.2,
                            "spatialSpotRadius": 1.5,
                            "spatialSpotFilled": True,
                            "spatialSpotStrokeWidth": 0.0,
                            "spatialLayerColor": [255, 255, 255],
                            "tooltipsVisible": True,
                            "tooltipCrosshairsVisible": True,
                        }
                    ]
                ),
            },
            scope_prefix=get_initial_coordination_scope_prefix("A", "obsSpots"),
        )
    """

    layer_controller.set_props(disableChannelsIfRgbDetected=False)
    # Layout strategy:
    # - No feature list: 2 columns (spatial + right stack).
    # - Feature list present: 3 columns (spatial + middle stack + right stack).
    if feature_list is None:
        # Requested order in 2-column mode:
        # layer_controller -> heatmap -> umap -> obs_sets
        right_views = [layer_controller]
        if heatmap is not None:
            right_views.append(heatmap)
        if umap is not None:
            right_views.append(umap)
        if obs_sets is not None:
            right_views.append(obs_sets)

        if len(right_views) == 1:
            right_column = right_views[0]
        elif len(right_views) == 2:
            right_column = vconcat(*right_views, split=[4, 8])
        elif len(right_views) == 3:
            # Keep final panel compact (typically obs_sets).
            right_column = vconcat(*right_views, split=[3, 6, 3])
        else:
            # layer_controller, heatmap, umap, obs_sets (obs_sets compact).
            right_column = vconcat(*right_views, split=[2, 4, 4, 2])

        vc.layout(hconcat(spatial_plot, right_column, split=[8, 4]))
    else:
        middle_views = [layer_controller]
        if heatmap is not None:
            middle_views.append(heatmap)
        if umap is not None:
            middle_views.append(umap)

        if len(middle_views) == 1:
            middle_column = middle_views[0]
        elif len(middle_views) == 2:
            middle_column = vconcat(*middle_views, split=[4, 8])
        else:
            middle_column = vconcat(*middle_views, split=[3, 5, 4])

        right_views = [feature_list]
        if obs_sets is not None:
            right_views.append(obs_sets)
        if len(right_views) == 1:
            right_column = right_views[0]
        else:
            # Keep obs_sets compact in 3-column mode too.
            right_column = vconcat(*right_views, split=[8, 4])

        vc.layout(hconcat(spatial_plot, middle_column, right_column, split=[6, 3, 3]))

    return vc
