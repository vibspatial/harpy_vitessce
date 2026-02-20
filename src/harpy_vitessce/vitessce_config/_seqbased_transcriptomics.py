import uuid
from pathlib import Path
from typing import Literal, Mapping, Sequence

from loguru import logger
from vitessce import (
    AnnDataWrapper,
    ImageOmeZarrWrapper,
    SpatialDataWrapper,
    VitessceConfig,
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
    FEATURE_TYPE_GENE,
    FEATURE_TYPE_QC,
    FEATURE_VALUE_TYPE_EXPRESSION,
    FEATURE_VALUE_TYPE_QC,
    LAYER_CONTROLLER_VIEW,
    OBS_COLOR_CELL_SET_SELECTION,
    OBS_COLOR_GENE_SELECTION,
    OBS_TYPE_BIN,
    OBS_TYPE_SPOT,
    SPATIAL_VIEW,
)
from harpy_vitessce.vitessce_config._image import (
    _resolve_image_coordinate_transformations,
    build_image_layer_config,
)
from harpy_vitessce.vitessce_config._utils import _normalize_path_or_url


def single_channel_image(
    img_source: str | Path,  # local path relative to base_dir or remote URL
    name: str = "Visium HD Image",
    description: str = "Visium HD image-only view",
    schema_version: str = "1.0.18",
    base_dir: str | Path | None = None,
    center: tuple[float, float] | None = None,
    zoom: float | None = -4,
    photometric_interpretation: Literal[
        "RGB", "BlackIsZero", "BlackWhite"
    ] = "BlackIsZero",
    microns_per_pixel: float | tuple[float, float] | None = None,
    coordinate_transformations: Sequence[Mapping[str, object]] | None = None,
) -> VitessceConfig:
    """
    Build a Vitessce configuration that visualizes only the OME-Zarr image.

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
    photometric_interpretation
        Image photometric interpretation used by the spatial image layer.
        Use ``"BlackIsZero"`` for single-channel grayscale images and ``"RGB"``
        for RGB images. ``"BlackWhite"`` is accepted as a backwards-compatible
        alias for ``"BlackIsZero"``.
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
    image_coordinate_transformations = _resolve_image_coordinate_transformations(
        coordinate_transformations=coordinate_transformations,
        microns_per_pixel=microns_per_pixel,
    )
    if photometric_interpretation == "BlackWhite":
        logger.warning(
            "photometric_interpretation='BlackWhite' is deprecated; "
            "normalizing to 'BlackIsZero'."
        )
        photometric_interpretation = "BlackIsZero"
    if photometric_interpretation not in {"RGB", "BlackIsZero"}:
        raise ValueError(
            "photometric_interpretation must be one of: "
            "'RGB', 'BlackIsZero' (or legacy alias 'BlackWhite')."
        )
    if center is not None and len(center) != 2:
        raise ValueError("center must be a tuple of two floats: (x, y).")
    if zoom is not None and center is None:
        logger.warning(
            "zoom was provided without center. Vitessce may ignore zoom unless "
            "center is also set."
        )
    if center is not None and zoom is None:
        logger.warning(
            "center was provided without zoom. Vitessce may ignore center unless "
            "zoom is also set."
        )

    vc = VitessceConfig(
        schema_version=schema_version,
        description=description,
        # base_dir only applies to local *_path entries.
        base_dir=(
            None if is_img_remote else (str(base_dir) if base_dir is not None else None)
        ),
    )

    spatial_zoom, spatial_target_x, spatial_target_y = vc.add_coordination(
        ct.SPATIAL_ZOOM,
        ct.SPATIAL_TARGET_X,
        ct.SPATIAL_TARGET_Y,
    )

    if zoom is not None:
        spatial_zoom.set_value(zoom)
    if center is not None:
        spatial_target_x.set_value(center[0])
        spatial_target_y.set_value(center[1])

    file_uuid = f"img_single_channel_{uuid.uuid4()}"  # can be set to any value
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

    spatial_plot.use_coordination(spatial_zoom, spatial_target_x, spatial_target_y)
    image_layer = {
        "fileUid": file_uuid,
        "spatialLayerVisible": True,
        "spatialLayerOpacity": 1.0,
        "photometricInterpretation": photometric_interpretation,
    }
    if photometric_interpretation != "RGB":
        image_layer["imageChannel"] = CL(
            [
                {
                    "spatialTargetC": 0,
                    "spatialChannelColor": [255, 255, 255],
                    "spatialChannelVisible": True,
                    "spatialChannelOpacity": 1.0,
                }
            ]
        )

    vc.link_views_by_dict(
        [spatial_plot, layer_controller],
        {"imageLayer": CL([image_layer])},
    )
    layer_controller.set_props(
        disableChannelsIfRgbDetected=(photometric_interpretation == "RGB")
    )
    vc.layout(hconcat(spatial_plot, layer_controller, split=[10, 2]))

    return vc


def visium_hd_from_spatialdata(
    sdata_path,  # TODO: change to spatialdata_source
    img_layer: str,
    table_layer: str,
    labels_layer: str,
    name: str = "Visium HD",
    description: str = "Visium HD",
    schema_version: str = "1.0.18",
    base_dir: str | Path | None = None,
    center: tuple[float, float] | None = None,
    zoom: float | None = -4,  # e.g. -4
    visualize_as_rgb: bool = True,
    channels: Sequence[int | str] | None = None,
    palette: Sequence[str] | None = None,
    to_coordinate_system: str = "global",
    spot_radius_size_micron: int = 8,
    cluster_key: str | None = "leiden",
    cluster_key_display_name: str = "Leiden",
    embedding_key: str | None = "X_umap",
    embedding_display_name: str = "UMAP",
    qc_obs_feature_keys: str | Sequence[str] | None = (
        "total_counts",
        "n_genes_by_counts",
    ),
    emb_radius_mode: Literal["auto", "manual"] = "auto",
    emb_radius: int = 3,  # ignored if emb_radius_mode is "auto"
) -> VitessceConfig:
    """
    Build a Vitessce configuration for exploring Visium HD data.

    Parameters
    ----------
    sdata_path
        ``SpatialData`` object. When provided, image source is resolved
        as ``sdata.path / "images" / img_layer`` and table source as
        ``sdata.path / "tables" / table_layer``.
    img_layer
        Image layer name under ``images`` in ``sdata``. Required when ``sdata``
        is provided.
    table_layer
        Table layer name under ``tables`` in ``sdata``. Required when ``sdata``
        is provided.
        Required field is ``obsm/{spatial_key}``.
        Optional fields are ``obs/{cluster_key}``, ``obsm/{embedding_key}``,
        and ``obs/{key}`` for each entry in ``qc_obs_feature_keys``.
        When optional keys are provided, missing fields will still cause Vitessce
        data loading/view rendering failures for the corresponding component.
    name
        Dataset name shown in Vitessce.
    description
        Configuration description.
    schema_version
        Vitessce schema version.
    base_dir
        Optional base directory for relative local paths in the config.
        Remote URLs are used as-is. When both ``img_source`` and ``adata_source``
        are remote URLs, ``base_dir`` is ignored and set to ``None`` in the
        generated Vitessce config.
        Also ignored when ``sdata`` is provided.
    center
        Initial spatial target as ``(x, y)`` camera center coordinates.
        Use ``None`` to keep Vitessce defaults.
    zoom
        Initial spatial zoom level. Use ``None`` to keep Vitessce defaults.
    visualize_as_rgb
        If ``True``, render the image layer with ``photometricInterpretation="RGB"``.
        If ``False``, render with ``photometricInterpretation="BlackIsZero"``.
    channels
        Initial channels rendered in the image layer.
        Entries can be integer channel indices or channel names.
        If ``None``, defaults to ``[0, 1, 2]`` when ``visualize_as_rgb=True``,
        otherwise ``[0]``.
    palette
        Optional list of channel colors in hex format (``"#RRGGBB"``) used
        by position for selected channels in non-RGB mode.
    microns_per_pixel
        Convenience option to add a file-level scale transform on ``(y, x)``.
        A scalar applies isotropically.
        Values are multiplicative scale factors.
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
    spot_radius_size_micron
        Spot radius in microns used by the spatial spot layer.
    spatial_key
        Key under ``obsm`` used for spot coordinates, e.g. ``"spatial"`` -> ``"obsm/spatial"``.
        Note: it is assumed that the coordinates of these spots are in micron coordinates.
    cluster_key
        Key under ``obs`` used for cluster/cell-set annotations, e.g. ``"leiden"`` -> ``"obs/leiden"``.
        Set to ``None`` to disable cluster/cell-set views and color encoding.
    cluster_key_display_name
        Display label for the cluster annotation in the Vitessce UI.
    embedding_key
        Key under ``obsm`` used for embedding coordinates, e.g. ``"X_umap"`` -> ``"obsm/X_umap"``.
        Set to ``None`` to disable the embedding scatterplot view.
    embedding_display_name
        Display label for the embedding in the Vitessce UI and scatterplot mapping.
    qc_obs_feature_keys
        QC feature keys under ``obs`` exposed in the QC feature list and histogram,
        e.g. ``("total_counts", "n_genes_by_counts")`` or ``"total_counts"``.
        Set to ``None`` (or an empty sequence) to disable QC views.
    emb_radius_mode
        Embedding point radius mode. Must be ``"auto"`` or ``"manual"``.
    emb_radius
        Embedding point radius value used by Vitessce. Ignored when
        ``emb_radius_mode="auto"``.

    Returns
    -------
    VitessceConfig
        A configured Vitessce configuration object with spatial and gene
        expression views, plus optional cluster/embedding/QC views.

    Raises
    ------
    ValueError
        If ``spatial_key`` is empty, a provided ``cluster_key``/``embedding_key`` is
        empty, ``qc_obs_feature_keys`` contains empty keys, or
        ``emb_radius_mode`` is not ``"auto"``/``"manual"``, or ``center`` is not a
        2-item tuple, or ``spot_radius_size_micron <= 0``, or ``emb_radius <= 0`` when
        ``emb_radius_mode="manual"``, or if required source inputs are missing.

    Examples
    --------
    .. code-block:: python

        from IPython.display import display, HTML

        vc = visium_hd(
            base_dir="/your/path/"
            img_source="data/sample_image.ome.zarr", # relative to base_dir
            adata_source="data/sample_adata.zarr", # relative to base_dir
            qc_obs_feature_keys=("total_counts", "pct_counts_mt"),
        )
        url = vc.web_app()
        display(HTML(f'<a href="{url}" target="_blank">Open in Vitessce</a>'))

    """

    def _normalize_optional_key(key: str | None, name: str) -> str | None:
        if key is None:
            return None
        if not key:
            raise ValueError(f"{name} must be a non-empty string when provided.")
        return key

    def _normalize_qc_keys(
        keys: str | Sequence[str] | None,
    ) -> tuple[str, ...]:
        if keys is None:
            return ()
        if isinstance(keys, str):
            keys = (keys,)
        normalized = tuple(keys)
        if any(not key for key in normalized):
            raise ValueError("qc_obs_feature_keys cannot contain empty keys.")
        return normalized

    cluster_key = _normalize_optional_key(cluster_key, "cluster_key")
    embedding_key = _normalize_optional_key(embedding_key, "embedding_key")
    qc_obs_feature_keys = _normalize_qc_keys(qc_obs_feature_keys)

    normalized_sdata_path, is_sdata_remote = _normalize_path_or_url(
        sdata_path,
        "sdata_path",
    )

    has_clusters = cluster_key is not None
    has_embedding = embedding_key is not None
    has_qc = len(qc_obs_feature_keys) > 0

    if has_embedding and emb_radius_mode not in {"auto", "manual"}:
        raise ValueError(
            "emb_radius_mode must be either 'auto' or 'manual'; "
            f"got {emb_radius_mode!r}."
        )
    if spot_radius_size_micron <= 0:
        raise ValueError(
            "spot_radius_size_micron must be > 0 so spatial spot radius is valid."
        )
    if has_embedding and emb_radius_mode == "manual" and emb_radius <= 0:
        raise ValueError("emb_radius must be > 0 when emb_radius_mode='manual'.")
    if center is not None and len(center) != 2:
        raise ValueError("center must be a tuple of two floats: (x, y).")
    if zoom is not None and center is None:
        logger.warning(
            "zoom was provided without center. Vitessce ignores zoom unless "
            "center is also set."
        )
    if center is not None and zoom is None:
        logger.warning(
            "center was provided without zoom. Vitessce ignores center unless "
            "zoom is also set."
        )

    vc = VitessceConfig(
        schema_version=schema_version,
        description=description,
        # base_dir only applies to local *_path entries.
        base_dir=(
            None
            if is_sdata_remote
            else (str(base_dir) if base_dir is not None else None)
        ),
    )

    spatial_zoom, spatial_target_x, spatial_target_y = vc.add_coordination(
        ct.SPATIAL_ZOOM,
        ct.SPATIAL_TARGET_X,
        ct.SPATIAL_TARGET_Y,
    )

    if zoom is not None:
        spatial_zoom.set_value(zoom)
    if center is not None:
        spatial_target_x.set_value(center[0])
        spatial_target_y.set_value(center[1])

    table_path = f"tables/{table_layer}"

    file_uuid = f"seqbased_{uuid.uuid4()}"

    # Add expression data to the configuration.
    # NOTE: for the spatialdatawrapper linking between labels layer and table happens via the instance key
    # (with a fall back to the index of the AnnData table).
    expression_wrapper = SpatialDataWrapper(
        sdata_path=normalized_sdata_path,
        # The following paths are relative to the root of the SpatialData zarr store on-disk.
        image_path=f"images/{img_layer}",
        table_path=table_path,
        obs_feature_matrix_path=f"{table_path}/X",
        obs_segmentations_path=f"labels/{labels_layer}",
        obs_set_paths=[f"{table_path}/obs/{cluster_key}"]
        if cluster_key is not None
        else None,
        obs_set_names=[cluster_key_display_name] if cluster_key is not None else None,
        obs_embedding_paths=[f"{table_path}/obsm/{embedding_key}"]
        if embedding_key is not None
        else None,
        obs_embedding_names=[
            embedding_display_name
        ]  # embedding_display_name should be renamed to embedding_key_display_name
        if embedding_key is not None
        else None,
        region=labels_layer,
        coordinate_system=to_coordinate_system,
        coordination_values={
            # The following tells Vitessce to consider each observation as a "bin"
            "obsType": OBS_TYPE_BIN,
            "featureType": FEATURE_TYPE_GENE,
            "featureValueType": FEATURE_VALUE_TYPE_EXPRESSION,
            "fileUid": file_uuid,
        },
    )
    dataset = vc.add_dataset(name=name).add_object(expression_wrapper)

    if has_qc:
        # SpatialDataWrapper currently does not expose obs_feature_column_paths
        # in its file definition, so use an AnnDataWrapper for table-level QC columns.
        # NOTE!!! for this to work, it requires that the index of the AnnData matches the ID's in the labels layer.
        qc_table_source = (
            f"{normalized_sdata_path.rstrip('/')}/{table_path}"
            if is_sdata_remote
            else str(Path(normalized_sdata_path) / table_path)
        )
        qc_wrapper_kwargs: dict[str, object] = {
            "obs_feature_matrix_path": None,
            "obs_feature_column_paths": [f"obs/{key}" for key in qc_obs_feature_keys],
            "coordination_values": {
                # The following tells Vitessce to consider each observation as a "bin"
                "obsType": OBS_TYPE_BIN,  # TODO: move to constants of harpy_vitessce
                "featureType": FEATURE_TYPE_QC,
                "featureValueType": FEATURE_VALUE_TYPE_QC,
                "fileUid": file_uuid,
            },
        }
        qc_wrapper_kwargs["adata_url" if is_sdata_remote else "adata_path"] = (
            qc_table_source
        )
        dataset.add_object(AnnDataWrapper(**qc_wrapper_kwargs))

    # 1) create views:
    # i) gene expression (+ optional clusters / embedding)
    spatial_plot = vc.add_view(SPATIAL_VIEW, dataset=dataset)
    if has_clusters:
        spatial_plot.set_props(
            title=f"{cluster_key_display_name} Clusters + Gene Expression"
        )
    else:
        spatial_plot.set_props(title="Gene Expression")

    layer_controller = vc.add_view(LAYER_CONTROLLER_VIEW, dataset=dataset)
    genes = vc.add_view(cm.FEATURE_LIST, dataset=dataset)
    cell_sets = vc.add_view(cm.OBS_SETS, dataset=dataset) if has_clusters else None
    umap = (
        vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping=embedding_display_name)
        if has_embedding
        else None
    )
    # ii) qc
    histogram = (
        vc.add_view(cm.FEATURE_VALUE_HISTOGRAM, dataset=dataset) if has_qc else None
    )
    spatial_qc = vc.add_view(SPATIAL_VIEW, dataset=dataset) if has_qc else None
    if spatial_qc is not None:
        spatial_qc.set_props(title="QC")
    qc_list = vc.add_view(cm.FEATURE_LIST, dataset=dataset) if has_qc else None
    if qc_list is not None:
        qc_list.set_props(title="QC list")

    # 2) add coordination (that will then be used on the views)
    # i) gene expression (+ optional clusters / embedding)
    obs_type, feat_type, feat_val_type, obs_color, feat_sel, obs_set_sel = (
        vc.add_coordination(
            ct.OBS_TYPE,
            ct.FEATURE_TYPE,
            ct.FEATURE_VALUE_TYPE,
            ct.OBS_COLOR_ENCODING,
            ct.FEATURE_SELECTION,
            ct.OBS_SET_SELECTION,
        )
    )
    obs_type.set_value(OBS_TYPE_BIN)
    feat_type.set_value(
        FEATURE_TYPE_GENE
    )  # defined in coordination_values when we add addata
    feat_val_type.set_value(FEATURE_VALUE_TYPE_EXPRESSION)
    # When clusters are unavailable, default to gene-based coloring.
    obs_color.set_value(
        OBS_COLOR_CELL_SET_SELECTION if has_clusters else OBS_COLOR_GENE_SELECTION
    )
    obs_set_sel.set_value(None)

    emb_radius_mode_coord = None
    emb_radius_coord = None
    if has_embedding:
        emb_radius_mode_coord, emb_radius_coord = vc.add_coordination(
            ct.EMBEDDING_OBS_RADIUS_MODE,
            ct.EMBEDDING_OBS_RADIUS,
        )
        emb_radius_mode_coord.set_value(emb_radius_mode)
        emb_radius_coord.set_value(emb_radius)

    # ii) qc
    if has_qc:
        obs_color_qc, feat_type_qc, feat_val_type_qc, feat_sel_qc, obs_set_sel_qc = (
            vc.add_coordination(
                ct.OBS_COLOR_ENCODING,
                ct.FEATURE_TYPE,
                ct.FEATURE_VALUE_TYPE,
                ct.FEATURE_SELECTION,
                ct.OBS_SET_SELECTION,
            )
        )
        obs_color_qc.set_value(OBS_COLOR_GENE_SELECTION)  # use feature values for QC
        feat_type_qc.set_value(FEATURE_TYPE_QC)
        feat_val_type_qc.set_value(FEATURE_VALUE_TYPE_QC)
        feat_sel_qc.set_value([qc_obs_feature_keys[0]])
        obs_set_sel_qc.set_value(None)

    # 3) use coordination on the views
    # i) gene expression (+ optional clusters / embedding)
    spatial_plot.use_coordination(
        obs_type, feat_type, feat_val_type, obs_color, feat_sel, obs_set_sel
    )
    spatial_plot.use_coordination(spatial_zoom, spatial_target_x, spatial_target_y)
    genes.use_coordination(obs_type, obs_color, feat_sel, feat_type, feat_val_type)
    if cell_sets is not None:
        cell_sets.use_coordination(obs_type, obs_set_sel, obs_color)
    if (
        umap is not None
        and emb_radius_mode_coord is not None
        and emb_radius_coord is not None
    ):
        umap.use_coordination(
            obs_type,
            feat_type,
            feat_val_type,
            obs_color,
            feat_sel,
            obs_set_sel,
            emb_radius_mode_coord,
            emb_radius_coord,
        )
    # ii) qc
    if spatial_qc is not None and qc_list is not None and histogram is not None:
        spatial_qc.use_coordination(
            obs_type,
            obs_color_qc,
            feat_sel_qc,
            feat_type_qc,
            feat_val_type_qc,
            obs_set_sel_qc,
        )
        spatial_qc.use_coordination(spatial_zoom, spatial_target_x, spatial_target_y)
        qc_list.use_coordination(
            obs_type, obs_color_qc, feat_sel_qc, feat_type_qc, feat_val_type_qc
        )
        histogram.use_coordination(
            obs_type,
            feat_type_qc,
            feat_val_type_qc,
            feat_sel_qc,
            # obs_color_qc,
            # obs_set_sel_qc # this does not work -> lasso on the qc is not passed to qc
        )

    # note that it is also possible to create two seqmentation layers, one for qc, one for clusters+genes
    # but then we need two layer controllers, which looks kinda weird in the UI
    linked_views = [spatial_plot, layer_controller]
    if spatial_qc is not None:
        linked_views.append(spatial_qc)

    image_layer = build_image_layer_config(
        file_uid=file_uuid,
        channels=channels,
        palette=palette,
        visualize_as_rgb=visualize_as_rgb,
    )

    vc.link_views_by_dict(
        linked_views,
        {
            "imageLayer": CL([image_layer]),
            "segmentationLayer": CL(
                [
                    {
                        "fileUid": file_uuid,
                        "segmentationChannel": CL(
                            [
                                {
                                    "obsType": OBS_TYPE_BIN,
                                    "spatialChannelOpacity": 0.5,
                                    # "obsColorEncoding": obs_color,
                                    # "featureValueColormapRange": [0, 0.5],
                                }
                            ]
                        ),
                    }
                ]
            ),
        },
    )

    # could also work with adding this, but then we require a layer_controller for qc.
    """
    vc.link_views_by_dict(
        [spatial_qc, layer_controller],
        {
            "imageLayer": CL([image_layer]),
            "segmentationLayer": CL(
                [
                    {
                        "fileUid": "test",
                        "segmentationChannel": CL(
                            [
                                {
                                    "spatialChannelOpacity": 0.5,
                                    "obsColorEncoding": obs_color_qc,
                                    "featureValueColormapRange": [0, 0.5],
                                }
                            ]
                        ),
                    }
                ]
            ),
        },
    )
    """

    layer_controller.set_props(disableChannelsIfRgbDetected=visualize_as_rgb)

    main_column = (
        vconcat(
            spatial_plot,
            umap,
            split=[8, 4],
        )
        if umap is not None
        else spatial_plot
    )

    control_views = [layer_controller, genes]
    if qc_list is not None:
        control_views.append(qc_list)
    if cell_sets is not None:
        control_views.append(cell_sets)

    if has_qc and has_clusters:
        control_split = [3, 4, 3, 2]
    elif has_qc and not has_clusters:
        control_split = [3, 5, 4]
    elif not has_qc and has_clusters:
        control_split = [3, 6, 3]
    else:
        control_split = [3, 9]

    control_column = vconcat(*control_views, split=control_split)

    if spatial_qc is not None and histogram is not None:
        qc_column = vconcat(
            spatial_qc,
            histogram,
            split=[8, 4],
        )
        layout_split = [5, 5, 2] if umap is not None else [6, 4, 2]
        layout_view = hconcat(
            main_column,
            qc_column,
            control_column,
            split=layout_split,
        )
    else:
        layout_split = [9, 3] if umap is not None else [10, 2]
        layout_view = hconcat(main_column, control_column, split=layout_split)

    vc.layout(layout_view)

    return vc


def visium_hd_from_split_sources(
    img_source: str | Path,  # local path relative to base_dir or remote URL
    adata_source: str | Path,  # local path relative to base_dir or remote URL
    name: str = "Visium HD",
    description: str = "Visium HD",
    schema_version: str = "1.0.18",
    base_dir: str | Path | None = None,
    center: tuple[float, float] | None = None,
    zoom: float | None = -4,  # e.g. -4
    visualize_as_rgb: bool = True,
    channels: Sequence[int | str] | None = None,
    palette: Sequence[str] | None = None,
    microns_per_pixel: float | tuple[float, float] | None = None,
    coordinate_transformations: Sequence[Mapping[str, object]] | None = None,
    spot_radius_size_micron: int = 8,
    spatial_key: str = "spatial",  # center of the spots. In micron coordinates
    cluster_key: str | None = "leiden",
    cluster_key_display_name: str = "Leiden",
    embedding_key: str | None = "X_umap",
    embedding_display_name: str = "UMAP",
    qc_obs_feature_keys: str | Sequence[str] | None = (
        "total_counts",
        "n_genes_by_counts",
    ),
    emb_radius_mode: Literal["auto", "manual"] = "auto",
    emb_radius: int = 3,  # ignored if emb_radius_mode is "auto"
) -> VitessceConfig:
    """
    Build a Vitessce configuration for exploring Visium HD data.
    This functions uses the SpatialDataWrapper.

    Parameters
    ----------
    img_source
        Path/URL to the OME-Zarr image. Local paths are relative to ``base_dir``
        when provided.
        You can generate this image with
        :func:`harpy_vitessce.data_utils.xarray_to_ome_zarr` or
        :func:`harpy_vitessce.data_utils.array_to_ome_zarr`.
        Ignored when ``sdata`` is provided.
    adata_source
        Path/URL to the AnnData ``.zarr``/``.h5ad`` source. Local paths are
        relative to ``base_dir`` when provided.
        Required field is ``obsm/{spatial_key}``.
        Optional fields are ``obs/{cluster_key}``, ``obsm/{embedding_key}``,
        and ``obs/{key}`` for each entry in ``qc_obs_feature_keys``.
        When optional keys are provided, missing fields will still cause Vitessce
        data loading/view rendering failures for the corresponding component.
        Ignored when ``sdata`` is provided.
    name
        Dataset name shown in Vitessce.
    description
        Configuration description.
    schema_version
        Vitessce schema version.
    base_dir
        Optional base directory for relative local paths in the config.
        Remote URLs are used as-is. When both ``img_source`` and ``adata_source``
        are remote URLs, ``base_dir`` is ignored and set to ``None`` in the
        generated Vitessce config.
        Also ignored when ``sdata`` is provided.
    center
        Initial spatial target as ``(x, y)`` camera center coordinates.
        Use ``None`` to keep Vitessce defaults.
    zoom
        Initial spatial zoom level. Use ``None`` to keep Vitessce defaults.
    visualize_as_rgb
        If ``True``, render the image layer with ``photometricInterpretation="RGB"``.
        If ``False``, render with ``photometricInterpretation="BlackIsZero"``.
    channels
        Initial channels rendered in the image layer.
        Entries can be integer channel indices or channel names.
        If ``None``, defaults to ``[0, 1, 2]`` when ``visualize_as_rgb=True``,
        otherwise ``[0]``.
    palette
        Optional list of channel colors in hex format (``"#RRGGBB"``) used
        by position for selected channels in non-RGB mode.
    microns_per_pixel
        Convenience option to add a file-level scale transform on ``(y, x)``.
        A scalar applies isotropically.
        Values are multiplicative scale factors.
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
    spot_radius_size_micron
        Spot radius in microns used by the spatial spot layer.
    spatial_key
        Key under ``obsm`` used for spot coordinates, e.g. ``"spatial"`` -> ``"obsm/spatial"``.
        Note: it is assumed that the coordinates of these spots are in micron coordinates.
    cluster_key
        Key under ``obs`` used for cluster/cell-set annotations, e.g. ``"leiden"`` -> ``"obs/leiden"``.
        Set to ``None`` to disable cluster/cell-set views and color encoding.
    cluster_key_display_name
        Display label for the cluster annotation in the Vitessce UI.
    embedding_key
        Key under ``obsm`` used for embedding coordinates, e.g. ``"X_umap"`` -> ``"obsm/X_umap"``.
        Set to ``None`` to disable the embedding scatterplot view.
    embedding_display_name
        Display label for the embedding in the Vitessce UI and scatterplot mapping.
    qc_obs_feature_keys
        QC feature keys under ``obs`` exposed in the QC feature list and histogram,
        e.g. ``("total_counts", "n_genes_by_counts")`` or ``"total_counts"``.
        Set to ``None`` (or an empty sequence) to disable QC views.
    emb_radius_mode
        Embedding point radius mode. Must be ``"auto"`` or ``"manual"``.
    emb_radius
        Embedding point radius value used by Vitessce. Ignored when
        ``emb_radius_mode="auto"``.

    Returns
    -------
    VitessceConfig
        A configured Vitessce configuration object with spatial and gene
        expression views, plus optional cluster/embedding/QC views.

    Raises
    ------
    ValueError
        If ``spatial_key`` is empty, a provided ``cluster_key``/``embedding_key`` is
        empty, ``qc_obs_feature_keys`` contains empty keys, or
        ``emb_radius_mode`` is not ``"auto"``/``"manual"``, or ``center`` is not a
        2-item tuple, or ``spot_radius_size_micron <= 0``, or ``emb_radius <= 0`` when
        ``emb_radius_mode="manual"``, or if required source inputs are missing.

    Examples
    --------
    .. code-block:: python

        from IPython.display import display, HTML

        vc = visium_hd(
            base_dir="/your/path/"
            img_source="data/sample_image.ome.zarr", # relative to base_dir
            adata_source="data/sample_adata.zarr", # relative to base_dir
            qc_obs_feature_keys=("total_counts", "pct_counts_mt"),
        )
        url = vc.web_app()
        display(HTML(f'<a href="{url}" target="_blank">Open in Vitessce</a>'))

    """
    if not spatial_key:
        raise ValueError("spatial_key must be a non-empty string.")

    def _normalize_optional_key(key: str | None, name: str) -> str | None:
        if key is None:
            return None
        if not key:
            raise ValueError(f"{name} must be a non-empty string when provided.")
        return key

    def _normalize_qc_keys(
        keys: str | Sequence[str] | None,
    ) -> tuple[str, ...]:
        if keys is None:
            return ()
        if isinstance(keys, str):
            keys = (keys,)
        normalized = tuple(keys)
        if any(not key for key in normalized):
            raise ValueError("qc_obs_feature_keys cannot contain empty keys.")
        return normalized

    cluster_key = _normalize_optional_key(cluster_key, "cluster_key")
    embedding_key = _normalize_optional_key(embedding_key, "embedding_key")
    qc_obs_feature_keys = _normalize_qc_keys(qc_obs_feature_keys)
    assert img_source is not None
    assert adata_source is not None
    img_source, is_img_remote = _normalize_path_or_url(img_source, "img_source")
    adata_source, is_adata_remote = _normalize_path_or_url(adata_source, "adata_source")

    image_coordinate_transformations = _resolve_image_coordinate_transformations(
        coordinate_transformations=coordinate_transformations,
        microns_per_pixel=microns_per_pixel,
    )

    has_clusters = cluster_key is not None
    has_embedding = embedding_key is not None
    has_qc = len(qc_obs_feature_keys) > 0

    if has_embedding and emb_radius_mode not in {"auto", "manual"}:
        raise ValueError(
            "emb_radius_mode must be either 'auto' or 'manual'; "
            f"got {emb_radius_mode!r}."
        )
    if spot_radius_size_micron <= 0:
        raise ValueError(
            "spot_radius_size_micron must be > 0 so spatial spot radius is valid."
        )
    if has_embedding and emb_radius_mode == "manual" and emb_radius <= 0:
        raise ValueError("emb_radius must be > 0 when emb_radius_mode='manual'.")
    if center is not None and len(center) != 2:
        raise ValueError("center must be a tuple of two floats: (x, y).")
    if zoom is not None and center is None:
        logger.warning(
            "zoom was provided without center. Vitessce ignores zoom unless "
            "center is also set."
        )
    if center is not None and zoom is None:
        logger.warning(
            "center was provided without zoom. Vitessce ignores center unless "
            "zoom is also set."
        )

    vc = VitessceConfig(
        schema_version=schema_version,
        description=description,
        # base_dir only applies to local *_path entries.
        base_dir=(
            None
            if is_img_remote and is_adata_remote
            else (str(base_dir) if base_dir is not None else None)
        ),
    )

    spatial_zoom, spatial_target_x, spatial_target_y = vc.add_coordination(
        ct.SPATIAL_ZOOM,
        ct.SPATIAL_TARGET_X,
        ct.SPATIAL_TARGET_Y,
    )

    if zoom is not None:
        spatial_zoom.set_value(zoom)
    if center is not None:
        spatial_target_x.set_value(center[0])
        spatial_target_y.set_value(center[1])

    # h&e
    _file_uuid = f"img_h&e_{uuid.uuid4()}"  # can be set to any value
    img_wrapper_kwargs: dict[str, object] = {
        "coordination_values": {"fileUid": _file_uuid},
    }
    if image_coordinate_transformations is not None:
        img_wrapper_kwargs["coordinate_transformations"] = (
            image_coordinate_transformations
        )
    img_wrapper_kwargs["img_url" if is_img_remote else "img_path"] = img_source
    dataset = vc.add_dataset(name=name).add_object(
        ImageOmeZarrWrapper(
            **img_wrapper_kwargs,
        )
    )

    # clusters + genes
    expression_wrapper_kwargs: dict[str, object] = {
        "obs_feature_matrix_path": "X",
        "obs_spots_path": f"obsm/{spatial_key}",
        "coordination_values": {
            "obsType": OBS_TYPE_SPOT,
            "featureType": FEATURE_TYPE_GENE,
            "featureValueType": FEATURE_VALUE_TYPE_EXPRESSION,
        },
    }
    expression_wrapper_kwargs["adata_url" if is_adata_remote else "adata_path"] = (
        adata_source
    )
    if has_clusters:
        expression_wrapper_kwargs["obs_set_paths"] = [f"obs/{cluster_key}"]
        expression_wrapper_kwargs["obs_set_names"] = [
            cluster_key_display_name
        ]  # display name in UI
    if has_embedding:
        expression_wrapper_kwargs["obs_embedding_paths"] = [f"obsm/{embedding_key}"]
        expression_wrapper_kwargs["obs_embedding_names"] = [embedding_display_name]

    # add the Anndata for the clusters + genes
    dataset.add_object(
        AnnDataWrapper(
            **expression_wrapper_kwargs,
        )
    )

    # qc
    if has_qc:
        qc_wrapper_kwargs: dict[str, object] = {
            "obs_feature_matrix_path": None,
            "obs_feature_column_paths": [f"obs/{key}" for key in qc_obs_feature_keys],
            "coordination_values": {
                "obsType": OBS_TYPE_SPOT,
                "featureType": FEATURE_TYPE_QC,
                "featureValueType": FEATURE_VALUE_TYPE_QC,
            },
        }
        qc_wrapper_kwargs["adata_url" if is_adata_remote else "adata_path"] = (
            adata_source
        )
        dataset.add_object(
            AnnDataWrapper(
                **qc_wrapper_kwargs,
            )
        )

    # 1) create views:
    # i) gene expression (+ optional clusters / embedding)
    spatial_plot = vc.add_view(SPATIAL_VIEW, dataset=dataset)
    if has_clusters:
        spatial_plot.set_props(
            title=f"{cluster_key_display_name} Clusters + Gene Expression"
        )
    else:
        spatial_plot.set_props(title="Gene Expression")

    layer_controller = vc.add_view(LAYER_CONTROLLER_VIEW, dataset=dataset)
    genes = vc.add_view(cm.FEATURE_LIST, dataset=dataset)
    cell_sets = vc.add_view(cm.OBS_SETS, dataset=dataset) if has_clusters else None
    umap = (
        vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping=embedding_display_name)
        if has_embedding
        else None
    )
    # ii) qc
    histogram = (
        vc.add_view(cm.FEATURE_VALUE_HISTOGRAM, dataset=dataset) if has_qc else None
    )
    spatial_qc = vc.add_view(SPATIAL_VIEW, dataset=dataset) if has_qc else None
    if spatial_qc is not None:
        spatial_qc.set_props(title="QC")
    qc_list = vc.add_view(cm.FEATURE_LIST, dataset=dataset) if has_qc else None
    if qc_list is not None:
        qc_list.set_props(title="QC list")

    # 2) add coordination (that will then be used on the views)
    # i) gene expression (+ optional clusters / embedding)
    obs_type, feat_type, feat_val_type, obs_color, feat_sel, obs_set_sel = (
        vc.add_coordination(
            ct.OBS_TYPE,
            ct.FEATURE_TYPE,
            ct.FEATURE_VALUE_TYPE,
            ct.OBS_COLOR_ENCODING,
            ct.FEATURE_SELECTION,
            ct.OBS_SET_SELECTION,
        )
    )
    obs_type.set_value(OBS_TYPE_SPOT)
    feat_type.set_value(
        FEATURE_TYPE_GENE
    )  # defined in coordination_values when we add addata
    feat_val_type.set_value(FEATURE_VALUE_TYPE_EXPRESSION)
    # When clusters are unavailable, default to gene-based coloring.
    obs_color.set_value(
        OBS_COLOR_CELL_SET_SELECTION if has_clusters else OBS_COLOR_GENE_SELECTION
    )
    obs_set_sel.set_value(None)

    emb_radius_mode_coord = None
    emb_radius_coord = None
    if has_embedding:
        emb_radius_mode_coord, emb_radius_coord = vc.add_coordination(
            ct.EMBEDDING_OBS_RADIUS_MODE,
            ct.EMBEDDING_OBS_RADIUS,
        )
        emb_radius_mode_coord.set_value(emb_radius_mode)
        emb_radius_coord.set_value(emb_radius)

    # ii) qc
    if has_qc:
        obs_color_qc, feat_type_qc, feat_val_type_qc, feat_sel_qc, obs_set_sel_qc = (
            vc.add_coordination(
                ct.OBS_COLOR_ENCODING,
                ct.FEATURE_TYPE,
                ct.FEATURE_VALUE_TYPE,
                ct.FEATURE_SELECTION,
                ct.OBS_SET_SELECTION,
            )
        )
        obs_color_qc.set_value(OBS_COLOR_GENE_SELECTION)  # use feature values for QC
        feat_type_qc.set_value(FEATURE_TYPE_QC)
        feat_val_type_qc.set_value(FEATURE_VALUE_TYPE_QC)
        feat_sel_qc.set_value([qc_obs_feature_keys[0]])
        obs_set_sel_qc.set_value(None)

    # 3) use coordination on the views
    # i) gene expression (+ optional clusters / embedding)
    spatial_plot.use_coordination(
        obs_type, feat_type, feat_val_type, obs_color, feat_sel, obs_set_sel
    )
    spatial_plot.use_coordination(spatial_zoom, spatial_target_x, spatial_target_y)
    genes.use_coordination(obs_type, obs_color, feat_sel, feat_type, feat_val_type)
    if cell_sets is not None:
        cell_sets.use_coordination(obs_type, obs_set_sel, obs_color)
    if (
        umap is not None
        and emb_radius_mode_coord is not None
        and emb_radius_coord is not None
    ):
        umap.use_coordination(
            obs_type,
            feat_type,
            feat_val_type,
            obs_color,
            feat_sel,
            obs_set_sel,
            emb_radius_mode_coord,
            emb_radius_coord,
        )
    # ii) qc
    if spatial_qc is not None and qc_list is not None and histogram is not None:
        spatial_qc.use_coordination(
            obs_type,
            obs_color_qc,
            feat_sel_qc,
            feat_type_qc,
            feat_val_type_qc,
            obs_set_sel_qc,
        )
        spatial_qc.use_coordination(spatial_zoom, spatial_target_x, spatial_target_y)
        qc_list.use_coordination(
            obs_type, obs_color_qc, feat_sel_qc, feat_type_qc, feat_val_type_qc
        )
        histogram.use_coordination(
            obs_type,
            feat_type_qc,
            feat_val_type_qc,
            feat_sel_qc,
            # obs_color_qc,
            # obs_set_sel_qc # this does not work -> lasso on the qc is not passed to qc
        )

    # note that it is also possible to create two spotlayers, one for qc, one for clusters+genes
    # but then we need two layer controllers, which looks kinda weird in the UI
    linked_views = [spatial_plot, layer_controller]
    if spatial_qc is not None:
        linked_views.append(spatial_qc)

    image_layer = build_image_layer_config(
        file_uid=_file_uuid,
        channels=channels,
        palette=palette,
        visualize_as_rgb=visualize_as_rgb,
    )

    vc.link_views_by_dict(
        linked_views,
        {
            "imageLayer": CL([image_layer]),
            "spotLayer": CL(
                [
                    {
                        "obsType": obs_type,
                        "spatialLayerVisible": True,
                        "spatialLayerOpacity": 1.0,
                        "spatialSpotRadius": spot_radius_size_micron,
                        "spatialSpotFilled": True,
                        "spatialSpotStrokeWidth": 1.0,
                        "spatialLayerColor": [255, 255, 255],
                        # NOTE: no obsColorEncoding / featureSelection
                        "tooltipsVisible": True,
                        "tooltipCrosshairsVisible": True,
                    }
                ]
            ),
        },
    )
    layer_controller.set_props(disableChannelsIfRgbDetected=visualize_as_rgb)

    main_column = (
        vconcat(
            spatial_plot,
            umap,
            split=[8, 4],
        )
        if umap is not None
        else spatial_plot
    )

    control_views = [layer_controller, genes]
    if qc_list is not None:
        control_views.append(qc_list)
    if cell_sets is not None:
        control_views.append(cell_sets)

    if has_qc and has_clusters:
        control_split = [3, 4, 3, 2]
    elif has_qc and not has_clusters:
        control_split = [3, 5, 4]
    elif not has_qc and has_clusters:
        control_split = [3, 6, 3]
    else:
        control_split = [3, 9]

    control_column = vconcat(*control_views, split=control_split)

    if spatial_qc is not None and histogram is not None:
        qc_column = vconcat(
            spatial_qc,
            histogram,
            split=[8, 4],
        )
        layout_split = [5, 5, 2] if umap is not None else [6, 4, 2]
        layout_view = hconcat(
            main_column,
            qc_column,
            control_column,
            split=layout_split,
        )
    else:
        layout_split = [9, 3] if umap is not None else [10, 2]
        layout_view = hconcat(main_column, control_column, split=layout_split)

    vc.layout(layout_view)

    return vc
