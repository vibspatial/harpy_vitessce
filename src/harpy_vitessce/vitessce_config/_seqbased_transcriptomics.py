import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

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
from vitessce.config import VitessceConfigDataset

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
from harpy_vitessce.vitessce_config._utils import (
    _normalize_path_or_url,
    _validate_camera,
)


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


@dataclass(frozen=True)
class _SeqBasedModes:
    has_clusters: bool
    has_embedding: bool
    has_qc: bool


@dataclass(frozen=True)
class _SeqBasedViews:
    spatial_plot: Any
    layer_controller: Any
    genes: Any
    cell_sets: Any | None
    umap: Any | None
    histogram: Any | None
    spatial_qc: Any | None
    qc_list: Any | None


def _normalize_optional_key(key: str | None, name: str) -> str | None:
    if key is None:
        return None
    if not key:
        raise ValueError(f"{name} must be a non-empty string when provided.")
    return key


def _normalize_qc_keys(keys: str | Sequence[str] | None) -> tuple[str, ...]:
    if keys is None:
        return ()
    if isinstance(keys, str):
        keys = (keys,)
    normalized = tuple(keys)
    if any(not key for key in normalized):
        raise ValueError("qc_obs_feature_keys cannot contain empty keys.")
    return normalized


def _compute_modes(
    *,
    cluster_key: str | None,
    embedding_key: str | None,
    qc_obs_feature_keys: tuple[str, ...],
) -> _SeqBasedModes:
    return _SeqBasedModes(
        has_clusters=cluster_key is not None,
        has_embedding=embedding_key is not None,
        has_qc=len(qc_obs_feature_keys) > 0,
    )


def _validate_seqbased_options(
    *,
    modes: _SeqBasedModes,
    emb_radius_mode: Literal["auto", "manual"],
    emb_radius: int,
    spot_radius_size_micron: int | None = None,
    center: tuple[float, float] | None,
    zoom: float | None,
) -> None:
    if modes.has_embedding and emb_radius_mode not in {"auto", "manual"}:
        raise ValueError(
            "emb_radius_mode must be either 'auto' or 'manual'; "
            f"got {emb_radius_mode!r}."
        )
    if spot_radius_size_micron is not None and spot_radius_size_micron <= 0:
        raise ValueError(
            "spot_radius_size_micron must be > 0 so spatial spot radius is valid."
        )
    if modes.has_embedding and emb_radius_mode == "manual" and emb_radius <= 0:
        raise ValueError("emb_radius must be > 0 when emb_radius_mode='manual'.")
    _validate_camera(center=center, zoom=zoom)


def _apply_seqbased_layout(
    vc: VitessceConfig, *, views: _SeqBasedViews, modes: _SeqBasedModes
) -> None:
    main_column = (
        vconcat(views.spatial_plot, views.umap, split=[8, 4])
        if views.umap is not None
        else views.spatial_plot
    )

    control_views = [views.layer_controller, views.genes]
    if views.qc_list is not None:
        control_views.append(views.qc_list)
    if views.cell_sets is not None:
        control_views.append(views.cell_sets)

    if modes.has_qc and modes.has_clusters:
        control_split = [3, 4, 3, 2]
    elif modes.has_qc:
        control_split = [3, 5, 4]
    elif modes.has_clusters:
        control_split = [3, 6, 3]
    else:
        control_split = [3, 9]
    control_column = vconcat(*control_views, split=control_split)

    if views.spatial_qc is not None and views.histogram is not None:
        qc_column = vconcat(views.spatial_qc, views.histogram, split=[8, 4])
        layout_split = [5, 5, 2] if views.umap is not None else [6, 4, 2]
        vc.layout(
            hconcat(
                main_column,
                qc_column,
                control_column,
                split=layout_split,
            )
        )
        return

    layout_split = [9, 3] if views.umap is not None else [10, 2]
    vc.layout(hconcat(main_column, control_column, split=layout_split))


def _build_seqbased_visualization(
    vc: VitessceConfig,
    *,
    dataset: VitessceConfigDataset,
    file_uuid: str,
    obs_type_value: str,
    use_spot_layer: bool,
    spot_radius_size_micron: int | None,
    modes: _SeqBasedModes,
    cluster_key_display_name: str,
    embedding_display_name: str,
    qc_obs_feature_keys: tuple[str, ...],
    emb_radius_mode: Literal["auto", "manual"],
    emb_radius: int,
    center: tuple[float, float] | None,
    zoom: float | None,
    visualize_as_rgb: bool,
    channels: Sequence[int | str] | None,
    palette: Sequence[str] | None,
) -> None:
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

    spatial_plot = vc.add_view(SPATIAL_VIEW, dataset=dataset)
    spatial_plot.set_props(
        title=(
            f"{cluster_key_display_name} Clusters + Gene Expression"
            if modes.has_clusters
            else "Gene Expression"
        )
    )

    views = _SeqBasedViews(
        spatial_plot=spatial_plot,
        layer_controller=vc.add_view(LAYER_CONTROLLER_VIEW, dataset=dataset),
        genes=vc.add_view(cm.FEATURE_LIST, dataset=dataset),
        cell_sets=vc.add_view(cm.OBS_SETS, dataset=dataset)
        if modes.has_clusters
        else None,
        umap=vc.add_view(
            cm.SCATTERPLOT,
            dataset=dataset,
            mapping=embedding_display_name,
        )
        if modes.has_embedding
        else None,
        histogram=vc.add_view(cm.FEATURE_VALUE_HISTOGRAM, dataset=dataset)
        if modes.has_qc
        else None,
        spatial_qc=vc.add_view(SPATIAL_VIEW, dataset=dataset) if modes.has_qc else None,
        qc_list=vc.add_view(cm.FEATURE_LIST, dataset=dataset) if modes.has_qc else None,
    )
    if views.spatial_qc is not None:
        views.spatial_qc.set_props(title="QC")
    if views.qc_list is not None:
        views.qc_list.set_props(title="QC list")

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
    obs_type.set_value(obs_type_value)
    feat_type.set_value(FEATURE_TYPE_GENE)
    feat_val_type.set_value(FEATURE_VALUE_TYPE_EXPRESSION)
    obs_color.set_value(
        OBS_COLOR_CELL_SET_SELECTION if modes.has_clusters else OBS_COLOR_GENE_SELECTION
    )
    obs_set_sel.set_value(None)

    emb_radius_mode_coord = None
    emb_radius_coord = None
    if modes.has_embedding:
        emb_radius_mode_coord, emb_radius_coord = vc.add_coordination(
            ct.EMBEDDING_OBS_RADIUS_MODE,
            ct.EMBEDDING_OBS_RADIUS,
        )
        emb_radius_mode_coord.set_value(emb_radius_mode)
        emb_radius_coord.set_value(emb_radius)

    if modes.has_qc:
        obs_color_qc, feat_type_qc, feat_val_type_qc, feat_sel_qc, obs_set_sel_qc = (
            vc.add_coordination(
                ct.OBS_COLOR_ENCODING,
                ct.FEATURE_TYPE,
                ct.FEATURE_VALUE_TYPE,
                ct.FEATURE_SELECTION,
                ct.OBS_SET_SELECTION,
            )
        )
        obs_color_qc.set_value(OBS_COLOR_GENE_SELECTION)
        feat_type_qc.set_value(FEATURE_TYPE_QC)
        feat_val_type_qc.set_value(FEATURE_VALUE_TYPE_QC)
        feat_sel_qc.set_value([qc_obs_feature_keys[0]])
        obs_set_sel_qc.set_value(None)

    views.spatial_plot.use_coordination(
        obs_type, feat_type, feat_val_type, obs_color, feat_sel, obs_set_sel
    )
    views.spatial_plot.use_coordination(
        spatial_zoom,
        spatial_target_x,
        spatial_target_y,
    )
    views.genes.use_coordination(
        obs_type,
        obs_color,
        feat_sel,
        feat_type,
        feat_val_type,
    )
    if views.cell_sets is not None:
        views.cell_sets.use_coordination(obs_type, obs_set_sel, obs_color)
    if (
        views.umap is not None
        and emb_radius_mode_coord is not None
        and emb_radius_coord is not None
    ):
        views.umap.use_coordination(
            obs_type,
            feat_type,
            feat_val_type,
            obs_color,
            feat_sel,
            obs_set_sel,
            emb_radius_mode_coord,
            emb_radius_coord,
        )
    if (
        views.spatial_qc is not None
        and views.qc_list is not None
        and views.histogram is not None
    ):
        views.spatial_qc.use_coordination(
            obs_type,
            obs_color_qc,
            feat_sel_qc,
            feat_type_qc,
            feat_val_type_qc,
            obs_set_sel_qc,
        )
        views.spatial_qc.use_coordination(
            spatial_zoom,
            spatial_target_x,
            spatial_target_y,
        )
        views.qc_list.use_coordination(
            obs_type,
            obs_color_qc,
            feat_sel_qc,
            feat_type_qc,
            feat_val_type_qc,
        )
        views.histogram.use_coordination(
            obs_type,
            feat_type_qc,
            feat_val_type_qc,
            feat_sel_qc,
        )

    linked_views = [views.spatial_plot, views.layer_controller]
    if views.spatial_qc is not None:
        linked_views.append(views.spatial_qc)

    image_layer = build_image_layer_config(
        file_uid=file_uuid,
        channels=channels,
        palette=palette,
        visualize_as_rgb=visualize_as_rgb,
    )
    linked_layers: dict[str, CL] = {"imageLayer": CL([image_layer])}

    if use_spot_layer:
        linked_layers["spotLayer"] = CL(
            [
                {
                    "obsType": obs_type,
                    "spatialLayerVisible": True,
                    "spatialLayerOpacity": 1.0,
                    "spatialSpotRadius": spot_radius_size_micron,
                    "spatialSpotFilled": True,
                    "spatialSpotStrokeWidth": 1.0,
                    "spatialLayerColor": [255, 255, 255],
                    "tooltipsVisible": True,
                    "tooltipCrosshairsVisible": True,
                }
            ]
        )
    else:
        linked_layers["segmentationLayer"] = CL(
            [
                {
                    "fileUid": file_uuid,
                    "segmentationChannel": CL(
                        [
                            {
                                "obsType": obs_type_value,
                                "spatialChannelOpacity": 0.75,
                            }
                        ]
                    ),
                }
            ]
        )

    vc.link_views_by_dict(linked_views, linked_layers)
    views.layer_controller.set_props(disableChannelsIfRgbDetected=visualize_as_rgb)
    _apply_seqbased_layout(vc, views=views, modes=modes)


def _add_spatialdata_wrappers(
    vc: VitessceConfig,
    *,
    name: str,
    sdata_path: str,
    is_sdata_remote: bool,
    img_layer: str,
    table_layer: str,
    labels_layer: str,
    to_coordinate_system: str,
    modes: _SeqBasedModes,
    cluster_key: str | None,
    cluster_key_display_name: str,
    embedding_key: str | None,
    embedding_display_name: str,
    qc_obs_feature_keys: tuple[str, ...],
) -> tuple[VitessceConfigDataset, str]:
    table_path = f"tables/{table_layer}"
    file_uuid = f"seqbased_{uuid.uuid4()}"
    expression_wrapper = SpatialDataWrapper(
        sdata_path=sdata_path,  # need to fix the stuff with the spatialdata URL here.
        image_path=f"images/{img_layer}",
        table_path=table_path,
        obs_feature_matrix_path=f"{table_path}/X",
        obs_segmentations_path=f"labels/{labels_layer}",
        obs_set_paths=[f"{table_path}/obs/{cluster_key}"]
        if modes.has_clusters
        else None,
        obs_set_names=[cluster_key_display_name] if modes.has_clusters else None,
        obs_embedding_paths=[f"{table_path}/obsm/{embedding_key}"]
        if modes.has_embedding
        else None,
        obs_embedding_names=[embedding_display_name] if modes.has_embedding else None,
        region=labels_layer,
        coordinate_system=to_coordinate_system,
        coordination_values={
            "obsType": OBS_TYPE_BIN,
            "featureType": FEATURE_TYPE_GENE,
            "featureValueType": FEATURE_VALUE_TYPE_EXPRESSION,
            "fileUid": file_uuid,
        },
    )
    dataset = vc.add_dataset(name=name).add_object(expression_wrapper)

    if modes.has_qc:
        qc_table_source = (
            f"{sdata_path.rstrip('/')}/{table_path}"
            if is_sdata_remote
            else str(Path(sdata_path) / table_path)
        )
        qc_wrapper_kwargs: dict[str, object] = {
            "obs_feature_matrix_path": None,
            "obs_feature_column_paths": [f"obs/{key}" for key in qc_obs_feature_keys],
            "coordination_values": {
                "obsType": OBS_TYPE_BIN,
                "featureType": FEATURE_TYPE_QC,
                "featureValueType": FEATURE_VALUE_TYPE_QC,
                "fileUid": file_uuid,
            },
        }
        qc_wrapper_kwargs["adata_url" if is_sdata_remote else "adata_path"] = (
            qc_table_source
        )
        dataset.add_object(AnnDataWrapper(**qc_wrapper_kwargs))

    return dataset, file_uuid


def _add_raw_wrappers(
    vc: VitessceConfig,
    *,
    name: str,
    img_source: str,
    adata_source: str,
    is_img_remote: bool,
    is_adata_remote: bool,
    image_coordinate_transformations: Sequence[Mapping[str, object]] | None,
    spatial_key: str,
    modes: _SeqBasedModes,
    cluster_key: str | None,
    cluster_key_display_name: str,
    embedding_key: str | None,
    embedding_display_name: str,
    qc_obs_feature_keys: tuple[str, ...],
) -> tuple[VitessceConfigDataset, str]:
    file_uuid = f"img_h&e_{uuid.uuid4()}"
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
    if modes.has_clusters:
        expression_wrapper_kwargs["obs_set_paths"] = [f"obs/{cluster_key}"]
        expression_wrapper_kwargs["obs_set_names"] = [cluster_key_display_name]
    if modes.has_embedding:
        expression_wrapper_kwargs["obs_embedding_paths"] = [f"obsm/{embedding_key}"]
        expression_wrapper_kwargs["obs_embedding_names"] = [embedding_display_name]
    dataset.add_object(AnnDataWrapper(**expression_wrapper_kwargs))

    if modes.has_qc:
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
        dataset.add_object(AnnDataWrapper(**qc_wrapper_kwargs))

    return dataset, file_uuid


def seqbased_transcriptomics_from_spatialdata(
    sdata_path: str | Path,
    img_layer: str,
    labels_layer: str,
    table_layer: str,
    base_dir: str | Path | None = None,
    name: str = "Visium HD",
    description: str = "Visium HD",
    schema_version: str = "1.0.18",
    center: tuple[float, float] | None = None,
    zoom: float | None = -4,
    visualize_as_rgb: bool = True,
    channels: Sequence[int | str] | None = None,
    palette: Sequence[str] | None = None,
    to_coordinate_system: str = "global",
    cluster_key: str | None = "leiden",
    cluster_key_display_name: str = "Leiden",
    embedding_key: str | None = "X_umap",
    embedding_display_name: str = "UMAP",
    qc_obs_feature_keys: str | Sequence[str] | None = (
        "total_counts",
        "n_genes_by_counts",
    ),
    emb_radius_mode: Literal["auto", "manual"] = "auto",
    emb_radius: int = 3,
) -> VitessceConfig:
    """
    Build a seq-based transcriptomics Vitessce configuration from a SpatialData store using SpatialDataWrapper class.

    This mode renders bins through a segmentation layer (labels layer + table).
    For spot-based geometries (for example hexagonal bins), use
    :func:`seqbased_transcriptomics_from_split_sources`.

    Note that this will only work if spots are square bins, because then they can be represented via a grid
    of bin ID's (labels_layer). For hexagonal spots (e.g. Nova-ST, use visium_hd_from_split_sources), and specify the center of the hexagons via .obsm[spatial_key]

    Note that currently the SpatialDataWrapper in Vitessce does not support obs_feature_column_paths.
    So they are added using the AnnDataWrapper. Because obs_labels_path is currently ignored in AnnDataWrapper,
    the index of the table needs to match the ID in the labels layer.

    Parameters
    ----------
    sdata_path
        Path or URL to the SpatialData zarr root.
    img_layer
        Image layer name under ``images`` in SpatialData.
        Required.
    labels_layer
        Labels layer name under ``labels`` in SpatialData.
        Required when table-driven visualizations are enabled.
        This layer should be annotated by ``table_layer`` via ``region_key`` and ``instance_key``.
    table_layer
        Table layer name under ``tables`` in SpatialData.
        The table is expected to annotate ``labels_layer``:
        ``region_key`` is the key in ``adata.obs`` that specifies the region
        (i.e. the ``labels_layer``), and ``instance_key`` is the key in
        ``adata.obs`` that specifies the instance (i.e. instance IDs in
        ``labels_layer``).
        If this annotation metadata is unavailable, Vitessce falls back to
        linking via the table ``obs`` index. Vitessce currently applies
        JavaScript ``parseInt`` to observation indices, so values like
        ``"123_cellA"`` still map to label ID ``123``.
        Note that currently the SpatialDataWrapper in Vitessce does not support obs_feature_column_paths,
        so if ``qc_obs_feature_keys`` is not None, they are added using the AnnDataWrapper. Because obs_labels_path is currently ignored in AnnDataWrapper,
        the index of the table needs to match the ID in the labels layer if ``qc_obs_feature_keys`` is not None.
    base_dir
        Optional base directory for local path resolution in the emitted config.
        Ignored when ``sdata_path`` is a remote URL.
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
    to_coordinate_system
        Coordinate system key passed to ``SpatialDataWrapper``.
        Used to resolve image/labels rendering in a shared coordinate system.
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
        If a provided ``cluster_key``/``embedding_key`` is
        empty, ``qc_obs_feature_keys`` contains empty keys, or
        ``emb_radius_mode`` is not ``"auto"``/``"manual"``, or ``center`` is not a
        2-item tuple, or ``emb_radius <= 0`` when
        ``emb_radius_mode="manual"``, or if required source inputs are missing.

    Examples
    --------
    .. code-block:: python

        from IPython.display import display, HTML

        vc = visium_hd(
            sdata_path="/your/path/sdata.zarr" # relative to base_dir
            img_layer="your_img_layer",
            labels_layer="your_labels_layer",
            table_layer="your_table_layer", # index should match ID's in labels_layer, because qc_obs_feature_keys is not None.
            qc_obs_feature_keys=("total_counts", "pct_counts_mt"),
        )
        url = vc.web_app()
        display(HTML(f'<a href="{url}" target="_blank">Open in Vitessce</a>'))
    """
    cluster_key = _normalize_optional_key(cluster_key, "cluster_key")
    embedding_key = _normalize_optional_key(embedding_key, "embedding_key")
    normalized_qc_keys = _normalize_qc_keys(qc_obs_feature_keys)
    modes = _compute_modes(
        cluster_key=cluster_key,
        embedding_key=embedding_key,
        qc_obs_feature_keys=normalized_qc_keys,
    )
    _validate_seqbased_options(
        modes=modes,
        emb_radius_mode=emb_radius_mode,
        emb_radius=emb_radius,
        spot_radius_size_micron=None,
        center=center,
        zoom=zoom,
    )

    normalized_sdata_path, is_sdata_remote = _normalize_path_or_url(
        sdata_path,
        "sdata_path",
    )
    vc = VitessceConfig(
        schema_version=schema_version,
        description=description,
        base_dir=(
            None
            if is_sdata_remote
            else (str(base_dir) if base_dir is not None else None)
        ),
    )

    dataset, file_uuid = _add_spatialdata_wrappers(
        vc,
        name=name,
        sdata_path=normalized_sdata_path,
        is_sdata_remote=is_sdata_remote,
        img_layer=img_layer,
        table_layer=table_layer,
        labels_layer=labels_layer,
        to_coordinate_system=to_coordinate_system,
        modes=modes,
        cluster_key=cluster_key,
        cluster_key_display_name=cluster_key_display_name,
        embedding_key=embedding_key,
        embedding_display_name=embedding_display_name,
        qc_obs_feature_keys=normalized_qc_keys,
    )
    _build_seqbased_visualization(
        vc,
        dataset=dataset,
        file_uuid=file_uuid,
        obs_type_value=OBS_TYPE_BIN,
        use_spot_layer=False,
        spot_radius_size_micron=None,
        modes=modes,
        cluster_key_display_name=cluster_key_display_name,
        embedding_display_name=embedding_display_name,
        qc_obs_feature_keys=normalized_qc_keys,
        emb_radius_mode=emb_radius_mode,
        emb_radius=emb_radius,
        center=center,
        zoom=zoom,
        visualize_as_rgb=visualize_as_rgb,
        channels=channels,
        palette=palette,
    )
    return vc


def seqbased_transcriptomics_from_split_sources(
    img_source: str | Path,
    adata_source: str | Path,
    base_dir: str | Path | None = None,
    name: str = "Visium HD",
    description: str = "Visium HD",
    schema_version: str = "1.0.18",
    center: tuple[float, float] | None = None,
    zoom: float | None = -4,
    visualize_as_rgb: bool = True,
    channels: Sequence[int | str] | None = None,
    palette: Sequence[str] | None = None,
    microns_per_pixel: float | tuple[float, float] | None = None,
    coordinate_transformations: Sequence[Mapping[str, object]] | None = None,
    spot_radius_size_micron: int = 8,
    spatial_key: str = "spatial",
    cluster_key: str | None = "leiden",
    cluster_key_display_name: str = "Leiden",
    embedding_key: str | None = "X_umap",
    embedding_display_name: str = "UMAP",
    qc_obs_feature_keys: str | Sequence[str] | None = (
        "total_counts",
        "n_genes_by_counts",
    ),
    emb_radius_mode: Literal["auto", "manual"] = "auto",
    emb_radius: int = 3,
) -> VitessceConfig:
    """
    Build a seq-based transcriptomics Vitessce configuration from explicit sources.

    This mode renders observations via a spot layer using ``obsm/{spatial_key}``.

    Parameters
    ----------
    img_source
        Path/URL to the OME-Zarr image. Local paths are relative to ``base_dir``
        when provided.
        You can generate this image with
        :func:`harpy_vitessce.data_utils.xarray_to_ome_zarr` or
        :func:`harpy_vitessce.data_utils.array_to_ome_zarr`.
    adata_source
        Path/URL to the AnnData ``.zarr``/``.h5ad`` source. Local paths are
        relative to ``base_dir`` when provided.
        Required field is ``obsm/{spatial_key}``.
        Optional fields are ``obs/{cluster_key}``, ``obsm/{embedding_key}``,
        and ``obs/{key}`` for each entry in ``qc_obs_feature_keys``.
        When optional keys are provided, missing fields will still cause Vitessce
        data loading/view rendering failures for the corresponding component.
    base_dir
        Optional base directory for local path resolution in the emitted config.
        Ignored when ``sdata_path`` is a remote URL.
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
        Convenience option to add a file-level scale transform on ``(y, x)`` of ``img_source`` when rendering.
        A scalar applies isotropically.
        Values are multiplicative scale factors.
        This transform is composed *after* OME-NGFF metadata transforms.
    coordinate_transformations
        Raw file-level OME-NGFF coordinate transformations passed to
        ``ImageOmeZarrWrapper``.
        Mutually exclusive with ``microns_per_pixel``.
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
            sdata_path="/your/path/sdata.zarr" # relative to base_dir
            img_layer="your_img_layer",
            labels_layer="your_labels_layer",
            table_layer="your_table_layer", # index should match ID's in labels_layer, because qc_obs_feature_keys is not None.
            qc_obs_feature_keys=("total_counts", "pct_counts_mt"),
        )
        url = vc.web_app()
        display(HTML(f'<a href="{url}" target="_blank">Open in Vitessce</a>'))

    """
    if not spatial_key:
        raise ValueError("spatial_key must be a non-empty string.")

    cluster_key = _normalize_optional_key(cluster_key, "cluster_key")
    embedding_key = _normalize_optional_key(embedding_key, "embedding_key")
    normalized_qc_keys = _normalize_qc_keys(qc_obs_feature_keys)
    modes = _compute_modes(
        cluster_key=cluster_key,
        embedding_key=embedding_key,
        qc_obs_feature_keys=normalized_qc_keys,
    )
    _validate_seqbased_options(
        modes=modes,
        emb_radius_mode=emb_radius_mode,
        emb_radius=emb_radius,
        spot_radius_size_micron=spot_radius_size_micron,
        center=center,
        zoom=zoom,
    )

    normalized_img_source, is_img_remote = _normalize_path_or_url(
        img_source,
        "img_source",
    )
    normalized_adata_source, is_adata_remote = _normalize_path_or_url(
        adata_source,
        "adata_source",
    )
    image_coordinate_transformations = _resolve_image_coordinate_transformations(
        coordinate_transformations=coordinate_transformations,
        microns_per_pixel=microns_per_pixel,
    )

    vc = VitessceConfig(
        schema_version=schema_version,
        description=description,
        base_dir=(
            None
            if is_img_remote and is_adata_remote
            else (str(base_dir) if base_dir is not None else None)
        ),
    )

    dataset, file_uuid = _add_raw_wrappers(
        vc,
        name=name,
        img_source=normalized_img_source,
        adata_source=normalized_adata_source,
        is_img_remote=is_img_remote,
        is_adata_remote=is_adata_remote,
        image_coordinate_transformations=image_coordinate_transformations,
        spatial_key=spatial_key,
        modes=modes,
        cluster_key=cluster_key,
        cluster_key_display_name=cluster_key_display_name,
        embedding_key=embedding_key,
        embedding_display_name=embedding_display_name,
        qc_obs_feature_keys=normalized_qc_keys,
    )
    _build_seqbased_visualization(
        vc,
        dataset=dataset,
        file_uuid=file_uuid,
        obs_type_value=OBS_TYPE_SPOT,
        use_spot_layer=True,
        spot_radius_size_micron=spot_radius_size_micron,
        modes=modes,
        cluster_key_display_name=cluster_key_display_name,
        embedding_display_name=embedding_display_name,
        qc_obs_feature_keys=normalized_qc_keys,
        emb_radius_mode=emb_radius_mode,
        emb_radius=emb_radius,
        center=center,
        zoom=zoom,
        visualize_as_rgb=visualize_as_rgb,
        channels=channels,
        palette=palette,
    )
    return vc
