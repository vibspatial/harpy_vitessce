import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from vitessce import (
    AnnDataWrapper,
    ImageOmeZarrWrapper,
    ObsSegmentationsOmeZarrWrapper,
    SpatialDataWrapper,
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
from vitessce.config import VitessceConfigDataset

from harpy_vitessce.vitessce_config._constants import (
    LAYER_CONTROLLER_VIEW,
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

OBS_TYPE_CELL = "cell"
FEATURE_TYPE_MARKER = "marker"
FEATURE_VALUE_TYPE_INTENSITY = "intensity"
OBS_COLOR_GENE_SELECTION = "geneSelection"
OBS_COLOR_CELL_SET_SELECTION = "cellSetSelection"


@dataclass(frozen=True)
class _ProteomicsModes:
    has_feature_matrix: bool
    has_heatmap: bool
    has_matrix_data: bool
    has_clusters: bool
    has_embedding: bool
    needs_adata: bool


@dataclass(frozen=True)
class _ProteomicsViews:
    spatial_plot: Any
    layer_controller: Any
    feature_list: Any | None
    heatmap: Any | None
    obs_sets: Any | None
    umap: Any | None


def _compute_modes(
    *,
    visualize_feature_matrix: bool,
    visualize_heatmap: bool,
    cluster_key: str | None,
    embedding_key: str | None,
) -> _ProteomicsModes:
    has_feature_matrix = visualize_feature_matrix
    has_heatmap = visualize_heatmap
    has_matrix_data = has_feature_matrix or has_heatmap
    has_clusters = cluster_key is not None
    has_embedding = embedding_key is not None
    return _ProteomicsModes(
        has_feature_matrix=has_feature_matrix,
        has_heatmap=has_heatmap,
        has_matrix_data=has_matrix_data,
        has_clusters=has_clusters,
        has_embedding=has_embedding,
        needs_adata=(has_matrix_data or has_clusters or has_embedding),
    )


def _validate_annotation_keys(
    *,
    cluster_key: str | None,
    cluster_key_display_name: str,
    embedding_key: str | None,
    embedding_key_display_name: str,
) -> None:
    if cluster_key is not None and not cluster_key:
        raise ValueError("cluster_key must be a non-empty string when provided.")
    if cluster_key is not None and not cluster_key_display_name:
        raise ValueError(
            "cluster_key_display_name must be non-empty when cluster_key is provided."
        )
    if embedding_key is not None and not embedding_key:
        raise ValueError("embedding_key must be a non-empty string when provided.")
    if embedding_key is not None and not embedding_key_display_name:
        raise ValueError(
            "embedding_key_display_name must be non-empty when embedding_key is provided."
        )


def _validate_layer_opacity(layer_opacity: float) -> None:
    if not 0.0 <= layer_opacity <= 1.0:
        raise ValueError("layer_opacity must be between 0.0 and 1.0.")


def _apply_layout(vc: VitessceConfig, *, views: _ProteomicsViews) -> None:
    views.layer_controller.set_props(disableChannelsIfRgbDetected=False)

    if views.feature_list is None:
        right_views = [views.layer_controller]
        if views.heatmap is not None:
            right_views.append(views.heatmap)
        if views.umap is not None:
            right_views.append(views.umap)
        if views.obs_sets is not None:
            right_views.append(views.obs_sets)

        if len(right_views) == 1:
            right_column = right_views[0]
        elif len(right_views) == 2:
            right_column = vconcat(*right_views, split=[4, 8])
        elif len(right_views) == 3:
            right_column = vconcat(*right_views, split=[3, 6, 3])
        else:
            right_column = vconcat(*right_views, split=[2, 4, 4, 2])

        vc.layout(hconcat(views.spatial_plot, right_column, split=[8, 4]))
        return

    middle_views = [views.layer_controller]
    if views.heatmap is not None:
        middle_views.append(views.heatmap)
    if views.umap is not None:
        middle_views.append(views.umap)

    if len(middle_views) == 1:
        middle_column = middle_views[0]
    elif len(middle_views) == 2:
        middle_column = vconcat(*middle_views, split=[4, 8])
    else:
        middle_column = vconcat(*middle_views, split=[3, 5, 4])

    right_views = [views.feature_list]
    if views.obs_sets is not None:
        right_views.append(views.obs_sets)
    if len(right_views) == 1:
        right_column = right_views[0]
    else:
        right_column = vconcat(*right_views, split=[8, 4])

    vc.layout(hconcat(views.spatial_plot, middle_column, right_column, split=[6, 3, 3]))


def _build_shared_visualization(
    vc: VitessceConfig,
    *,
    dataset: VitessceConfigDataset,
    file_uuid: str,
    labels_file_uuid: str | None,
    modes: _ProteomicsModes,
    embedding_key_display_name: str,
    center: tuple[float, float] | None,
    zoom: float | None,
    channels: Sequence[int | str] | None,
    palette: Sequence[str] | None,
    layer_opacity: float,
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
    layer_controller = vc.add_view(LAYER_CONTROLLER_VIEW, dataset=dataset)
    feature_list = (
        vc.add_view(cm.FEATURE_LIST, dataset=dataset)
        if modes.has_feature_matrix
        else None
    )
    heatmap = vc.add_view(cm.HEATMAP, dataset=dataset) if modes.has_heatmap else None
    """
    obs_sets = (
        vc.add_view(cm.OBS_SETS, dataset=dataset)
        if modes.has_clusters
        else None
    )
    """
    obs_sets = vc.add_view(cm.OBS_SETS, dataset=dataset) # we choose to always show obs selection (e.g. if user wants to annotate cells)
    umap = (
        vc.add_view(
            cm.SCATTERPLOT,
            dataset=dataset,
            mapping=embedding_key_display_name,
        )
        if modes.has_embedding
        else None
    )

    views = _ProteomicsViews(
        spatial_plot=spatial_plot,
        layer_controller=layer_controller,
        feature_list=feature_list,
        heatmap=heatmap,
        obs_sets=obs_sets,
        umap=umap,
    )
    views.spatial_plot.use_coordination(
        spatial_zoom, spatial_target_x, spatial_target_y
    )

    obs_color = None
    if modes.has_matrix_data and modes.has_clusters:
        (
            obs_type,
            feat_type,
            feat_val_type,
            obs_color,
            feat_sel,
            obs_set_sel,
        ) = vc.add_coordination(
            ct.OBS_TYPE,
            ct.FEATURE_TYPE,
            ct.FEATURE_VALUE_TYPE,
            ct.OBS_COLOR_ENCODING,
            ct.FEATURE_SELECTION,
            ct.OBS_SET_SELECTION,
        )
        obs_type.set_value(OBS_TYPE_CELL)
        feat_type.set_value(FEATURE_TYPE_MARKER)
        feat_val_type.set_value(FEATURE_VALUE_TYPE_INTENSITY)
        obs_color.set_value(OBS_COLOR_CELL_SET_SELECTION)
        feat_sel.set_value(None)
        obs_set_sel.set_value(None)

        views.spatial_plot.use_coordination(
            obs_type,
            feat_type,
            feat_val_type,
            obs_color,
            feat_sel,
            obs_set_sel,
        )
        if views.feature_list is not None:
            views.feature_list.use_coordination(
                obs_type,
                obs_color,
                feat_sel,
                feat_type,
                feat_val_type,
            )
        if views.obs_sets is not None:
            views.obs_sets.use_coordination(obs_type, obs_set_sel, obs_color)
        if views.heatmap is not None:
            views.heatmap.use_coordination(
                obs_type,
                feat_type,
                feat_val_type,
                feat_sel,
                obs_color,
                obs_set_sel,
            )
        if views.umap is not None:
            views.umap.use_coordination(
                obs_type,
                feat_type,
                feat_val_type,
                obs_color,
                feat_sel,
                obs_set_sel,
            )
    elif modes.has_matrix_data:
        (
            obs_type,
            feat_type,
            feat_val_type,
            obs_color,
            feat_sel,
            obs_set_sel,
        ) = vc.add_coordination(
            ct.OBS_TYPE,
            ct.FEATURE_TYPE,
            ct.FEATURE_VALUE_TYPE,
            ct.OBS_COLOR_ENCODING,
            ct.FEATURE_SELECTION,
            ct.OBS_SET_SELECTION,
        )

        # obs_type, feat_type, feat_val_type, obs_color, feat_sel = vc.add_coordination(
        #    ct.OBS_TYPE,
        #    ct.FEATURE_TYPE,
        #    ct.FEATURE_VALUE_TYPE,
        #    ct.OBS_COLOR_ENCODING,
        #    ct.FEATURE_SELECTION,
        # )
        obs_type.set_value(OBS_TYPE_CELL)
        feat_type.set_value(FEATURE_TYPE_MARKER)
        feat_val_type.set_value(FEATURE_VALUE_TYPE_INTENSITY)
        obs_color.set_value(OBS_COLOR_GENE_SELECTION)
        feat_sel.set_value(None)
        obs_set_sel.set_value(None)

        views.spatial_plot.use_coordination(
            obs_type,
            feat_type,
            feat_val_type,
            obs_color,
            feat_sel,
            obs_set_sel,
        )
        if views.feature_list is not None:
            views.feature_list.use_coordination(
                obs_type,
                obs_color,
                feat_sel,
                feat_type,
                feat_val_type,
            )
        if views.obs_sets is not None:
            views.obs_sets.use_coordination(obs_type, obs_set_sel, obs_color)
        if views.heatmap is not None:
            views.heatmap.use_coordination(
                obs_type,
                feat_type,
                feat_val_type,
                feat_sel,
                obs_color,
                obs_set_sel,
            )
        if views.umap is not None:
            views.umap.use_coordination(
                obs_type,
                feat_type,
                feat_val_type,
                obs_color,
                feat_sel,
                obs_set_sel,
            )

    elif modes.has_clusters:
        obs_type, obs_color, obs_set_sel = vc.add_coordination(
            ct.OBS_TYPE,
            ct.OBS_COLOR_ENCODING,
            ct.OBS_SET_SELECTION,
        )
        obs_type.set_value(OBS_TYPE_CELL)
        obs_color.set_value(OBS_COLOR_CELL_SET_SELECTION)
        obs_set_sel.set_value(None)

        views.spatial_plot.use_coordination(obs_type, obs_color, obs_set_sel)
        if views.obs_sets is not None:
            views.obs_sets.use_coordination(obs_type, obs_set_sel, obs_color)
        if views.umap is not None:
            views.umap.use_coordination(obs_type, obs_color, obs_set_sel)
    elif modes.has_embedding:
        (obs_type,) = vc.add_coordination(ct.OBS_TYPE)
        obs_type.set_value(OBS_TYPE_CELL)
        views.spatial_plot.use_coordination(obs_type)
        if views.umap is not None:
            views.umap.use_coordination(obs_type)

    image_layer = build_image_layer_config(
        file_uid=file_uuid,
        channels=channels,
        palette=palette,
        visualize_as_rgb=False,
        layer_opacity=layer_opacity,
    )

    vc.link_views_by_dict(
        [views.spatial_plot, views.layer_controller],
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
        if modes.has_matrix_data:
            segmentation_channel["featureValueColormapRange"] = [0, 1]

        vc.link_views_by_dict(
            [views.spatial_plot, views.layer_controller],
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

    _apply_layout(vc, views=views)


def _add_raw_wrappers(
    vc: VitessceConfig,
    *,
    name: str,
    img_source: str,
    labels_source: str | None,
    adata_source: str | None,
    is_img_remote: bool,
    is_labels_remote: bool,
    is_adata_remote: bool,
    coordinate_transformations_image: Sequence[Mapping[str, object]] | None,
    coordinate_transformations_mask: Sequence[Mapping[str, object]] | None,
    modes: _ProteomicsModes,
    cluster_key: str | None,
    cluster_key_display_name: str,
    embedding_key: str | None,
    embedding_key_display_name: str,
) -> tuple[VitessceConfigDataset, str, str | None]:
    file_uuid = f"img_proteomics_{uuid.uuid4()}"
    img_wrapper_kwargs: dict[str, object] = {
        "coordination_values": {"fileUid": file_uuid},
    }
    if coordinate_transformations_image is not None:
        img_wrapper_kwargs["coordinate_transformations"] = (
            coordinate_transformations_image
        )
    img_wrapper_kwargs["img_url" if is_img_remote else "img_path"] = img_source

    dataset = vc.add_dataset(name=name).add_object(
        ImageOmeZarrWrapper(**img_wrapper_kwargs)
    )

    labels_file_uuid: str | None = None
    if labels_source is not None:
        labels_file_uuid = f"seg_proteomics_{uuid.uuid4()}"
        seg_wrapper_kwargs: dict[str, object] = {
            "coordination_values": {"fileUid": labels_file_uuid},
        }
        if coordinate_transformations_mask is not None:
            seg_wrapper_kwargs["coordinate_transformations"] = (
                coordinate_transformations_mask
            )
        seg_wrapper_kwargs["img_url" if is_labels_remote else "img_path"] = (
            labels_source
        )
        dataset.add_object(ObsSegmentationsOmeZarrWrapper(**seg_wrapper_kwargs))

    if modes.needs_adata:
        assert adata_source is not None
        adata_wrapper_kwargs: dict[str, object] = {
            "obs_feature_matrix_path": "X" if modes.has_matrix_data else None,
            "coordination_values": {"obsType": OBS_TYPE_CELL},
        }
        if modes.has_matrix_data:
            adata_wrapper_kwargs["coordination_values"].update(
                {
                    "featureType": FEATURE_TYPE_MARKER,
                    "featureValueType": FEATURE_VALUE_TYPE_INTENSITY,
                }
            )
        if modes.has_clusters:
            assert cluster_key is not None
            adata_wrapper_kwargs["obs_set_paths"] = [f"obs/{cluster_key}"]
            adata_wrapper_kwargs["obs_set_names"] = [cluster_key_display_name]
        if modes.has_embedding:
            assert embedding_key is not None
            adata_wrapper_kwargs["obs_embedding_paths"] = [f"obsm/{embedding_key}"]
            adata_wrapper_kwargs["obs_embedding_names"] = [embedding_key_display_name]
        adata_wrapper_kwargs["adata_url" if is_adata_remote else "adata_path"] = (
            adata_source
        )
        dataset.add_object(AnnDataWrapper(**adata_wrapper_kwargs))

    return dataset, file_uuid, labels_file_uuid


def _add_spatialdata_wrapper(
    vc: VitessceConfig,
    *,
    name: str,
    sdata_path: str,
    is_sdata_remote: bool,
    img_layer: str,
    labels_layer: str | None,
    table_layer: str | None,
    to_coordinate_system: str,
    modes: _ProteomicsModes,
    cluster_key: str | None,
    cluster_key_display_name: str,
    embedding_key: str | None,
    embedding_key_display_name: str,
) -> tuple[VitessceConfigDataset, str, str | None]:
    file_uuid = f"sdata_{uuid.uuid4()}"
    labels_file_uuid: str | None = file_uuid if labels_layer is not None else None

    table_prefix = f"tables/{table_layer}" if table_layer is not None else None

    file_coordination_values: dict[str, object] = {
        "obsType": OBS_TYPE_CELL,
        "fileUid": file_uuid,
    }
    if modes.has_matrix_data:
        file_coordination_values.update(
            {
                "featureType": FEATURE_TYPE_MARKER,
                "featureValueType": FEATURE_VALUE_TYPE_INTENSITY,
            }
        )

    spatialdata_wrapper_kwargs: dict[str, object] = {
        "table_path": table_prefix,
        "image_path": f"images/{img_layer}",
        "obs_segmentations_path": (
            f"labels/{labels_layer}" if labels_layer is not None else None
        ),
        "obs_feature_matrix_path": (
            f"{table_prefix}/X"
            if (modes.has_matrix_data and table_prefix is not None)
            else None
        ),
        "obs_set_paths": (
            [f"{table_prefix}/obs/{cluster_key}"]
            if (modes.has_clusters and table_prefix is not None and cluster_key is not None)
            else None
        ),
        "obs_set_names": [cluster_key_display_name] if modes.has_clusters else None,
        "region": labels_layer,
        "obs_embedding_paths": (
            [f"{table_prefix}/obsm/{embedding_key}"]
            if (
                modes.has_embedding
                and table_prefix is not None
                and embedding_key is not None
            )
            else None
        ),
        "obs_embedding_names": [embedding_key_display_name]
        if modes.has_embedding
        else None,
        "coordinate_system": to_coordinate_system,
        "coordination_values": file_coordination_values,
    }
    spatialdata_wrapper_kwargs["sdata_url" if is_sdata_remote else "sdata_path"] = (
        sdata_path
    )
    wrapper = SpatialDataWrapper(**spatialdata_wrapper_kwargs)

    dataset = vc.add_dataset(name=name).add_object(wrapper)
    return dataset, file_uuid, labels_file_uuid


def proteomics_from_spatialdata(
    sdata_path: str | Path,
    img_layer: str | None = None,
    labels_layer: str | None = None,
    table_layer: str | None = None,
    base_dir: str | Path | None = None,
    name: str = "Proteomics",
    description: str = "Proteomics",
    schema_version: str = "1.0.18",
    center: tuple[float, float] | None = None,
    zoom: float | None = -4,
    channels: Sequence[int | str] | None = None,
    palette: Sequence[str] | None = None,
    layer_opacity: float = 1.0,
    to_coordinate_system: str = "global",
    visualize_feature_matrix: bool = False,
    visualize_heatmap: bool = False,
    cluster_key: str | None = None,
    cluster_key_display_name: str = "Clusters",
    embedding_key: str | None = None,
    embedding_key_display_name: str = "UMAP",
) -> VitessceConfig:
    """
    Build a Vitessce configuration for proteomics image/segmentation visualization
    from a SpatialData store.

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
        When table-driven visualizations are enabled, this layer should be
        annotated by ``table_layer`` via ``region_key`` and ``instance_key``.
    table_layer
        Table layer name under ``tables`` in SpatialData.
        Required when feature matrix, heatmap, clusters, or embedding
        visualizations are enabled.
        The table is expected to annotate ``labels_layer``:
        ``region_key`` is the key in ``adata.obs`` that specifies the region
        (i.e. the ``labels_layer``), and ``instance_key`` is the key in
        ``adata.obs`` that specifies the instance (i.e. instance IDs in
        ``labels_layer``).
        If this annotation metadata is unavailable, Vitessce falls back to
        linking via the table ``obs`` index. Vitessce currently applies
        JavaScript ``parseInt`` to observation indices, so values like
        ``"123_cellA"`` still map to label ID ``123``.
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
        Initial camera target as ``(x, y)``.
    zoom
        Initial camera zoom.
    channels
        Initial channels rendered by ``spatialBeta``.
        Entries can be integer channel indices or channel names.
    palette
        Optional list of channel colors in ``\"#RRGGBB\"`` format.
    layer_opacity
        Opacity of the image layer in ``[0, 1]``.
    to_coordinate_system
        Coordinate system key passed to ``SpatialDataWrapper``.
        Used to resolve image/labels rendering in a shared coordinate system.
    visualize_feature_matrix
        If ``True``, expose table ``X`` as marker intensities in a feature list.
    visualize_heatmap
        If ``True``, expose a heatmap view driven by table ``X``.
    cluster_key
        Optional key under table ``obs`` used for cell-set annotations.
        Set to ``None`` to disable cluster/cell-set views and color encoding.
    cluster_key_display_name
        Display label for the cluster annotation in Vitessce.
    embedding_key
        Optional key under table ``obsm`` used for embedding coordinates.
        Set to ``None`` to disable the UMAP scatterplot view.
    embedding_key_display_name
        Display label for the embedding in Vitessce and scatterplot mapping.

    Returns
    -------
    VitessceConfig
        A configured Vitessce config with image, optional segmentation, and
        optional table-driven views (feature list, heatmap, cell sets, UMAP).

    Raises
    ------
    ValueError
        If ``cluster_key`` is provided as an empty string.
        If ``cluster_key_display_name`` is empty when ``cluster_key`` is provided.
        If ``embedding_key`` is provided as an empty string.
        If ``embedding_key_display_name`` is empty when ``embedding_key`` is provided.
        If ``center`` is provided but is not a 2-item tuple.
        If ``layer_opacity`` is outside ``[0, 1]``.
        If ``img_layer`` is missing.
        If table-driven visualization is requested but ``table_layer`` is missing.
        If table-driven visualization is requested but ``labels_layer`` is missing.
        If ``sdata_path`` is invalid (empty or unsupported URL format).
    """
    modes = _compute_modes(
        visualize_feature_matrix=visualize_feature_matrix,
        visualize_heatmap=visualize_heatmap,
        cluster_key=cluster_key,
        embedding_key=embedding_key,
    )

    _validate_annotation_keys(
        cluster_key=cluster_key,
        cluster_key_display_name=cluster_key_display_name,
        embedding_key=embedding_key,
        embedding_key_display_name=embedding_key_display_name,
    )
    _validate_camera(center=center, zoom=zoom)
    _validate_layer_opacity(layer_opacity)

    if img_layer is None:
        raise ValueError("img_layer is required when sdata is provided.")
    if modes.needs_adata and table_layer is None:
        raise ValueError(
            "table_layer is required when visualize_feature_matrix=True, "
            "visualize_heatmap=True or cluster_key/embedding_key is provided."
        )
    if modes.needs_adata and labels_layer is None:
        raise ValueError(
            "labels_layer is required when visualize_feature_matrix=True, "
            "visualize_heatmap=True or cluster_key/embedding_key is provided."
        )
    if table_layer is not None and not modes.needs_adata:
        logger.warning(
            "table_layer was provided, but visualize_feature_matrix=False, "
            "visualize_heatmap=False and cluster_key/embedding_key are None; "
            "table layer is ignored."
        )

    normalized_sdata_path, is_sdata_remote = _normalize_path_or_url(
        sdata_path,
        "sdata_path",
    )
    vc = VitessceConfig(
        name = name,
        schema_version=schema_version,
        description=description,
        base_dir=(
            None
            if is_sdata_remote
            else (str(base_dir) if base_dir is not None else None)
        ),
    )

    dataset, file_uuid, labels_file_uuid = _add_spatialdata_wrapper(
        vc,
        name=name,
        sdata_path=normalized_sdata_path,
        is_sdata_remote=is_sdata_remote,
        img_layer=img_layer,
        labels_layer=labels_layer,
        table_layer=table_layer,
        to_coordinate_system=to_coordinate_system,
        modes=modes,
        cluster_key=cluster_key,
        cluster_key_display_name=cluster_key_display_name,
        embedding_key=embedding_key,
        embedding_key_display_name=embedding_key_display_name,
    )

    _build_shared_visualization(
        vc,
        dataset=dataset,
        file_uuid=file_uuid,
        labels_file_uuid=labels_file_uuid,
        modes=modes,
        embedding_key_display_name=embedding_key_display_name,
        center=center,
        zoom=zoom,
        channels=channels,
        palette=palette,
        layer_opacity=layer_opacity,
    )
    return vc


def proteomics_from_split_sources(
    img_source: str | Path,
    labels_source: str | Path | None = None,
    adata_source: str | Path | None = None,
    base_dir: str | Path | None = None,
    name: str = "Proteomics",
    description: str = "Proteomics",
    schema_version: str = "1.0.18",
    center: tuple[float, float] | None = None,
    zoom: float | None = -4,
    channels: Sequence[int | str] | None = None,
    palette: Sequence[str] | None = None,
    layer_opacity: float = 1.0,
    microns_per_pixel_image: float | tuple[float, float] | None = None,
    coordinate_transformations_image: Sequence[Mapping[str, object]] | None = None,
    microns_per_pixel_mask: float | tuple[float, float] | None = None,
    coordinate_transformations_mask: Sequence[Mapping[str, object]] | None = None,
    visualize_feature_matrix: bool = False,
    visualize_heatmap: bool = False,
    cluster_key: str | None = None,
    cluster_key_display_name: str = "Clusters",
    embedding_key: str | None = None,
    embedding_key_display_name: str = "UMAP",
) -> VitessceConfig:
    """
    Build a Vitessce configuration for proteomics image/segmentation visualization
    from explicit image/labels/AnnData sources.

    Parameters
    ----------
    img_source
        Path/URL to an OME-Zarr image.
        Expected axes order is ``(c, y, x)``.
    labels_source
        Path/URL to an OME-Zarr labels segmentation.
        Required when table-driven visualizations are enabled.
        Expected axes order is ``(y, x)``.
    adata_source
        Path/URL to an AnnData ``.zarr``/``.h5ad`` source.
        Required when feature matrix, clusters, or embedding visualizations are
        enabled.
        In this mode, ``obs`` indices should match segmentation label IDs.
        Note that Vitessce currently applies JavaScript ``parseInt`` to
        observation indices. So indices like ``"<ID>_..."`` (for example
        ``"123_cellA"``) still link to label ID ``123`` in ``labels_source``.
    base_dir
        Optional base directory for local paths in the config.
        Ignored when all sources are remote URLs.
    name
        Dataset name shown in Vitessce.
    description
        Configuration description.
    schema_version
        Vitessce schema version.
    center
        Initial camera target as ``(x, y)``.
    zoom
        Initial camera zoom level.
    channels
        Initial channels rendered by ``spatialBeta``.
        Entries can be integer channel indices or channel names.
    palette
        Optional list of channel colors in ``\"#RRGGBB\"`` format.
    layer_opacity
        Opacity of the image layer in ``[0, 1]``.
    microns_per_pixel_image
        Convenience scale transform on image ``(y, x)`` axes.
        Mutually exclusive with ``coordinate_transformations_image``.
    coordinate_transformations_image
        Raw OME-NGFF transforms for the image wrapper.
        Mutually exclusive with ``microns_per_pixel_image``.
    microns_per_pixel_mask
        Convenience scale transform on segmentation ``(y, x)`` axes.
        Mutually exclusive with ``coordinate_transformations_mask``.
    coordinate_transformations_mask
        Raw OME-NGFF transforms for the segmentation wrapper.
        Mutually exclusive with ``microns_per_pixel_mask``.
    visualize_feature_matrix
        If ``True``, expose AnnData ``X`` as marker intensities.
    visualize_heatmap
        If ``True``, expose a heatmap view driven by AnnData ``X``.
    cluster_key
        Optional key under AnnData ``obs`` for cell-set annotations.
    cluster_key_display_name
        Display label for ``cluster_key`` in the UI.
    embedding_key
        Optional key under AnnData ``obsm`` for embedding coordinates.
    embedding_key_display_name
        Display label for the embedding in the UI and mapping name.

    Returns
    -------
    VitessceConfig
        A configured Vitessce config with image, optional segmentation, and
        optional AnnData-driven views (feature list, heatmap, cell sets, UMAP).

    Raises
    ------
    ValueError
        If ``cluster_key``/``embedding_key`` is provided as an empty string.
        If ``cluster_key_display_name``/``embedding_key_display_name`` is empty
        while its corresponding key is provided.
        If ``center`` is not length 2.
        If ``layer_opacity`` is outside ``[0, 1]``.
        If table-driven visualization is requested but ``adata_source`` is
        missing.
        If table-driven visualization is requested but ``labels_source`` is
        missing.
        If a path/URL argument is invalid (empty or unsupported URL format).
        If transformation arguments are inconsistent or invalid (for example
        both transformation styles provided or non-positive scale values).
    """
    modes = _compute_modes(
        visualize_feature_matrix=visualize_feature_matrix,
        visualize_heatmap=visualize_heatmap,
        cluster_key=cluster_key,
        embedding_key=embedding_key,
    )

    _validate_annotation_keys(
        cluster_key=cluster_key,
        cluster_key_display_name=cluster_key_display_name,
        embedding_key=embedding_key,
        embedding_key_display_name=embedding_key_display_name,
    )
    _validate_camera(center=center, zoom=zoom)
    _validate_layer_opacity(layer_opacity)

    if not modes.needs_adata and adata_source is not None:
        logger.warning(
            "adata_source was provided, but visualize_feature_matrix=False, "
            "visualize_heatmap=False and cluster_key/embedding_key are None; "
            "AnnData is ignored."
        )
        adata_source = None

    normalized_img_source, is_img_remote = _normalize_path_or_url(
        img_source, "img_source"
    )
    if labels_source is not None:
        normalized_labels_source, is_labels_remote = _normalize_path_or_url(
            labels_source,
            "labels_source",
        )
    else:
        normalized_labels_source = None
        is_labels_remote = False

    if adata_source is not None:
        normalized_adata_source, is_adata_remote = _normalize_path_or_url(
            adata_source,
            "adata_source",
        )
    else:
        normalized_adata_source = None
        is_adata_remote = False

    if modes.needs_adata and normalized_adata_source is None:
        raise ValueError(
            "adata_source is required when visualize_feature_matrix=True, "
            "visualize_heatmap=True or cluster_key/embedding_key is provided."
        )
    if modes.needs_adata and normalized_labels_source is None:
        raise ValueError(
            "labels_source is required when visualize_feature_matrix=True, "
            "visualize_heatmap=True or cluster_key/embedding_key is provided."
        )

    resolved_image_transforms = _resolve_image_coordinate_transformations(
        coordinate_transformations=coordinate_transformations_image,
        microns_per_pixel=microns_per_pixel_image,
        axes=("c", "y", "x"),
    )
    resolved_mask_transforms = _resolve_image_coordinate_transformations(
        coordinate_transformations=coordinate_transformations_mask,
        microns_per_pixel=microns_per_pixel_mask,
        axes=("y", "x"),
    )

    all_sources_remote = (
        is_img_remote
        and (normalized_labels_source is None or is_labels_remote)
        and (normalized_adata_source is None or is_adata_remote)
    )
    vc = VitessceConfig(
        schema_version=schema_version,
        description=description,
        base_dir=(
            None
            if all_sources_remote
            else (str(base_dir) if base_dir is not None else None)
        ),
    )

    dataset, file_uuid, labels_file_uuid = _add_raw_wrappers(
        vc,
        name=name,
        img_source=normalized_img_source,
        labels_source=normalized_labels_source,
        adata_source=normalized_adata_source,
        is_img_remote=is_img_remote,
        is_labels_remote=is_labels_remote,
        is_adata_remote=is_adata_remote,
        coordinate_transformations_image=resolved_image_transforms,
        coordinate_transformations_mask=resolved_mask_transforms,
        modes=modes,
        cluster_key=cluster_key,
        cluster_key_display_name=cluster_key_display_name,
        embedding_key=embedding_key,
        embedding_key_display_name=embedding_key_display_name,
    )

    _build_shared_visualization(
        vc,
        dataset=dataset,
        file_uuid=file_uuid,
        labels_file_uuid=labels_file_uuid,
        modes=modes,
        embedding_key_display_name=embedding_key_display_name,
        center=center,
        zoom=zoom,
        channels=channels,
        palette=palette,
        layer_opacity=layer_opacity,
    )
    return vc
