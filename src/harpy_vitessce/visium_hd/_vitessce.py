import uuid
from pathlib import Path
from typing import Literal, Sequence
from urllib.parse import urlparse

from vitessce import (
    AnnDataWrapper,
    ImageOmeZarrWrapper,
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

# Vitessce component identifiers used by this config.
SPATIAL_VIEW = "spatialBeta"
LAYER_CONTROLLER_VIEW = "layerControllerBeta"
OBS_TYPE_SPOT = "spot"
OBS_COLOR_CELL_SET_SELECTION = "cellSetSelection"
OBS_COLOR_GENE_SELECTION = "geneSelection"

# Vitessce coordination defaults used by this config (i.e. can be changed).
FEATURE_TYPE_GENE = "gene"
FEATURE_VALUE_TYPE_EXPRESSION = "expression"
FEATURE_TYPE_QC = "qc"
FEATURE_VALUE_TYPE_QC = "qc_value"

def _is_remote_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _normalize_path_or_url(path: str | Path, name: str) -> tuple[str, bool]:
    path_str = str(path)
    if not path_str:
        raise ValueError(f"{name} must be a non-empty path or URL.")
    parsed = urlparse(path_str)
    if parsed.scheme and not _is_remote_url(path_str):
        raise ValueError(
            f"{name} URL must start with http:// or https:// and include a host."
        )
    return path_str, _is_remote_url(path_str)


def visium_hd(
    path_img: str | Path,  # local path relative to base_dir or remote URL
    path_adata: str | Path,  # local path relative to base_dir or remote URL
    name: str = "Visium HD",
    description: str = "Visium HD",
    schema_version: str = "1.0.18",
    base_dir: str | Path | None = None,
    center: tuple[float, float] | None = None,
    zoom: float | None = -4,  # e.g. -4
    spot_size_micron: int = 16,
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

    Parameters
    ----------
    path_img
        Path/URL to the OME-Zarr image. Local paths are relative to ``base_dir``
        when provided.
        You can generate this image with
        :func:`harpy_vitessce.data_utils.xarray_to_ome_zarr` or
        :func:`harpy_vitessce.data_utils.array_to_ome_zarr`.
    path_adata
        Path/URL to the AnnData ``.zarr``/``.h5ad`` source. Local paths are
        relative to ``base_dir`` when provided.
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
        Remote URLs are used as-is. When both ``path_img`` and ``path_adata``
        are remote URLs, ``base_dir`` is ignored and set to ``None`` in the
        generated Vitessce config.
    center
        Initial spatial target as ``(x, y)`` camera center coordinates.
        Use ``None`` to keep Vitessce defaults.
    zoom
        Initial spatial zoom level. Use ``None`` to keep Vitessce defaults.
    spot_size_micron
        Spot diameter in microns; rendered radius is ``spot_size_micron // 2``.
    spatial_key
        Key under ``obsm`` used for spot coordinates, e.g. ``"spatial"`` -> ``"obsm/spatial"``.
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
        2-item tuple, or ``spot_size_micron <= 0``, or ``emb_radius <= 0`` when
        ``emb_radius_mode="manual"``.

    Examples
    --------
    .. code-block:: python

        from IPython.display import display, HTML

        vc = visium_hd(
            base_dir="/your/path/"
            path_img="data/sample_image.ome.zarr", # relative to base_dir
            path_adata="data/sample_adata.zarr", # relative to base_dir
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
    path_img, is_img_remote = _normalize_path_or_url(path_img, "path_img")
    path_adata, is_adata_remote = _normalize_path_or_url(path_adata, "path_adata")

    has_clusters = cluster_key is not None
    has_embedding = embedding_key is not None
    has_qc = len(qc_obs_feature_keys) > 0

    if has_embedding and emb_radius_mode not in {"auto", "manual"}:
        raise ValueError(
            "emb_radius_mode must be either 'auto' or 'manual'; "
            f"got {emb_radius_mode!r}."
        )
    if spot_size_micron <= 0:
        raise ValueError(
            "spot_size_micron must be > 0 so spatial spot radius is valid."
        )
    if has_embedding and emb_radius_mode == "manual" and emb_radius <= 0:
        raise ValueError("emb_radius must be > 0 when emb_radius_mode='manual'.")
    if center is not None and len(center) != 2:
        raise ValueError("center must be a tuple of two floats: (x, y).")

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
    image_wrapper_kwargs: dict[str, object] = {
        "coordination_values": {"fileUid": _file_uuid},
    }
    image_wrapper_kwargs["img_url" if is_img_remote else "img_path"] = path_img
    dataset = vc.add_dataset(name=name).add_object(
        ImageOmeZarrWrapper(
            **image_wrapper_kwargs,
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
        path_adata
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
            path_adata
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
    #  but then we need two layer controllers, which looks weird in the UI
    linked_views = [spatial_plot, layer_controller]
    if spatial_qc is not None:
        linked_views.append(spatial_qc)

    vc.link_views_by_dict(
        linked_views,
        {
            "imageLayer": CL(
                [
                    {
                        "fileUid": _file_uuid,
                        "spatialLayerVisible": True,
                        "spatialLayerOpacity": 1.0,
                        "photometricInterpretation": "RGB",
                    }
                ]
            ),
            "spotLayer": CL(
                [
                    {
                        "obsType": obs_type,
                        "spatialLayerVisible": True,
                        "spatialLayerOpacity": 1.0,
                        "spatialSpotRadius": spot_size_micron // 2,
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
    layer_controller.set_props(disableChannelsIfRgbDetected=True)

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
