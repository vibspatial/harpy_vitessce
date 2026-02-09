import uuid
from pathlib import Path
from typing import Literal

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


def visium_hd(
    path_img: str | Path,  # relative to BASE_DIR
    path_adata: str | Path,  # relative to BASE_DIR
    name: str = "Visium HD",
    description: str = "Visium HD",
    schema_version: str = "1.0.18",
    BASE_DIR: str | Path | None = None,
    center_x: float | None = None,
    center_y: float | None = None,
    zoom: float | None = -4,  # e.g. -4
    spot_size_micron: int = 16,
    spatial_key: str = "spatial",  # center of the spots. In micron coordinates
    cluster_key: str = "leiden",
    cluster_key_display_name: str = "Leiden",
    embedding_key: str = "X_umap",
    embedding_display_name: str = "UMAP",
    emb_radius_mode: Literal["auto", "manual"] = "auto",
    emb_radius: int = 3,  # ignored if emb_radius_mode is "auto"
):
    """
    Build a Vitessce configuration for exploring Visium HD data.

    Parameters
    ----------
    path_img
        Path to the OME-Zarr image (relative to ``BASE_DIR`` when provided).
        You can generate this image with
        :func:`harpy_vitessce.data_utils.xarray_to_ome_zarr` or
        :func:`harpy_vitessce.data_utils.array_to_ome_zarr`.
    path_adata
        Path to the AnnData ``.zarr``/``.h5ad`` source (relative to ``BASE_DIR`` when provided).
    name
        Dataset name shown in Vitessce.
    description
        Configuration description.
    schema_version
        Vitessce schema version.
    BASE_DIR
        Optional base directory for relative paths in the config.
    center_x
        Initial spatial X target (camera center).
    center_y
        Initial spatial Y target (camera center).
    zoom
        Initial spatial zoom level. Use ``None`` to keep Vitessce defaults.
    spot_size_micron
        Spot diameter in microns; rendered radius is ``spot_size_micron // 2``.
    spatial_key
        Key under ``obsm`` used for spot coordinates, e.g. ``"spatial"`` -> ``"obsm/spatial"``.
    cluster_key
        Key under ``obs`` used for cluster/cell-set annotations, e.g. ``"leiden"`` -> ``"obs/leiden"``.
    cluster_key_display_name
        Display label for the cluster annotation in the Vitessce UI.
    embedding_key
        Key under ``obsm`` used for embedding coordinates, e.g. ``"X_umap"`` -> ``"obsm/X_umap"``.
    embedding_display_name
        Display label for the embedding in the Vitessce UI and scatterplot mapping.
    emb_radius_mode
        Embedding point radius mode. Must be ``"auto"`` or ``"manual"``.
    emb_radius
        Embedding point radius value used by Vitessce. Ignored when
        ``emb_radius_mode="auto"``).

    Returns
    -------
    VitessceConfig
        A configured Vitessce configuration object with spatial, embedding,
        cluster, gene expression, and QC views.

    Raises
    ------
    ValueError
        If ``spatial_key`` is empty, ``cluster_key`` is empty, ``embedding_key`` is
        empty, or ``emb_radius_mode`` is not ``"auto"``/``"manual"``.
    """
    if not spatial_key:
        raise ValueError("spatial_key must be a non-empty string.")
    if not cluster_key:
        raise ValueError("cluster_key must be a non-empty string.")
    if not embedding_key:
        raise ValueError("embedding_key must be a non-empty string.")

    if emb_radius_mode not in {"auto", "manual"}:
        raise ValueError(
            "emb_radius_mode must be either 'auto' or 'manual'; "
            f"got {emb_radius_mode!r}."
        )

    # default to BASE_DIR "/" if None?. Do not set to None by default?
    vc = VitessceConfig(
        schema_version=schema_version,
        description=description,
        base_dir=BASE_DIR,
    )

    spatial_zoom, spatial_target_x, spatial_target_y = vc.add_coordination(
        ct.SPATIAL_ZOOM,
        ct.SPATIAL_TARGET_X,
        ct.SPATIAL_TARGET_Y,
    )

    if zoom is not None:
        spatial_zoom.set_value(zoom)
    if center_x is not None and center_y is not None:
        spatial_target_x.set_value(center_x)
        spatial_target_y.set_value(center_y)

    _file_uuid = f"img_h&e_{uuid.uuid4()}"  # can be set to any value
    dataset = vc.add_dataset(name=name).add_object(
        ImageOmeZarrWrapper(
            img_path=path_img,
            coordination_values={"fileUid": _file_uuid},
        )
    )

    dataset.add_object(
        AnnDataWrapper(
            adata_path=path_adata,
            obs_feature_matrix_path="X",
            obs_spots_path=f"obsm/{spatial_key}",
            obs_set_paths=[f"obs/{cluster_key}"],
            obs_set_names=[cluster_key_display_name],  # display name in UI
            obs_embedding_paths=[f"obsm/{embedding_key}"],
            obs_embedding_names=[embedding_display_name],
            coordination_values={
                "obsType": "spot",
                "featureType": "gene",
                "featureValueType": "expression",
            },
        )
    )

    # QC / obs features
    dataset.add_object(
        AnnDataWrapper(
            adata_path=path_adata,
            obs_feature_matrix_path=None,
            obs_feature_column_paths=[
                "obs/total_counts",
                "obs/n_genes_by_counts",
                # "obs/total_counts_mt",
                # "obs/pct_counts_mt",
            ],
            coordination_values={
                "obsType": "spot",
                "featureType": "qc",
                "featureValueType": "qc_value",
            },
        )
    )

    # 1) views:
    # leiden + genes)
    spatial_plot = vc.add_view("spatialBeta", dataset=dataset)
    spatial_plot.set_props(title="Leiden Clusters + Gene Expression")
    layer_controller = vc.add_view("layerControllerBeta", dataset=dataset)
    genes = vc.add_view(cm.FEATURE_LIST, dataset=dataset)
    cell_sets = vc.add_view(cm.OBS_SETS, dataset=dataset)
    umap = vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping=embedding_display_name)
    # qc
    histogram = vc.add_view(cm.FEATURE_VALUE_HISTOGRAM, dataset=dataset)
    # qc (spatial only; not linked to UMAP)
    spatial_qc = vc.add_view("spatialBeta", dataset=dataset)
    spatial_qc.set_props(title="QC")
    qc_list = vc.add_view(cm.FEATURE_LIST, dataset=dataset)
    qc_list.set_props(title="QC list")

    # coordinate views spatial (Leiden + genes)
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
    obs_type.set_value("spot")
    feat_type.set_value("gene")  # defined in coordination_values when we add addata
    feat_val_type.set_value("expression")
    # Color by Leiden (cell sets), not by gene selection
    obs_color.set_value("cellSetSelection")
    obs_set_sel.set_value(None)

    emb_radius_mode_coord, emb_radius_coord = vc.add_coordination(
        ct.EMBEDDING_OBS_RADIUS_MODE,
        ct.EMBEDDING_OBS_RADIUS,
    )

    emb_radius_mode_coord.set_value(emb_radius_mode)
    emb_radius_coord.set_value(emb_radius)

    # coordinate views spatial (qc)
    obs_color_qc, feat_type_qc, feat_val_type_qc, feat_sel_qc, obs_set_sel_qc = (
        vc.add_coordination(
            ct.OBS_COLOR_ENCODING,
            ct.FEATURE_TYPE,
            ct.FEATURE_VALUE_TYPE,
            ct.FEATURE_SELECTION,
            ct.OBS_SET_SELECTION,
        )
    )
    obs_color_qc.set_value("geneSelection")  # use feature values for QC
    feat_type_qc.set_value("qc")
    feat_val_type_qc.set_value("qc_value")
    feat_sel_qc.set_value(["total_counts"])
    obs_set_sel_qc.set_value(None)

    # use coordination
    # spatial (leiden + genes)
    spatial_plot.use_coordination(
        obs_type, feat_type, feat_val_type, obs_color, feat_sel, obs_set_sel
    )
    spatial_plot.use_coordination(spatial_zoom, spatial_target_x, spatial_target_y)
    genes.use_coordination(obs_type, obs_color, feat_sel, feat_type, feat_val_type)
    cell_sets.use_coordination(obs_type, obs_set_sel, obs_color)
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
    # qc
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
        # obs_set_sel_qc # check if this would work
    )

    vc.link_views_by_dict(
        [spatial_plot, spatial_qc, layer_controller],
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

    vc.layout(
        hconcat(
            # COLUMN 1: spatial_plot, umap
            vconcat(
                spatial_plot,
                umap,
                split=[8, 4],
            ),
            # COLUMN 2: spatial_qc, histogram
            vconcat(
                spatial_qc,
                histogram,
                split=[8, 4],
            ),
            vconcat(
                layer_controller,
                genes,  # gene_list
                qc_list,
                cell_sets,  # spot_sets
                split=[3, 4, 3, 2],  # 3 + 4 + 2 + 3 = 12
            ),
            # Column widths: left wide, middle wide, right narrow
            split=[5, 5, 2],
        )
    )

    return vc
