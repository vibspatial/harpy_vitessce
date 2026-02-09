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

from pathlib import Path


def visium_hd(
    path_img: str | Path,  # relative to BASE_DIR
    path_adata: str | Path,  # relative to BASE_DIR
    name: str = "Benchmark",
    description: str = "Visium HD",
    schema_version: str = "1.0.18",
    BASE_DIR: str | Path | None = None,
    center_x: float | None = None,
    center_y: float | None = None,
    zoom: float | None = -4,  # e.g. -4
):
    # default to BASE_DIR "/" if None?. Do not set to None by default?
    spot_size_micron = 16
    umap_radius = 3

    vc = VitessceConfig(
        schema_version=schema_version,
        name=name,
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

    dataset = vc.add_dataset(name="Liver dataset").add_object(
        ImageOmeZarrWrapper(
            img_path=path_img,
            coordination_values={"fileUid": "img1"},
        )
    )

    dataset.add_object(
        AnnDataWrapper(
            adata_path=path_adata,
            obs_feature_matrix_path="X",
            obs_spots_path="obsm/spatial",
            obs_set_paths=["obs/leiden"],
            obs_set_names=["Leiden"],  # display name in UI
            obs_embedding_paths=["obsm/X_umap"],
            obs_embedding_names=["UMAP"],
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
    umap = vc.add_view(cm.SCATTERPLOT, dataset=dataset, mapping="UMAP")
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
    # feat_sel.set_value([str(adata.var_names[0])]) # initialize with an existing gene
    obs_set_sel.set_value(None)

    emb_radius_mode, emb_radius = vc.add_coordination(
        ct.EMBEDDING_OBS_RADIUS_MODE,
        ct.EMBEDDING_OBS_RADIUS,
    )
    emb_radius_mode.set_value("auto")  # or "auto"
    emb_radius.set_value(umap_radius)

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
        emb_radius_mode,
        emb_radius,
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
                        "fileUid": "img1",
                        "spatialLayerVisible": True,
                        "spatialLayerOpacity": 1.0,
                        "photometricInterpretation": "RGB",
                    }
                ]
            ),
            "spotLayer": CL(
                [
                    {
                        "obsType": obs_type,  # or "spot"
                        "spatialLayerVisible": True,
                        "spatialLayerOpacity": 1.0,
                        "spatialSpotRadius": spot_size_micron // 2,
                        "spatialSpotFilled": True,
                        "spatialSpotStrokeWidth": 1.0,
                        "spatialLayerColor": [255, 255, 255],
                        # NOTE: no obsColorEncoding / featureSelection here!
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
