import argparse
from pathlib import Path

import anndata as ad
import harpy as hp
from harpy.image._image import _get_translation_values
from harpy.utils._keys import _INSTANCE_KEY
from loguru import logger
from spatialdata import read_zarr
from spatialdata.models import TableModel
from spatialdata.transformations import get_transformation

from harpy_vitessce.data_utils import (
    copy_annotations,
    downcast_int64_to_int32,
    normalize_array,
    xarray_to_ome_zarr,
)


MT_OBS_KEYS = {"total_counts_mt", "pct_counts_mt"}


def _none_or_str(value: str | None) -> str | None:
    if value is None:
        return None
    if value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert selected SpatialData content to Zarr v2 for Vitessce."
    )
    parser.add_argument(
        "--resolution",
        required=True,
        help="Grid resolution in microns (e.g. 20 for hexagonal_grid_20um_table).",
    )
    parser.add_argument(
        "--sdata-path",
        required=True,
        help="Path to input SpatialData zarr store.",
    )
    parser.add_argument(
        "--output-path-adata",
        required=True,
        help="Explicit output path for adata zarr (e.g. /path/to/adata.zarr).",
    )
    parser.add_argument(
        "--output-path-img",
        required=True,
        help="Explicit output path for image OME-Zarr (e.g. /path/to/image.ome.zarr).",
    )
    parser.add_argument(
        "--image-layer",
        type=_none_or_str,
        default=None,
        help=(
            "Optional image layer key in sdata (e.g. SCA001_full_image). "
            "Pass None to skip image export."
        ),
    )
    parser.add_argument(
        "--microns-per-pixel",
        required=True,
        type=float,
        help="Microns-per-pixel value used to scale coordinates and image metadata.",
    )
    parser.add_argument(
        "--to-copy-annotations",
        action="store_true",
        help="Copy annotations from processed hexagonal-grid AnnData into raw-count AnnData.",
    )
    parser.add_argument(
        "--exclude_mt",
        action="store_true",
        help="Exclude mitochondrial QC obs keys from the converted AnnData.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    to_copy_annotations = args.to_copy_annotations

    ad.settings.zarr_write_format = 2

    sdata = read_zarr(args.sdata_path)

    output_path_adata = Path(args.output_path_adata)
    output_path_img = Path(args.output_path_img)

    adata_exists = output_path_adata.exists()
    img_exists = output_path_img.exists()

    if adata_exists:
        logger.warning("Output exists, refusing to overwrite: {}", output_path_adata)
    if img_exists:
        logger.warning("Output exists, refusing to overwrite: {}", output_path_img)

    if adata_exists and (img_exists or args.image_layer is None):
        logger.warning("Required outputs already exist. Nothing to write.")
        return 0

    if not adata_exists:
        # Copy annotations from processed adata into raw-count adata for visualization.

        if to_copy_annotations:
            # custom bins, copy some data from one anndata to the other
            obs_keys = [
                "leiden_1",
                "total_counts",
                "n_genes_by_counts",
                "total_counts_mt",
                "pct_counts_mt",
                "pct_counts_in_top_50_genes",
            ]
            if args.exclude_mt:
                obs_keys = [key for key in obs_keys if key not in MT_OBS_KEYS]

            adata_annotated = copy_annotations(
                src=sdata[f"hexagonal_grid_{args.resolution}um_table_processed"],
                tgt=sdata[f"hexagonal_grid_{args.resolution}um_table"],
                obs_keys=obs_keys,
                obsm_keys=["X_umap"],
            )
        else:
            # 2,8,16 bins
            obs_keys = [
                "total_counts",
                "n_genes_by_counts",
                "total_counts_mt",
                "pct_counts_mt",
                "pct_counts_in_top_50_genes",
            ]
            if args.exclude_mt:
                obs_keys = [key for key in obs_keys if key not in MT_OBS_KEYS]

            adata_annotated = sdata[f"square_0{args.resolution}um"]
            adata_annotated = adata_annotated[
                adata_annotated.obs["in_tissue_annotation"] == 1
            ].copy()

        # Vitessce cannot handle int64 in obs metadata.
        obs_keys.append(_INSTANCE_KEY)
        df_obs, _, _ = downcast_int64_to_int32(
            adata_annotated.obs[obs_keys], strict=True
        )
        adata_annotated.obs = df_obs

        # Ensure suitable sparse format and numeric dtype.
        adata_annotated.X = normalize_array(adata_annotated.X)

        if to_copy_annotations:
            # calculate .obsm "spatial" if we set _to_copy_annotations (because if we set to_copy_annotations, it was generated
            # using a custom hexagon grid, and the harpy.tb.bin_counts, does not set center of these grids in .obsm["spatial"],
            # but the average coordinate of the bins that where aggregated.
            region = adata_annotated.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]
            assert len(region) == 1
            region = region[0]

            logger.info("Start calculating center of mass.")
            aggregator = hp.utils.RasterAggregator(
                mask_dask_array=hp.im.get_dataarray(sdata, layer=region).data[None, ...]
            )
            df = aggregator.center_of_mass()
            logger.info("End calculating center of mass.")

            centers = df[[_INSTANCE_KEY, 2, 1]]  # make it x,y

            x_trans, y_trans = (0.0, 0.0)
            if args.image_layer is not None:
                coordinate_systems = list(
                    get_transformation(sdata[args.image_layer], get_all=True).keys()
                )
                if len(coordinate_systems) != 1:
                    raise ValueError(
                        f"Expected exactly one coordinate system for image layer "
                        f"{args.image_layer!r}, found {len(coordinate_systems)}: "
                        f"{coordinate_systems}"
                    )

                to_coordinate_system = coordinate_systems[0]
                x_trans, y_trans = _get_translation_values(
                    get_transformation(
                        sdata[region],
                        to_coordinate_system=to_coordinate_system,
                    )
                )
            else:
                logger.info(
                    "No image layer provided; skipping image-based spatial translation."
                )

            # Align centers to adata.obs['cell_ID']; fail fast if any IDs are missing.
            centers_by_id = centers.set_index(_INSTANCE_KEY).copy()
            centers_by_id.index = centers_by_id.index.astype(str)
            obs_cell_ids = adata_annotated.obs[_INSTANCE_KEY].astype(str)

            matched_mask = obs_cell_ids.isin(centers_by_id.index).to_numpy()
            if not matched_mask.all():
                missing_ids = obs_cell_ids[~matched_mask].unique().tolist()
                preview = missing_ids[:10]
                suffix = "" if len(missing_ids) <= 10 else " ..."
                raise ValueError(
                    f"Could not match {len(missing_ids)} cell_ID values from .obs "
                    f"to centers: {preview}{suffix}"
                )

            adata_annotated.obsm["spatial"] = centers_by_id.reindex(
                obs_cell_ids
            ).to_numpy()
            logger.info(
                f"Stored spatial coordinates for {adata_annotated.n_obs} / {adata_annotated.n_obs} cells."
            )

            adata_annotated.obsm["spatial"] = adata_annotated.obsm["spatial"] + (
                x_trans,
                y_trans,
            )

        adata_annotated.obsm["spatial"] = (
            adata_annotated.obsm["spatial"] * args.microns_per_pixel
        )

        logger.info("Writing adata to {}", output_path_adata)
        adata_annotated.write_zarr(output_path_adata)

    if args.image_layer is not None and not img_exists:
        img_da = hp.im.get_dataarray(sdata, layer=args.image_layer)
        n_channels = int(img_da.sizes["c"]) if "c" in img_da.dims else 1

        if n_channels == 3:
            channel_names = ["r", "g", "b"]
            channel_colors = {"r": "FF0000", "g": "00FF00", "b": "0000FF"}
        elif n_channels == 1:
            channel_names = ["DAPI"]
            channel_colors = None
        else:
            channel_names = [f"ch{i + 1}" for i in range(n_channels)]
            channel_colors = None

        logger.info("Writing image to {}", output_path_img)
        xarray_to_ome_zarr(
            tree_or_da=sdata[args.image_layer],
            channel_names=channel_names,
            output_path=output_path_img,
            channel_colors=channel_colors,
            microns_per_pixel=args.microns_per_pixel,
            coords_in_microns=False,
            zarr_format=2,
        )
    elif args.image_layer is None:
        logger.info("No image layer provided; skipping image export.")

    logger.info("Conversion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
