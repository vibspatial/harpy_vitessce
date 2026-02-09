import anndata as ad
import pandas as pd
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY

from loguru import logger
import numpy as np

import scanpy as sc


def copy_annotations(
    src: ad.AnnData,  # typically the adata containing the preprocessed counts
    tgt: ad.AnnData,  # typically the adata containing the raw counts
    obs_keys=None,
    obsm_keys=None,
    instance_key: str = _INSTANCE_KEY,
    region_key: str = _REGION_KEY,
) -> ad.AnnData:  # returns a copy of adata. Note that instances in tgt that are not in src will be removed.
    """
    Align two `AnnData` objects by `(instance_key, region_key)` and copy annotations.

    The function matches observations between `src` and `tgt` using the pair of
    columns `[instance_key, region_key]` in `.obs`, keeps only the intersection
    of keys, reorders `tgt` to match `src`, and then copies selected annotation
    fields via :func:`_copy_annotations`.

    For copied categorical `.obs` columns, category definitions are preserved and
    matching color metadata from `src.uns[f"{obs_key}_colors"]` is copied into
    `tgt.uns` when available.

    Parameters
    ----------
    src
        Source `AnnData` from which annotations are copied. Typically the `AnnData` containing processed counts.
    tgt
        Target `AnnData` to receive copied annotations. Typically the `AnnData` containing the unprocessed counts.
    obs_keys
        Iterable of `.obs` column names to copy from `src` to `tgt`.
    obsm_keys
        Iterable of `.obsm` keys to copy from `src` to `tgt`.
    instance_key
        Column name in `.obs` used as instance identifier.
    region_key
        Column name in `.obs` used as region identifier.

    Returns
    -------
    ad.AnnData
        A copy of `tgt` filtered to observations present in both objects and
        ordered like `src` (after intersection), with requested `.obs` and
        `.obsm` annotations copied from `src`. If a copied `.obs` field is
        categorical and has corresponding colors in `src.uns`, those colors are
        copied as well.
    """
    key_cols = [instance_key, region_key]

    # 1. Build MultiIndex of keys for both objects
    idx_adata = pd.MultiIndex.from_frame(tgt.obs[key_cols])
    idx_adata_proc = pd.MultiIndex.from_frame(src.obs[key_cols])

    # 2. Map key -> obs_name for adata
    key_to_obs = pd.Series(tgt.obs_names, index=idx_adata)

    # 3. Keep only keys that are in adata (drops extra rows in adata_processed)
    common_keys = idx_adata_proc.intersection(key_to_obs.index)

    # Reorder adata_processed to only those common keys, preserving its (now filtered) order
    mask_proc = idx_adata_proc.isin(common_keys)
    adata_processed_sub = src[mask_proc].copy()
    idx_adata_proc_sub = idx_adata_proc[mask_proc]

    # 4. Get obs_names for adata in the same order as adata_processed_sub
    obs_order = key_to_obs.loc[idx_adata_proc_sub].values

    # Subset & reorder adata accordingly
    adata_sub = tgt[obs_order].copy()

    adata_sub = _copy_annotations(
        src=adata_processed_sub,
        tgt=adata_sub,
        obs_keys=obs_keys,
        obsm_keys=obsm_keys,
    )

    return adata_sub


def _copy_annotations(
    src: ad.AnnData,
    tgt: ad.AnnData,
    obs_keys=None,
    obsm_keys=None,
) -> ad.AnnData:
    """
    Copy selected .obs columns and .obsm matrices from `src` to `tgt`.

    Assumes `src.n_obs == tgt.n_obs` and rows are in the same order
    (as you've already aligned via cell_ID + fov_labels).
    """
    if src.n_obs != tgt.n_obs:
        raise ValueError(
            "Source and target AnnData must have the same number of observations: "
            f"src.n_obs={src.n_obs}, tgt.n_obs={tgt.n_obs}."
        )
    if not src.obs_names.equals(tgt.obs_names):
        raise ValueError(
            "Source and target AnnData observations are not in the same order "
            "(or have different obs_names)."
        )

    if obs_keys is None:
        obs_keys = []
    if obsm_keys is None:
        obsm_keys = []

    # Copy .obs columns
    for key in obs_keys:
        if key not in src.obs:
            raise KeyError(f"{key!r} not found in src.obs")
        src_col = src.obs[key]

        # Keep categorical dtype and category order stable while avoiding index alignment.
        if pd.api.types.is_categorical_dtype(src_col.dtype):
            tgt.obs[key] = pd.Categorical.from_codes(
                src_col.cat.codes.to_numpy(),
                categories=src_col.cat.categories,
                ordered=src_col.cat.ordered,
            )
            color_key = f"{key}_colors"
            if color_key in src.uns:
                tgt.uns[color_key] = src.uns[color_key].copy()
        else:
            # use .to_numpy() to avoid index alignment shenanigans
            tgt.obs[key] = src_col.to_numpy()

    # Copy .obsm matrices
    for key in obsm_keys:
        if key not in src.obsm:
            raise KeyError(f"{key!r} not found in src.obsm")
        # copy to avoid shared views
        tgt.obsm[key] = src.obsm[key].copy()

    return tgt


def example_visium_hd_processing(
    adata: ad.AnnData,
    min_counts: int = 1000,
    min_cells: int = 10,
    target_sum: float = 1e4,
    hvg_flavor: str = "seurat",
    n_top_genes: int = 3000,
    scale_max_value: float = 10,
    pca_n_comps: int = 50,
    pca_svd_solver: str = "arpack",
    n_neighbors: int = 15,
    n_pcs: int = 30,
    leiden_resolution: float = 0.8,
    leiden_key_added: str = "leiden",
    spatial_spot_size: float = 20,
) -> ad.AnnData:
    """
    Run an example Scanpy preprocessing and clustering workflow for Visium HD data.

    The pipeline performs quality control, filtering, normalization, log
    transformation, highly variable gene selection, scaling, PCA, neighbor graph
    construction, UMAP embedding, and Leiden clustering. It also generates UMAP
    and spatial plots colored by Leiden clusters.

    Parameters
    ----------
    adata
        Input `AnnData` containing Visium HD counts and spatial metadata.
        `adata.X` is expected to contain raw count values (not normalized or
        log-transformed), since this workflow performs normalization and log1p.
        The input object is updated in place during the full workflow, including
        filtering and highly-variable-gene subsetting.
    min_counts
        Minimum total counts required for a cell/spot to be retained.
    min_cells
        Minimum number of cells/spots in which a gene must be detected.
    target_sum
        Target total counts per cell/spot for library-size normalization.
    hvg_flavor
        Method used by `scanpy.pp.highly_variable_genes`.
    n_top_genes
        Number of highly variable genes to select.
    scale_max_value
        Maximum absolute value used in `scanpy.pp.scale`.
    pca_n_comps
        Number of principal components to compute.
    pca_svd_solver
        SVD solver passed to `scanpy.pp.pca`.
    n_neighbors
        Number of neighbors used to construct the neighborhood graph.
    n_pcs
        Number of PCs used for neighborhood graph construction.
    leiden_resolution
        Resolution parameter for Leiden clustering.
    leiden_key_added
        Column name in `adata.obs` where Leiden labels are stored.
    spatial_spot_size
        Spot size used for the spatial plot.

    Returns
    -------
    ad.AnnData
        The same input object, processed and subset to highly variable genes,
        with results stored in standard Scanpy slots (including PCA, neighbors,
        UMAP, and `adata.obs["leiden"]`).

    Notes
    -----
    - The function updates global Scanpy settings (`sc.settings.verbosity`,
      `sc.set_figure_params`) and emits log messages via `loguru`.
    - Plotting calls (`sc.pl.umap`, `sc.pl.spatial`) have side effects and may
      display figures depending on the active plotting backend.
    """
    # takes around 2 minutes
    logger.info("Starting Scanpy pipeline for Visium HD data.")

    # reasonable verbosity and figure sizes
    sc.settings.verbosity = 3
    sc.set_figure_params(figsize=(4, 4))

    # 1. Basic QC metrics
    logger.info("Calculating QC metrics.")
    sc.pp.calculate_qc_metrics(
        adata,
        # qc_vars=None,     # or ["MT"] if mito genes are flagged in adata.var
        inplace=True,
    )

    # 2. Filtering spots and genes
    logger.info(f"Filtering cells with min_counts={min_counts}.")
    sc.pp.filter_cells(adata, min_counts=min_counts)

    logger.info(f"Filtering genes with min_cells={min_cells}.")
    sc.pp.filter_genes(adata, min_cells=min_cells)

    logger.info(f"Shape after filtering: {adata.n_obs} cells × {adata.n_vars} genes.")

    # 3. Normalization & log transform
    logger.info(f"Normalizing total counts to {target_sum} per cell.")
    sc.pp.normalize_total(adata, target_sum=target_sum)

    logger.info("Applying log1p transform.")
    sc.pp.log1p(adata)

    # 4. Highly variable genes
    logger.info(
        f"Selecting highly variable genes with flavor='{hvg_flavor}', "
        f"n_top_genes={n_top_genes}."
    )
    sc.pp.highly_variable_genes(adata, flavor=hvg_flavor, n_top_genes=n_top_genes)

    hvg_count = adata.var["highly_variable"].sum()
    logger.info(f"Number of HVGs selected: {hvg_count}.")

    logger.info("Subsetting AnnData to HVGs in place.")
    adata._inplace_subset_var(adata.var["highly_variable"].to_numpy())
    logger.info(f"Shape after HVG subset: {adata.n_obs} cells × {adata.n_vars} genes.")

    # 5. Scaling
    logger.info(f"Scaling data with max_value={scale_max_value}.")
    sc.pp.scale(adata, max_value=scale_max_value)

    # 6. PCA
    logger.info(
        f"Running PCA with n_comps={pca_n_comps}, svd_solver='{pca_svd_solver}'."
    )
    sc.pp.pca(adata, svd_solver=pca_svd_solver, n_comps=pca_n_comps)

    # 7. Neighborhood graph
    logger.info(
        f"Computing neighborhood graph (n_neighbors={n_neighbors}, n_pcs={n_pcs})."
    )
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    # 8. UMAP embedding
    logger.info("Computing UMAP embedding.")
    sc.tl.umap(adata)

    # 9. Leiden clustering
    logger.info(
        f"Running Leiden clustering with resolution={leiden_resolution}, "
        f"key_added='{leiden_key_added}'."
    )
    sc.tl.leiden(adata, resolution=leiden_resolution, key_added=leiden_key_added)

    logger.info("Leiden clustering finished.")
    logger.info(f"Number of Leiden clusters: {adata.obs[leiden_key_added].nunique()}")

    # 10. Basic visualizations
    logger.info("Plotting UMAP colored by Leiden clusters.")
    sc.pl.umap(adata, color=[leiden_key_added])

    # logger.info("Plotting spatial map colored by Leiden clusters.")
    sc.pl.spatial(adata, color=[leiden_key_added], spot_size=spatial_spot_size)

    logger.info("Scanpy preprocessing and clustering pipeline completed.")

    return adata


def downcast_int64_to_int32(
    df: pd.DataFrame, strict: bool = True
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """
    Downcast `int64` columns to `int32` when values fit in `int32`.

    Parameters
    ----------
    df
        Input DataFrame.
    strict
        If `True`, raise on overflow. If `False`, keep overflowing columns unchanged.

    Returns
    -------
    tuple[pd.DataFrame, list[str], dict[str, str]]
        Converted DataFrame copy, converted column names, and skipped-column reasons.

    Raises
    ------
    OverflowError
        If `strict=True` and a column is outside `int32` range.
    """
    out = df.copy()
    i32 = np.iinfo(np.int32)

    int64_cols = out.select_dtypes(include=["int64"]).columns.tolist()
    converted_cols: list[str] = []
    skipped_cols: dict[str, str] = {}

    for col in int64_cols:
        s = out[col]
        cmin = s.min(skipna=True)
        cmax = s.max(skipna=True)

        if cmin >= i32.min and cmax <= i32.max:
            out[col] = s.astype(np.int32)
            converted_cols.append(col)
        else:
            msg = (
                f"out of int32 range: min={cmin}, max={cmax}, "
                f"allowed=[{i32.min}, {i32.max}]"
            )
            skipped_cols[col] = msg
            if strict:
                raise OverflowError(f"Column '{col}' cannot be safely cast: {msg}")

    return out, converted_cols, skipped_cols
