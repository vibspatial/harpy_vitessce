import anndata as ad
import scanpy as sc
from loguru import logger


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

    This pipeline is merely provided as an example to generate the fields required for
    visualization in Vitessce.

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
