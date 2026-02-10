import anndata as ad
import numpy as np
import pandas as pd
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY


def copy_annotations(
    src: ad.AnnData,  # typically the adata containing the preprocessed counts
    tgt: ad.AnnData,  # typically the adata containing the raw counts
    obs_keys=None,
    obsm_keys=None,
    instance_key: str = _INSTANCE_KEY,
    region_key: str = _REGION_KEY,
) -> ad.AnnData:  # returns a copy of adata. Note that instances in tgt that are not in src will be removed. tes
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
