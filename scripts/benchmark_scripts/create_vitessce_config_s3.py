import argparse
import json
from pathlib import Path

import numpy as np
from anndata import read_zarr
from loguru import logger

import harpy_vitessce as hpv


def _none_or_str(value: str | None) -> str | None:
    if value is None:
        return None
    if value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def _parse_channel_windows(
    values: list[float] | None,
) -> tuple[tuple[float, float], ...] | None:
    if values is None:
        return None
    if len(values) % 2 != 0:
        raise ValueError(
            "--channel-windows expects an even number of values: min max [min max ...]."
        )
    return tuple(
        (float(values[i]), float(values[i + 1])) for i in range(0, len(values), 2)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Vitessce config JSON from converted zarr inputs."
    )
    parser.add_argument(
        "--resolution",
        required=True,
        help="Grid resolution in microns (e.g. 20 for adata_20um.zarr).",
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Base directory containing adata/image files for relative config paths.",
    )
    parser.add_argument(
        "--adata-path",
        required=True,
        help="Path to input adata zarr (absolute or relative to base-dir).",
    )
    parser.add_argument(
        "--image-path",
        type=_none_or_str,
        default=None,
        help=(
            "Optional path to input OME-Zarr image (absolute or relative to base-dir). "
            "Pass None to omit the image source."
        ),
    )
    parser.add_argument(
        "--output-config-path",
        required=True,
        help="Output path for Vitessce config JSON.",
    )
    parser.add_argument(
        "--name",
        default="Example",
        help="Dataset name in the Vitessce config.",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=-3.2,
        help="Initial spatial zoom.",
    )
    parser.add_argument(
        "--cluster-key",
        type=_none_or_str,
        default=None,
        help="Optional obs cluster key (e.g. leiden_1). Omit to disable cluster view.",
    )
    parser.add_argument(
        "--embedding-key",
        type=_none_or_str,
        default=None,
        help="Optional obsm embedding key (e.g. X_umap). Omit to disable embedding.",
    )
    parser.add_argument(
        "--spatial-key",
        default="spatial",
        help="obsm key used for spatial coordinates.",
    )
    parser.add_argument(
        "--visualize-as-multiplex",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Render the image layer as multiplex (non-RGB). "
            "If omitted, the image renders as RGB by default."
        ),
    )
    parser.add_argument(
        "--qc-obs-feature-keys",
        nargs="*",
        default=[
            "total_counts",
            "n_genes_by_counts",
            "total_counts_mt",
            "pct_counts_mt",
            "pct_counts_in_top_50_genes",
        ],
        help="obs keys to expose as QC features (space-separated list).",
    )
    parser.add_argument(
        "--channel-windows",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Optional per-channel intensity windows as flat min/max pairs, "
            "e.g. --channel-windows 0 500 10 800. Defaults to None."
        ),
    )
    return parser.parse_args()


def _relative_to_base_or_str(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()

    base_dir = Path(args.base_dir)
    adata_path = Path(args.adata_path)
    image_path = Path(args.image_path) if args.image_path is not None else None
    output_config_path = Path(args.output_config_path)

    if not adata_path.is_absolute():
        adata_path = base_dir / adata_path
    if image_path is not None and not image_path.is_absolute():
        image_path = base_dir / image_path

    if not adata_path.exists():
        logger.error("adata-path does not exist: {}", adata_path)
        return 1
    if image_path is not None and not image_path.exists():
        logger.error("image-path does not exist: {}", image_path)
        return 1
    if output_config_path.exists():
        logger.warning("Output exists, overwriting: {}", output_config_path)

    adata = read_zarr(adata_path)

    # Center initial camera on the tissue centroid.
    xy = adata.obsm[args.spatial_key]
    center_x = float(np.mean(xy[:, 0]))
    center_y = float(np.mean(xy[:, 1]))

    adata_source = _relative_to_base_or_str(adata_path, base_dir)
    image_source = (
        _relative_to_base_or_str(image_path, base_dir)
        if image_path is not None
        else None
    )

    resolution = float(args.resolution)
    channel_windows = _parse_channel_windows(args.channel_windows)

    if resolution == 20 or resolution == 120:
        spot_radius_micron = np.sqrt(3) / 2 * (resolution // 2)
    else:
        spot_radius_micron = resolution // 2

    vc = hpv.seq_based_from_split_sources(
        img_source=image_source,
        adata_source=adata_source,
        name=args.name,
        base_dir=base_dir,
        center=(center_x, center_y),
        zoom=args.zoom,
        visualize_as_rgb=not args.visualize_as_multiplex,
        channel_windows=channel_windows,
        emb_radius_mode="auto",
        spot_radius_size_micron=spot_radius_micron,
        cluster_key=args.cluster_key,
        cluster_key_display_name="Leiden",
        embedding_key=args.embedding_key,
        qc_obs_feature_keys=args.qc_obs_feature_keys,
    )

    config_dict = vc.to_dict(
        base_url=str(base_dir),
    )

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with output_config_path.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Wrote config to {}", output_config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
