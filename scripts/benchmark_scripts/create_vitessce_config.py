import argparse
import json
from pathlib import Path

import numpy as np
from anndata import read_zarr
from dotenv import load_dotenv
from loguru import logger

import harpy_vitessce as hpv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


URL_S3 = "https://objectstor.vib.be"
S3_PATH = (
    "/spatial-hackathon-public/"
    "UT_benchmark"
)


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
        "--bucket-output-dir",
        default=None,
        help=(
            "Optional object-storage destination directory. When provided, "
            "the local base-dir folder is copied into this destination via "
            "Globus, and config_s3.json points to data under the public "
            "bucket URL. Omit to write only config_local.json."
        ),
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


def _relative_to_base(path: Path, base_dir: Path, name: str) -> Path:
    try:
        return path.resolve().relative_to(base_dir.resolve())
    except ValueError as err:
        raise ValueError(
            f"{name} must be inside base-dir when --bucket-output-dir is used: {path}"
        ) from err


def _join_url(base_url: str, *parts: str) -> str:
    url = base_url.rstrip("/")
    for part in parts:
        normalized_part = str(part).strip("/")
        if normalized_part:
            url = f"{url}/{normalized_part}"
    return url


def normalize_compute_source_path(source_path: str) -> str:
    local_prefix = "/data/groups"
    if source_path.startswith(local_prefix + "/"):
        normalized_source_path = source_path[len(local_prefix) :]
        logger.info(
            "Normalizing compute source path from '{}' to '{}'.",
            source_path,
            normalized_source_path,
        )
        return normalized_source_path
    return source_path


def _copy_base_dir_to_bucket(
    base_dir: Path, bucket_output_dir: str
) -> dict[str, object]:
    try:
        from globus_utilities import build_globus_collection_service_from_env
    except ImportError as err:
        raise RuntimeError(
            "Globus publishing requires the 'globus' optional dependency. "
            'Install it with: uv pip install -e ".[globus]"'
        ) from err

    globus_collection_service = build_globus_collection_service_from_env()
    source_collection_id = (
        globus_collection_service.config.compute_storage_collection_id
    )
    destination_collection_id = (
        globus_collection_service.config.object_storage_collection_id
    )

    if not source_collection_id:
        raise ValueError("GLOBUS_COMPUTE_STORAGE_COLLECTION_ID must be set.")
    if not destination_collection_id:
        raise ValueError("GLOBUS_OBJECT_STORAGE_COLLECTION_ID must be set.")

    destination_path = _join_url(S3_PATH, bucket_output_dir)
    logger.info("Copying base-dir into object-storage path '{}'.", destination_path)

    return globus_collection_service.copy_folder_into(
        source_collection_id=source_collection_id,
        source_path=normalize_compute_source_path(str(base_dir.resolve())),
        destination_collection_id=destination_collection_id,
        destination_path=destination_path,
        mode="fails_if_exists",
        label=f"Publish Vitessce config {base_dir.name}",
    )


def main() -> int:
    args = parse_args()

    base_dir = Path(args.base_dir)
    adata_path = Path(args.adata_path)
    image_path = Path(args.image_path) if args.image_path is not None else None
    bucket_output_dir = (
        args.bucket_output_dir.strip() if args.bucket_output_dir is not None else None
    )
    if args.bucket_output_dir is not None and not bucket_output_dir:
        raise ValueError("--bucket-output-dir must not be empty when provided.")
    channel_windows = _parse_channel_windows(args.channel_windows)

    local_config_path = base_dir / "config_local.json"
    s3_config_path = base_dir / "config_s3.json"

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
    if local_config_path.exists():
        logger.warning("Output exists, overwriting: {}", local_config_path)
    if bucket_output_dir is not None and s3_config_path.exists():
        logger.warning("Output exists, overwriting: {}", s3_config_path)

    adata = read_zarr(adata_path)

    # Center initial camera on the tissue centroid.
    xy = adata.obsm[args.spatial_key]
    center_x = float(np.mean(xy[:, 0]))
    center_y = float(np.mean(xy[:, 1]))

    local_adata_source = _relative_to_base_or_str(adata_path, base_dir)
    local_image_source = (
        _relative_to_base_or_str(image_path, base_dir)
        if image_path is not None
        else None
    )

    resolution = float(args.resolution)

    if resolution == 20 or resolution == 120:
        spot_radius_micron = np.sqrt(3) / 2 * (resolution // 2)
    else:
        spot_radius_micron = resolution // 2

    vc_local = hpv.seq_based_from_split_sources(
        img_source=local_image_source,
        adata_source=local_adata_source,
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

    config_local_dict = vc_local.to_dict(base_url=str(base_dir))

    base_dir.mkdir(parents=True, exist_ok=True)
    with local_config_path.open("w", encoding="utf-8") as f:
        json.dump(config_local_dict, f, indent=2)

    logger.info("Wrote local config to {}", local_config_path)
    if bucket_output_dir is not None:
        adata_relative_path = _relative_to_base(
            adata_path,
            base_dir,
            "adata-path",
        )
        image_relative_path = (
            _relative_to_base(image_path, base_dir, "image-path")
            if image_path is not None
            else None
        )
        bucket_base_dir_url = _join_url(
            URL_S3,
            S3_PATH,
            bucket_output_dir,
            base_dir.resolve().name,
        )
        s3_adata_source = _join_url(
            bucket_base_dir_url,
            adata_relative_path.as_posix(),
        )
        s3_image_source = (
            _join_url(bucket_base_dir_url, image_relative_path.as_posix())
            if image_relative_path is not None
            else None
        )

        vc_s3 = hpv.seq_based_from_split_sources(
            img_source=s3_image_source,
            adata_source=s3_adata_source,
            name=args.name,
            base_dir=None,
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
        config_s3_dict = vc_s3.to_dict(base_url=None)
        with s3_config_path.open("w", encoding="utf-8") as f:
            json.dump(config_s3_dict, f, indent=2)
        logger.info("Wrote S3 config to {}", s3_config_path)

        transfer_result = _copy_base_dir_to_bucket(base_dir, bucket_output_dir)
        logger.info("Submitted Globus transfer: {}", transfer_result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
