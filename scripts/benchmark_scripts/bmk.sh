#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PLATFORMS_TO_RUN=("BMK_S3000")
# Examples:
# PLATFORMS_TO_RUN=("BMK_S3000")
# PLATFORMS_TO_RUN=("BMK_S1000")

NOVA_ST_EXPERIMENT_NAMES=(
  #"Exp100"
  "Exp101"
  "Exp99"
  "Exp102"
  #"exp74"
)

declare -A NOVA_ST_MICRONS_PER_PIXEL_BY_EXPERIMENT=(
  ["Exp100"]="0.340"
  ["Exp101"]="0.340" 
  ["Exp99"]="0.340"
  ["Exp102"]="0.340"
  ["exp74"]="0.341"
)

declare -A NOVA_ST_PLATFORM_BY_EXPERIMENT=(
  ["Exp100"]="BMK_S3000"
  ["Exp101"]="BMK_S3000"
  ["Exp99"]="BMK_S3000"
  ["Exp102"]="BMK_S3000"
  ["exp74"]="BMK_S1000"
)

#RESOLUTIONS=("02" "08" "16" "20" "120")

RESOLUTIONS=( "20" "120")

for PLATFORM in "${PLATFORMS_TO_RUN[@]}"; do
  EXPERIMENT_NAMES=("${BMK_EXPERIMENT_NAMES[@]}")
  declare -n MICRONS_PER_PIXEL_BY_EXPERIMENT=BMK_MICRONS_PER_PIXEL_BY_EXPERIMENT

  for EXPERIMENT_NAME in "${EXPERIMENT_NAMES[@]}"; do
    EXPERIMENT_PLATFORM="${BMK_PLATFORM_BY_EXPERIMENT[${EXPERIMENT_NAME}]:-}"
    if [[ -z "${EXPERIMENT_PLATFORM}" ]]; then
      echo "Missing platform value for experiment ${EXPERIMENT_NAME}" >&2
      exit 1
    fi
    if [[ "${EXPERIMENT_PLATFORM}" != "${PLATFORM}" ]]; then
      continue
    fi

    MICRONS_PER_PIXEL="${MICRONS_PER_PIXEL_BY_EXPERIMENT[${EXPERIMENT_NAME}]:-}"
    if [[ -z "${MICRONS_PER_PIXEL}" ]]; then
      echo "Missing micron-per-pixel value for platform ${PLATFORM}, experiment ${EXPERIMENT_NAME}" >&2
      exit 1
    fi

    BASE_DIR="/data/groups/technologies/spatial.catalyst/Projects/2024-07-UTBenchmark-SpC/data/processed/${PLATFORM}/${EXPERIMENT_NAME}/subsampled_100M"
    INPUT_DIR="${BASE_DIR}/harpy"
    OUTPUT_DIR=/data/groups/technologies/spatial.catalyst/Arne/UTbenchmark/${PLATFORM}/${EXPERIMENT_NAME}/harpy_vitessce # for testing
    #OUTPUT_DIR="${BASE_DIR}/harpy_vitessce"

    SDATA_PATH="${INPUT_DIR}/sdata.zarr"
    OUTPUT_PATH_IMG="${OUTPUT_DIR}/image.ome.zarr"
    IMAGE_LAYER="${EXPERIMENT_NAME}_he"
    IMAGE_LAYER="${IMAGE_LAYER,,}" # make it lowercase

    # need zarr3 environment for conversion
    source /data/groups/technologies/spatial.catalyst/Arne/harpy_vitessce/.venv_harpy_vitessce_zarr3/bin/activate

    # conversion
    for RESOLUTION in "${RESOLUTIONS[@]}"; do
      OUTPUT_PATH_ADATA="${OUTPUT_DIR}/adata_${RESOLUTION}um.zarr"

      TO_COPY_ANNOTATIONS_ARG=()
      if [[ "${RESOLUTION}" == "120" || "${RESOLUTION}" == "20" ]]; then
        TO_COPY_ANNOTATIONS_ARG+=(--to-copy-annotations)
      fi

      python "${SCRIPT_DIR}/convert_zarr_3_zarr_2.py" \
        --resolution "${RESOLUTION}" \
        --sdata-path "${SDATA_PATH}" \
        --output-path-adata "${OUTPUT_PATH_ADATA}" \
        --output-path-img "${OUTPUT_PATH_IMG}" \
        --image-layer "${IMAGE_LAYER}" \
        --microns-per-pixel "${MICRONS_PER_PIXEL}" \
        "${TO_COPY_ANNOTATIONS_ARG[@]}"
    done

    # need zarr2 environment for creating vitessce configs
    source /data/groups/technologies/spatial.catalyst/Arne/harpy_vitessce/.venv_harpy_vitessce_zarr2/bin/activate

    # create config
    for RESOLUTION in "${RESOLUTIONS[@]}"; do
      OUTPUT_PATH_ADATA="${OUTPUT_DIR}/adata_${RESOLUTION}um.zarr"
      OUTPUT_PATH_CONFIG="${OUTPUT_DIR}/vitessce_${RESOLUTION}.config.json"

      if [[ "${RESOLUTION}" == "120" || "${RESOLUTION}" == "20" ]]; then
        CLUSTER_KEY="leiden_1"
        EMBEDDING_KEY="X_umap"
      else
        CLUSTER_KEY="None"
        EMBEDDING_KEY="None"
      fi

      python "${SCRIPT_DIR}/create_vitessce_config.py" \
        --resolution "${RESOLUTION}" \
        --base-dir "${OUTPUT_DIR}" \
        --adata-path "${OUTPUT_PATH_ADATA}" \
        --image-path "${OUTPUT_PATH_IMG}" \
        --output-config-path "${OUTPUT_PATH_CONFIG}" \
        --name "Example" \
        --zoom -3.2 \
        --cluster-key "${CLUSTER_KEY}" \
        --embedding-key "${EMBEDDING_KEY}"
    done
  done
done
