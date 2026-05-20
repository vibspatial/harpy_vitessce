#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PLATFORMS_TO_RUN=("Visium" "Visium_HD")
# Examples:
PLATFORMS_TO_RUN=("Visium")
# PLATFORMS_TO_RUN=("Visium_HD")

EXPERIMENT_NAMES=(
  "SCA016"
  "SCA001"
  "SCA002"
  "SCA003"
  "SCA005"
  "SCA010"
  "SCA017"
  "SCA018"
  "SCA019"
  "SCA020"
  "SCA021"
)

declare -A MICRONS_PER_PIXEL_BY_EXPERIMENT=(
  ["SCA013"]="0.442"
  ["SCA014"]="0.442"
  ["SCA015"]="0.442"
  ["SCA016"]="0.442"
  ["SCA001"]="0.2196490235773455"
  ["SCA002"]="0.21949344319707406"
  ["SCA003"]="0.4425852423681751"
  ["SCA005"]="0.44223077636391755"
  ["SCA010"]="0.4421652041595308"
  ["SCA017"]="0.44220784830557425"
  ["SCA018"]="0.4422736954685126"
  ["SCA019"]="0.4432412469726364"
  ["SCA020"]="0.4422816200157306"
  ["SCA021"]="0.44208963838418674"
)

declare -A PLATFORM_BY_EXPERIMENT=(
  ["SCA013"]="Visium"
  ["SCA014"]="Visium"
  ["SCA015"]="Visium"
  ["SCA016"]="Visium"
  ["SCA001"]="Visium_HD"
  ["SCA002"]="Visium_HD"
  ["SCA003"]="Visium_HD"
  ["SCA005"]="Visium_HD"
  ["SCA010"]="Visium_HD"
  ["SCA017"]="Visium_HD"
  ["SCA018"]="Visium_HD"
  ["SCA019"]="Visium_HD"
  ["SCA020"]="Visium_HD"
  ["SCA021"]="Visium_HD"
)

RESOLUTIONS=("02" "08" "16" "20" "120")

RESOLUTIONS=( "120")

for PLATFORM in "${PLATFORMS_TO_RUN[@]}"; do
  for EXPERIMENT_NAME in "${EXPERIMENT_NAMES[@]}"; do
    EXPERIMENT_PLATFORM="${PLATFORM_BY_EXPERIMENT[${EXPERIMENT_NAME}]:-}"
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
    OUTPUT_BASE_DIR="${BASE_DIR}/vitessce"
    BUCKET_OUTPUT_BASE_DIR="${PLATFORM}/${EXPERIMENT_NAME}/$(basename "${OUTPUT_BASE_DIR}")"

    SDATA_PATH="${INPUT_DIR}/sdata.zarr"
    IMAGE_LAYER="${EXPERIMENT_NAME}_full_image"

    # need zarr3 environment for conversion
    source /data/groups/technologies/spatial.catalyst/Arne/harpy_vitessce/.venv_harpy_vitessce_zarr3/bin/activate

    # conversion
    for RESOLUTION in "${RESOLUTIONS[@]}"; do
      if [[ "${PLATFORM}" == "Visium" && "${RESOLUTION}" != "20" && "${RESOLUTION}" != "120" ]]; then
        continue
      fi

      OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RESOLUTION}um"
      OUTPUT_PATH_IMG="${OUTPUT_DIR}/image.ome.zarr"
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
      if [[ "${PLATFORM}" == "Visium" && "${RESOLUTION}" != "20" && "${RESOLUTION}" != "120" ]]; then
        continue
      fi

      OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RESOLUTION}um"
      OUTPUT_PATH_IMG="${OUTPUT_DIR}/image.ome.zarr"
      OUTPUT_PATH_ADATA="${OUTPUT_DIR}/adata_${RESOLUTION}um.zarr"
      BUCKET_OUTPUT_DIR="${BUCKET_OUTPUT_BASE_DIR}"

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
        --bucket-output-dir "${BUCKET_OUTPUT_DIR}" \
        --name "Example" \
        --zoom -3.2 \
        --cluster-key "${CLUSTER_KEY}" \
        --embedding-key "${EMBEDDING_KEY}"
    done
  done
done
