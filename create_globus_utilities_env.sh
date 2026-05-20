#!/usr/bin/env bash

set -euo pipefail

uv venv .venv_harpy_vitessce_zarr_3_globus_utilities --python=3.12
source .venv_harpy_vitessce_zarr_3_globus_utilities/bin/activate

uv pip install -e ".[globus]"
uv pip install jupyter
