<div align="center">
  <img src="docs/_static/logo.png" alt="harpy-vitessce logo" width="220" />
  <p><strong><span style="font-size:1.5em;">Interactive Vitessce visualizations for Harpy workflows.</span></strong></p>
  <p>
    <a href="https://harpy-vitessce.readthedocs.io/en/latest/">
      <img src="https://readthedocs.org/projects/harpy-vitessce/badge/?version=latest" alt="Docs (latest)" />
    </a>
  </p>
</div>

# Harpy-Vitessce

`Harpy-Vitessce` provides utilities to move from Harpy outputs to shareable Vitessce configurations.

For more information on `Harpy` we refer to the [documentation](https://harpy.readthedocs.io/en/latest/).

## Installation (uv)

Install the current `main` branch:

```bash
uv venv .venv_harpy_vitessce_zarr2 --python=3.12
source .venv_harpy_vitessce_zarr_2/bin/activate
uv pip install "harpy-vitessce[vitessce] @ git+https://github.com/vibspatial/harpy_vitessce.git@main"
```

If you want to add it as a project dependency managed by `uv`:

```bash
uv add "harpy-vitessce[vitessce] @ git+https://github.com/vibspatial/harpy_vitessce.git@main"
```

# Documentation

Visium HD: [example with bins](./docs/tutorials/visium_hd/visium_hd_bins_from_spatialdata.ipynb) and an example using spots [here](./docs/tutorials/visium_hd/visium_hd_spots.ipynb).

Proteomics: example can be found [here](./docs/tutorials/proteomics/spatialdata_blobs.ipynb)

# Example:

Example visualization for Visium HD can be found [here](https://vib-data-core.github.io/vitessce/?url=spatial-hackathon-public/sparrow/public_datasets/transcriptomics/visium_hd/config_visium_hd_benchmark_s3_10_2_26.json).

## Disclaimer

This is an independent, third-party integration project and is not affiliated
with, endorsed by, or sponsored by the [Vitessce](https://github.com/vitessce/vitessce) maintainers. Vitessce and
related names may be trademarks of their respective owners.
