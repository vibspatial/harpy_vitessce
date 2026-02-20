<div align="center">
  <img src="docs/_static/logo.png" alt="harpy-vitessce logo" width="220" />
  <p><strong><span style="font-size:1.5em;">Interactive Vitessce visualizations for Harpy workflows</span></strong></p>
  <p>
    <a href="https://harpy-vitessce.readthedocs.io/en/latest/">
      <img src="https://readthedocs.org/projects/harpy-vitessce/badge/?version=latest" alt="Docs (latest)" />
    </a>
  </p>
  <p>
    <a href="https://harpy-vitessce.readthedocs.io/en/latest/">Documentation</a>
    ·
    <a href="#quick-start">Quick Start</a>
    ·
    <a href="#tutorials">Tutorials</a>
  </p>
</div>

# Harpy-Vitessce

`harpy-vitessce` helps you turn Harpy outputs into shareable, interactive
[Vitessce](https://github.com/vitessce/vitessce) configurations.

For more information on Harpy, see the
[Harpy documentation](https://harpy.readthedocs.io/en/latest/).

## Why Use Harpy-Vitessce?

- Build Vitessce configs directly from `SpatialData` or from seperate files on disk.
- Get ready-to-share, interactive visualizations for transcriptomics and proteomics
- Use sensible defaults while keeping control over layout and rendering options

## Installation (`uv`)

Install the current `main` branch:

```bash
uv venv .venv_harpy_vitessce_zarr2 --python=3.12
source .venv_harpy_vitessce_zarr2/bin/activate
uv pip install "harpy-vitessce[vitessce] @ git+https://github.com/vibspatial/harpy_vitessce.git@main"
```

Add it as a project dependency managed by `uv`:

```bash
uv add "harpy-vitessce[vitessce] @ git+https://github.com/vibspatial/harpy_vitessce.git@main"
```

## Quick Start

The full quick-start guide lives in [`docs/quickstart.md`](./docs/quickstart.md).

## Tutorials

- Visium HD (bins): [`docs/tutorials/visium_hd/visium_hd_bins_from_spatialdata.ipynb`](./docs/tutorials/visium_hd/visium_hd_bins_from_spatialdata.ipynb)
- Visium HD (spots): [`docs/tutorials/visium_hd/visium_hd_spots.ipynb`](./docs/tutorials/visium_hd/visium_hd_spots.ipynb)
- Proteomics SpatialData `blobs`: [`docs/tutorials/proteomics/spatialdata_blobs.ipynb`](./docs/tutorials/proteomics/spatialdata_blobs.ipynb)

## Live Example

Example Visium HD visualization:
[Open in Vitessce](https://vib-data-core.github.io/vitessce/?url=spatial-hackathon-public/sparrow/public_datasets/transcriptomics/visium_hd/config_visium_hd_benchmark_s3_10_2_26.json)

## Disclaimer

This is an independent, third-party integration project and is not affiliated
with, endorsed by, or sponsored by the
[Vitessce](https://github.com/vitessce/vitessce) maintainers. Vitessce and
related names may be trademarks of their respective owners.
