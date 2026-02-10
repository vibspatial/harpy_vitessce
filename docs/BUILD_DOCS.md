# Build and Publish Docs

## Local build

Install docs dependencies:

```bash
python -m pip install -r docs/requirements.txt
```

Build HTML docs:

```bash
make -C docs html
```

The generated site is written to:

`docs/_build/html`

Open locally:

```bash
python -m http.server --directory docs/_build/html 8000
```

Then visit `http://localhost:8000`.

## Clean build output

```bash
make -C docs clean
```

## Publish publicly with Read the Docs

This repo is configured for Read the Docs with:

`/.readthedocs.yaml`

One-time setup in Read the Docs:

1. Sign in at `https://readthedocs.org/`.
2. Import the repository `vibspatial/harpy_vitessce`.
3. Confirm the default branch (`main`) and Python version from `.readthedocs.yaml`.
4. Trigger the first build.

After setup, pushes to the default branch are built automatically by Read the Docs.
