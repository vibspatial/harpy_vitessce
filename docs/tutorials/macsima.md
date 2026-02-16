# MACSima

Use `harpy_vitessce.vitessce_config.macsima` to build an image-only Vitessce view.

The API supports two source modes:

1. `sdata` mode (preferred): resolve the image from `sdata.path / "images" / img_layer`.
2. `img_source` mode: pass a direct local path or remote URL.

## Example: from SpatialData

```python
from harpy_vitessce.vitessce_config import macsima

vc = macsima(
    sdata=sdata,
    img_layer="raw_image",
    name="MACSima",
    channels=[0, 1, 2],
)
```

## Example: from image source

```python
from harpy_vitessce.vitessce_config import macsima

vc = macsima(
    img_source="path/to/image.ome.zarr",
    base_dir=".",
    name="MACSima",
)
```

## Notes

- Parameters are keyword-only for clarity.
- If `sdata` is provided, `img_source` and `base_dir` are ignored (with warnings).
- If `sdata` is provided, `img_layer` is required.
