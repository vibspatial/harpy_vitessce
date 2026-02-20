# Quick Start

This example shows how to generate a Vitessce configuration from a
`SpatialData` example and open it in the browser.

```python
import tempfile
from pathlib import Path

import scanpy as sc
from IPython.display import HTML, display
from spatialdata.datasets import blobs

import harpy_vitessce as hpv

tmp_dir = Path(tempfile.mkdtemp(prefix="spatialdata_blobs"))

sdata = blobs()
adata = sdata["table"]

# add leiden clusters using a dummy scanpy pipeline
sc.pp.scale(adata, max_value=10)
sc.pp.pca(
    adata,
    n_comps=2,
    svd_solver="arpack",
)
sc.pp.neighbors(
    adata,
    use_rep="X_pca",
    n_neighbors=10,
)
sc.tl.leiden(adata, resolution=0.6, key_added="leiden")
sc.tl.umap(adata, min_dist=0.3)

sdata.write(tmp_dir / "sdata.zarr")

# generate the vitessce config:
vc = hpv.proteomics_from_spatialdata(
    sdata_path=sdata.path,
    labels_layer="blobs_labels",
    img_layer="blobs_multiscale_image",
    table_layer="table",
    channels=[0, 1, 2],
    palette=["#1F77B4", "#FF7F0E", "#2CA02C"],
    to_coordinate_system="global",
    visualize_heatmap=True,
    visualize_feature_matrix=False,
    cluster_key="leiden",
)

url = vc.web_app()
display(HTML(f'<a href="{url}" target="_blank">Open in Vitessce</a>'))
```
