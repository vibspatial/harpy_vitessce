"""Sphinx configuration."""

from datetime import datetime
from importlib.metadata import PackageNotFoundError, metadata


DIST_NAME = "harpy-vitessce"

try:
    info = metadata(DIST_NAME)
    project = info["Name"]
    version = info["Version"]
except PackageNotFoundError:
    project = DIST_NAME
    version = "0.1.0"

author = "SaeysLab"
copyright = f"{datetime.now():%Y}, {author}"
release = version

html_context = {
    "display_github": True,
    "github_user": "vibspatial",
    "github_repo": "harpy_vitessce",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_design",
]

autosummary_generate = True
autodoc_process_signature = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
myst_heading_anchors = 3
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

root_doc = "index"
source_suffix = [".rst", ".md", ".ipynb"]
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

intersphinx_mapping = {
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "spatialdata": ("https://spatialdata.scverse.org/en/stable/", None),
}

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_title = project
html_theme_options = {
    "repository_url": "https://github.com/vibspatial/harpy_vitessce",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "path_to_docs": "./docs",
    "show_navbar_depth": 1,
}

pygments_style = "default"
