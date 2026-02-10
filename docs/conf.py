"""Sphinx configuration."""

project = "harpy-vitessce"
extensions = [
    "myst_nb",
]

root_doc = "index"
source_suffix = [".rst", ".md", ".ipynb"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# Avoid executing notebooks during docs builds.
nb_execution_mode = "off"

html_theme = "alabaster"
