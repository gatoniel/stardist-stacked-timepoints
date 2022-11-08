"""Sphinx configuration."""
project = "Stardist Stacked Timepoints"
author = "Niklas Netter"
copyright = "2022, Niklas Netter"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
