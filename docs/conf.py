# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Imports -----------------------------------------------------------------

import datetime as dt

import sphinx_autosummary_accessors

# need to import so accessors get registered
import pint_xarray  # noqa: F401

# -- Project information -----------------------------------------------------

year = dt.datetime.now().year
project = "pint-xarray"
author = f"{project} developers"
copyright = f"{year}, {author}"
github_url = "https://github.com/xarray-contrib/pint-xarray"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autosummary_accessors",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates", sphinx_autosummary_accessors.templates_path]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]


# -- Extension configuration -------------------------------------------------

# extlinks
extlinks = {
    "issue": (f"{github_url}/issues/%s", "GH"),
    "pull": (f"{github_url}/pull/%s", "PR"),
}

# autosummary
autosummary_generate = True

# autodoc
autodoc_typehints = "none"

# napoleon
napoleon_use_param = False
napoleon_use_rtype = True

napoleon_preprocess_types = True
napoleon_type_aliases = {
    "dict-like": ":term:`dict-like <mapping>`",
    "mapping": ":term:`mapping`",
    "hashable": ":term:`hashable`",
    # xarray
    "Dataset": "~xarray.Dataset",
    "DataArray": "~xarray.DataArray",
    # pint / pint-xarray
    "unit-like": ":term:`unit-like`",
}

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "xarray": ("https://xarray.pydata.org/en/stable", None),
    "pint": ("https://pint.readthedocs.io/en/stable", None),
}
