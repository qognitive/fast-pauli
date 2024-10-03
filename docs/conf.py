# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "fast_pauli"
copyright = "2024, Qognitive, Inc."
author = "James E. T. Smith, Eugene Rublenko, Alex Lerner, Sebastien Roy, Jeffrey Berger"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["breathe", "sphinx.ext.autodoc", "sphinx_copybutton", "sphinx.ext.mathjax", "sphinx.ext.napoleon"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = []  # type: ignore


# Breathe configuration
breathe_projects = {
    "fast_pauli": "xml",
}
breathe_default_project = "fast_pauli"
