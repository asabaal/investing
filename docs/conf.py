# docs/conf.py

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Market Analyzer'
copyright = '2025, Asabaal Horan'
author = 'Asabaal Horan'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'nbsphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
]

autosummary_generate = True

# Theme settings
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# NBSphinx settings
nbsphinx_execute = 'never'  # Don't execute notebooks during docs build