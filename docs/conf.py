# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'hetGPy'
copyright = "2024, David O'Gara"
author = "David O'Gara"
release = '1'


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import shutil
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath(".."))

## from GPytorch docs
# - Copy over examples folder to docs/source
# This makes it so that nbsphinx properly loads the notebook images

examples_source = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples"))
examples_dest = os.path.abspath(os.path.join(os.path.dirname(__file__), "examples"))

if os.path.exists(examples_dest):
    shutil.rmtree(examples_dest)
os.mkdir(examples_dest)

# Include examples in documentation
# This adds a lot of time to the doc buiod; to bypass use the environment variable SKIP_EXAMPLES=true
for root, dirs, files in os.walk(examples_source):
    for dr in dirs:
        os.mkdir(os.path.join(root.replace(examples_source, examples_dest), dr))
    for fil in files:
        if os.path.splitext(fil)[1] in [".ipynb", ".md", ".rst"]:
            source_filename = os.path.join(root, fil)
            dest_filename = source_filename.replace(examples_source, examples_dest)

            # If we're skipping examples, put a dummy file in place
            if os.getenv("SKIP_EXAMPLES"):
                if dest_filename.endswith("index.rst"):
                    shutil.copyfile(source_filename, dest_filename)
                else:
                    with open(os.path.splitext(dest_filename)[0] + ".rst", "w") as f:
                        basename = os.path.splitext(os.path.basename(dest_filename))[0]
                        f.write(f"{basename}\n" + "=" * 80)

            # Otherwise, copy over the real example files
            else:
                shutil.copyfile(source_filename, dest_filename)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "numpydoc",
    "nbsphinx", # nbsphinx-0.9.4
    "sphinx_rtd_theme"
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    # 'logo_only': False,
}

pygments_style = "sphinx"
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
add_module_names = False
master_doc = "index"
autodoc_typehints = "description"
autodoc_typehints_format = 'short'
python_use_unqualified_type_names = True
autodoc_type_aliases = {
    'Iterable': 'Iterable',
    'ArrayLike': 'ArrayLike',
}