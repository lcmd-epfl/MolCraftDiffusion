# Configuration file for the Sphinx documentation builder.
import os
import sys

# Make the package importable for autoapi
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -------------------------------------------------------
project = "MolCraftDiffusion"
copyright = "2025, pregHosh"
author = "pregHosh"
release = "1.7.0"

# -- General configuration -----------------------------------------------------
user_docs_only = os.environ.get("MOLCRAFT_DOCS_USER_ONLY") == "1"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_sitemap",
    "sphinx_design",
]

# Generated API pages reflect docstrings from many integrated upstream models.
# Their warnings remain visible in the full build, while CI checks the
# hand-written, user-facing documentation strictly and deterministically.
if not user_docs_only:
    extensions.extend(["sphinx.ext.intersphinx", "autoapi.extension"])

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]

# autoapi: point at the src layout
autoapi_dirs = ["../src/MolecularDiffusion"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
autoapi_python_use_implicit_namespaces = True
autoapi_add_toctree_entry = False   # we add it manually via api.md
autoapi_ignore = [
    "*/assets/*",  # vendored third-party code (scscore etc.)
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
if user_docs_only:
    exclude_patterns.extend(
        [
            "api.md",
            "autoapi/**",
            "adding_new_models.md",
            "model_integrations/**",
        ]
    )

suppress_warnings = [
    "myst.xref_missing",
    "autoapi.python_import_resolution",
    "toc.not_readable",
]

# Sitemap / SEO
html_baseurl = "https://preghosh.github.io/MolCraftDiffusion/"
sitemap_url_scheme = "{link}"

# -- Options for HTML output ---------------------------------------------------
html_theme = "furo"
html_title = "MolCraftDiffusion"
html_logo = "_static/logo.png"

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_extra_path = ["google093a72c24f91da74.html"]

html_meta = {
    "description": "MolCraftDiffusion is a unified platform for diverse 3D molecular generation workflows in computational chemistry.",
    "keywords": "MolCraftDiffusion, 3D molecular generation, molecular generation, molecular design, diffusion models, flow matching, computational chemistry, generative AI chemistry, drug design, drug discovery, cheminformatics",
}

# Link checking is part of documentation CI. Avoid anchor checks because many
# scientific publishers generate anchors dynamically.
linkcheck_anchors = False
linkcheck_timeout = 20
linkcheck_retries = 2
linkcheck_ignore = [
    # These publisher pages are valid but reject automated link-check requests.
    r"https://chemrxiv\.org/.*",
    r"https://doi\.org/10\.26434/chemrxiv-2024-882hh",
    r"https://doi\.org/10\.1093/bib/bbad435",
    r"https://doi\.org/10\.1021/acs\.jcim\.3c00667",
    r"https://pubs\.acs\.org/doi/10\.1021/jacs\.5c19960",
]

html_theme_options = {
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/pregHosh/MolCraftDiffusion",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#4A90D9",
        "color-brand-content": "#4A90D9",
        "font-stack": "Inter, sans-serif",
        "font-stack--monospace": "JetBrains Mono, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6BAAD9",
        "color-brand-content": "#6BAAD9",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/pregHosh/MolCraftDiffusion",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0"
                     viewBox="0 0 16 16" height="1em" width="1em">
                    <path fill-rule="evenodd"
                          d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                          0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                          -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66
                          .07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15
                          -.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0
                          1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82
                          1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01
                          1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"
                    />
                </svg>
            """,
            "class": "",
        },
    ],
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
