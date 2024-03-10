project = "frtrg"
copyright = "2023, Valentin Bruch"
author = "Valentin Bruch"
release = "0.14.16"

extensions = ["sphinx.ext.napoleon", "sphinx.ext.imgmath", "autoapi.extension"]
autoapi_dirs = ["../src/frtrg"]
autoapi_python_use_implicit_namespaces = True
autoapi_python_class_content = "init"

templates_path = ["_templates"]
exclude_patterns = ["_build", "autoapi/index.rst"]

html_theme = "furo"
html_static_path = ["_static"]
