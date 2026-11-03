"""ComfyUI custom nodes: Gausify Orbit (orbit video -> Gaussian splat).

ComfyUI imports this package by its folder name. The nodes import the `gausify`
library package (next to this file) by absolute name, so the folder is added
to sys.path (appended: ComfyUI's own `nodes` module must keep precedence, and
our `nodes` subpackage is imported relatively).
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.append(_here)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
