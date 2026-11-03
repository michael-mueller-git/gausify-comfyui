"""ComfyUI nodes of Gausify Orbit."""

from .frame_select import GausifyFrameSelect
from .poses import GausifyPoses
from .train import GausifyTrainSplat

NODE_CLASS_MAPPINGS = {
    "GausifyOrbitFrameSelect": GausifyFrameSelect,
    "GausifyOrbitPoses": GausifyPoses,
    "GausifyOrbitTrainSplat": GausifyTrainSplat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GausifyOrbitFrameSelect": "Gausify Orbit Frame Select",
    "GausifyOrbitPoses": "Gausify Orbit Poses",
    "GausifyOrbitTrainSplat": "Gausify Orbit Train Splat",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
