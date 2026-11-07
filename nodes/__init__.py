"""ComfyUI nodes of Gausify Orbit."""

from .clean import GausifyCleanSplat
from .frame_select import GausifyFrameSelect
from .masks import GausifyMaskChroma, GausifyMaskRMBG, GausifyMaskSAM2, GausifySam2Points
from .poses import GausifyPoses
from .refine import GausifyRefinePoses
from .retouch import GausifyRetouchApply, GausifyRetouchClip
from .train import GausifyTrainAdvanced, GausifyTrainSplat

NODE_CLASS_MAPPINGS = {
    "GausifyOrbitFrameSelect": GausifyFrameSelect,
    "GausifyOrbitMaskChroma": GausifyMaskChroma,
    "GausifyOrbitMaskRMBG": GausifyMaskRMBG,
    "GausifyOrbitMaskSAM2": GausifyMaskSAM2,
    "GausifyOrbitPoses": GausifyPoses,
    "GausifyOrbitTrainSplat": GausifyTrainSplat,
    "GausifyOrbitTrainAdvanced": GausifyTrainAdvanced,
    "GausifyOrbitCleanSplat": GausifyCleanSplat,
    "GausifyOrbitSam2Points": GausifySam2Points,
    "GausifyOrbitRetouchClip": GausifyRetouchClip,
    "GausifyOrbitRetouchApply": GausifyRetouchApply,
    "GausifyOrbitRefinePoses": GausifyRefinePoses,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GausifyOrbitFrameSelect": "Gausify Orbit Frame Select",
    "GausifyOrbitMaskChroma": "Gausify Orbit Mask (Solid Colour)",
    "GausifyOrbitMaskRMBG": "Gausify Orbit Mask (RMBG-1.4)",
    "GausifyOrbitMaskSAM2": "Gausify Orbit Mask (SAM 2.1)",
    "GausifyOrbitPoses": "Gausify Orbit Poses",
    "GausifyOrbitTrainSplat": "Gausify Orbit Train Splat",
    "GausifyOrbitTrainAdvanced": "Gausify Orbit Train Advanced",
    "GausifyOrbitCleanSplat": "Gausify Orbit Clean Splat",
    "GausifyOrbitSam2Points": "Gausify Orbit SAM2 Points (for kijai's SAM2)",
    "GausifyOrbitRetouchClip": "Gausify Orbit Retouch Clip",
    "GausifyOrbitRetouchApply": "Gausify Orbit Retouch Apply",
    "GausifyOrbitRefinePoses": "Gausify Orbit Refine Poses",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
