"""Shared data types. Every module codes against these; keep them stable.

Conventions (same as gausify-orbit / COLMAP):
- Poses are world->camera: x_cam = R @ X + t. R is a (3,3) float64 array, t (3,).
- Single PINHOLE camera shared by all frames: f (fx = fy), cx, cy, width, height.
- Images are float32 numpy arrays in [0,1], HxWx3 (RGB). Masks are uint8 HxW, 255 = subject.
- Frame file names are 00000.png, 00001.png, ... in orbit order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Pose:
    R: np.ndarray  # (3,3) world->cam rotation
    t: np.ndarray  # (3,)  world->cam translation

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass
class Point3D:
    X: np.ndarray  # (3,)
    rgb: tuple[int, int, int]
    obs: list[tuple[int, int]]  # (image index, keypoint index)
    error: float = 0.0


@dataclass
class SfmModel:
    """Mirror of `SfmModel` in web/src/sfm/reconstruct.ts."""

    f: float
    cx: float
    cy: float
    width: int
    height: int
    poses: list[Pose | None]
    points: list[Point3D]
    names: list[str]
    xy: list[np.ndarray] = field(default_factory=list)  # keypoints (N,2) per image, pixels
    stats: dict = field(default_factory=dict)


@dataclass
class OrbitTimes:
    """ComfyUI type ORBIT_TIMES: timestamp (s) of each selected frame, or the
    source image index when frames came from images (`from_images`)."""

    times: list[float]
    from_images: bool = False
    #: Source frame index of each selected frame (set by frames.select_frames).
    indices: list[int] | None = None


@dataclass
class OrbitCameras:
    """ComfyUI type GAUSIFY_CAMERAS. `root` is a dataset folder:
        root/images/00000.png ...     frames (RGB)
        root/masks/00000.png ...      optional, grayscale 255 = subject
        root/sparse/0/{cameras,images,points3D}.bin   COLMAP model
    """

    root: Path
    names: list[str]
    width: int
    height: int
    masked: bool
    stats: dict = field(default_factory=dict)

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def masks_dir(self) -> Path:
        return self.root / "masks"

    @property
    def sparse_dir(self) -> Path:
        return self.root / "sparse" / "0"
