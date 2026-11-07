# Gausify Orbit for ComfyUI

Turn an orbit video (a camera circling a subject, e.g. an AI-generated
turntable) into a 3D Gaussian splat, inside ComfyUI. The nodes reproduce the
pipeline and output of [gausify-orbit](https://github.com/obraia/gausify-orbit):
sharp-frame selection, camera poses (AKAZE + incremental SfM, or COLMAP's SIFT
+ global mapper; beyond the app, learned ALIKED + LightGlue features), optional masked (cut-out) training, and training with
[Brush](https://github.com/ArthurBrussee/brush).

**NVIDIA / CUDA only, Linux x86-64 only.**

## Nodes (category "Gausify Orbit")

| Node | Inputs | Outputs |
|---|---|---|
| **Gausify Orbit Frame Select** | `video` (core VIDEO) or `images` + `fps` (0 = image sequence); preset, frames, max_side, candidates, trim_start, trim_end (-1 = end); skip_duplicates, min_sharpness; `masks` (optional, one per source frame) | IMAGE, ORBIT_TIMES, MASK (the picked frames' masks) |
| **Gausify Orbit Mask (Solid Colour)** | images; auto_color, color, tolerance, softness, contiguous | MASK, stats (JSON) |
| **Gausify Orbit Mask (RMBG-1.4)** | images; accept_noncommercial_license, device | MASK, stats (JSON) |
| **Gausify Orbit Mask (SAM 2.1)** | images; prompt_frame, box, points, variant (`fp32`/`fp16`), device | MASK, stats (JSON) |
| **Gausify Orbit Poses** | images, frame_times, masks (optional); preset, estimator (`akaze`/`aliked_lightglue`/`colmap`), match_window, closed_loop, flat_orbit, fill_missing, use_mask_for_poses, pose_check (`off`/`auto`/`always`, needs masks), sfm_max_side + max_features (AKAZE / ALIKED limits, app: 1600 / 4000) | GAUSIFY_CAMERAS, colmap_path, stats (JSON) |
| **Gausify Orbit Train Splat** | cameras; preset, steps, sh_degree, max_splats (millions), max_resolution, ssim, reduce_haze (masked only), psnr, despill + strength (masked only), seed, filename_prefix, psnr_holdout_every, final_frames (`holdout`/`all_frames`), drop_inconsistent, deflicker; advanced (optional) | ply_path (relative to the output folder), num_splats, psnr |
| **Gausify Orbit Train Advanced** | Brush settings: growth, contribution pruning, regularisers (opacity, volume, flatten, normal consistency), learning-rate multipliers, loss (`l1_ssim` = Brush's/the app's, or `l1_ssim_lpips` + lpips_weight), render_mode (`default`/`mip`), masked_bg_grey + masked_bg_noise (masked training background, app: 0.5 / 0.5). Defaults are Brush's own (the background: the app's), so connecting it unchanged does nothing | advanced → Train Splat |
| **Gausify Orbit Clean Splat** | cameras, ply_path; hull_carve + hull_tolerance + mask_margin (masked only), drop_unseen, max_distance, min_opacity, max_size, check_psnr, hull_extent, auto_tolerance + max_psnr_drop | ply_path (`<splat>_clean.ply`), num_splats, stats (JSON) |
| **Gausify Orbit Refine Poses** | cameras, ply_path; max_side, max_features, min_inliers, max_rotation (deg), max_shift (% of the orbit radius) | GAUSIFY_CAMERAS (corrected poses, for a second Train Splat), stats (JSON) |
| **Gausify Orbit SAM2 Points** | images; points (normalized `x,y,label;...`) | coordinates_positive, coordinates_negative (pixel JSON for kijai's SAM2 nodes) |
| **Gausify Orbit Retouch Clip** | cameras, ply_path; motion (`wave`, `orbit_below`/`_above`, `arc_below`/`_above`), tilt, photo, seconds, fps, long_side | reference (IMAGE), alpha (MASK), clip, info |
| **Gausify Orbit Retouch Apply** | cameras, ply_path, clip, restored (IMAGE); steps, restored_ratio, max_splats, max_resolution, ssim, reduce_haze, despill + strength, seed, drop_inconsistent, deflicker; advanced (optional) | ply_path (`<splat>_retouch.ply`), num_splats, stats (JSON) |

`preset` (`preview`, `balanced`, `quality`, `custom`) is the app's quality
preset: anything but `custom` overrides the node's preset-controlled widgets
(the UI updates them when you change the preset; editing one switches to
`custom`). Defaults are `balanced`.

Connect `ply_path` to the core **Preview3D** node to view the splat. An
example workflow is in [`workflows/gausify_orbit.json`](workflows/gausify_orbit.json).

Training frames: with `psnr` on, every `psnr_holdout_every`-th frame (app: 16)
is held out to measure PSNR, so the saved splat never sees those frames.
`final_frames = all_frames` then trains a second time on every frame and saves
that splat (about twice the time; `psnr` comes from the first run). With `psnr`
off, every frame is used and training runs once.

Inconsistent frames: `drop_inconsistent` (Train Splat and Retouch Apply, 0 =
off) leaves out up to that fraction of the frames that disagree most with the
rest, e.g. 0.1. Those are frames where the subject moved (common in generated
videos) or the pose is off, and they cause ghosting. The scores come from the
Poses node's pose check, so it needs masks and `pose_check` `auto` or `always`;
corrected poses are scored again after correction. Only clear outliers (score
above 1.5x the median) are dropped, and at least 8 frames are kept, so a clean
orbit loses nothing. Without pose check scores it logs a warning and trains on
every frame.

Deflicker: `deflicker` (Train Splat and Retouch Apply, 0 = off, 1 = full
correction) evens out exposure and white-balance flicker between frames, which is
common in generated videos and shows up as blotchy colour and haze in the
splat. Per frame and colour channel, the subject's average brightness (alpha >
0.5 with masks, else the whole frame, clipped pixels left out) is pulled
toward the median of its 9 neighbours in orbit order with a gain (at most
1.5x). Slow lighting changes around the orbit survive, because the median
follows them. The first and last few frames get a smaller window, and the
very first and last are left as they are. Only the temporary training copy is changed
(without masks, a copy of the dataset is made for it). The UI line reports the
mean and largest correction.

LPIPS: Train Advanced's `loss = l1_ssim_lpips` adds Brush's LPIPS perceptual
loss (VGG, built into brush-cli) to L1 + SSIM, weighted by `lpips_weight`
(default 0.1; 0.05-0.2 is typical, more can add high-frequency artefacts).
It gives sharper hair, fabric and skin texture, at the cost of slower steps
and more GPU memory. `l1_ssim` (the default) is the current loss.

Anti-aliasing: Train Advanced's `render_mode = mip` trains with
Mip-Splatting's 2D filter. The splat then stays clean when viewed closer or
farther than the orbit (less aliasing, fewer bloated small splats). The mode is
recorded in the .ply (`comment SplatRenderMode: mip`), and Brush and this pack's
renders (Clean Splat, Retouch, pose check) use it. Viewers that ignore it show
slightly different fine edges.

Masked background: masked training composites the cut-outs over a random
grey each step (the app: 0.5 +- 0.5), which forces clean transparency.
`masked_bg_grey` / `masked_bg_noise` change it. Less noise converges faster but
can leave fringes at the silhouette in the colour of that background. A
fixed background matching your viewer (e.g. grey 0, noise 0.1 for black) hides
them there. Unmasked training ignores these settings.

Frame selection extras (off by default, as in the app): `skip_duplicates`
drops picks that repeat the previous one (frozen or repeated frames of
generated videos), `min_sharpness` drops picks blurrier than that fraction of
the median sharpness (e.g. 0.4). Fewer frames may come out. `masks` takes one
mask per source frame (e.g. SAM2 tracking on the whole video, where tracking
works best). The `masks` output holds the picked frames' masks at the frames'
size, ready for the Poses node.

### Pose estimators

The Poses node's `estimator`:

- `akaze` (default, the app's): AKAZE features, nearest-neighbour matching and an
  incremental SfM. Fast, also on CPU.
- `aliked_lightglue` (beyond the app): learned features. RaCo keypoints with
  ALIKED descriptors, matched by LightGlue, then the same incremental SfM as
  `akaze` (so `fill_missing`, `flat_orbit` and the pose check work the same).
  Learned features stay stable under the things that make AKAZE lose track:
  skin, plain clothes, motion blur and large angle steps between frames. LightGlue
  also rejects matches that only look alike, as on fabric patterns or hair. Try it when
  `akaze` registers too few cameras or the pose check reports bad frames.
  `sfm_max_side` is the size the network runs at (rounded to multiples of 32).
  `max_features` picks the model: 1024, 2048, 3072 or 3584 keypoints per frame, the
  largest up to the value (the default 4000 picks 3584). With `use_mask_for_poses`,
  only keypoints on the subject are kept, so with a small subject in the frame
  fewer keypoints remain; then use the 3584 model. The ONNX model (about
  65 MB, from [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX),
  Apache-2.0; RaCo and LightGlue Apache-2.0, ALIKED BSD-3-Clause) is downloaded
  on first use into `models/gausify/models`. It runs on ONNX Runtime's CUDA
  provider; on CPU it takes seconds per frame pair (minutes for a whole
  orbit). On first use the model is split into its feature extractor and
  its LightGlue matcher (needs the `onnx` package, in `requirements.txt`;
  cached next to the model). Every frame then goes through the extractor once,
  and only the matcher runs per pair, with the same output. Without `onnx`, the
  published pair pipeline runs as is (it recomputes a frame's features for every
  pair; a warning says so).
- `colmap`: COLMAP's SIFT + global mapper (isolated venv, see Install).

### Refining the poses against the splat

Brush can't adjust the cameras while it trains, so a slightly wrong pose
stays in the splat as ghosting, doubled edges and soft detail. **Gausify Orbit
Refine Poses** corrects the poses against a trained splat:

```
Poses -> Train Splat -> Refine Poses (cameras + ply_path) -> Train Splat -> Clean Splat ...
```

It renders the splat from every camera (colour and depth), matches each
render with its photo using ALIKED-LightGlue (the same model as the Poses
node's `aliked_lightglue`), lifts the render's keypoints to 3D with the render
depth (opaque pixels away from depth edges; with masks, only keypoints on the
subject), and solves the photo's pose from these 2D-3D matches (PnP-RANSAC,
then least squares). The render is the consensus of all frames, so a photo whose
pose is off sits shifted against it. A new pose is used only when it lowers the
frame's median reprojection error clearly (by at least 1.5x and 0.25 px),
has at least `min_inliers` inliers, and changes the pose by at most
`max_rotation` / `max_shift`. Everything else keeps its pose; the stats list
why. Smaller improvements are within the matching noise, where a small
rotation and a sideways shift look alike. The world frame, the intrinsics and
the 3D points stay as they were. The output is a new dataset folder (frames
hard-linked), so the Poses node's cache is untouched.

[`workflows/gausify_orbit_person_refine.json`](workflows/gausify_orbit_person_refine.json)
is the person workflow with this built in: a 15k-step first pass
(`gausify/person_pass1`), Refine Poses, then the 30k-step final training on
the refined poses, cleaned and previewed. That's about 1.5x the time of the
single-pass workflow.

Since the splat was also fitted to the wrong poses, one round under-corrects:
train, refine, train again. A second refine round on the new splat corrects
further. Needs the ALIKED-LightGlue model (downloaded on first use) and, to be
fast, a GPU.

### Cleaning up the splat

**Gausify Orbit Clean Splat** (not part of the app) goes between Train Splat and
Preview3D. It removes Gaussians that can't belong to the subject, judged from
the dataset's cameras, and saves `<splat>_clean.ply` next to the input:

- `hull_carve` (masked datasets): it removes a splat whose centre falls outside the subject
  mask (grown by `mask_margin` % of the image) in more than `hull_tolerance` of
  the views that see it. This catches floaters, the ghost silhouette and leftover background.
  A higher tolerance forgives masks that miss a hand or hair in a few frames.
- `drop_unseen`: it removes splats that no camera sees.
- `max_distance`, `min_opacity`, `max_size` (0 = off): these remove splats far from the
  orbit centre (in orbit radii; e.g. 1.5 drops distant background without
  masks), nearly transparent ones, and huge ones (largest axis in orbit radii, e.g. 0.1).
- `check_psnr`: it renders 12 training views before and after and puts both PSNRs in
  `stats`. A clear drop means the clean-up cut into the subject.
- `hull_extent`: it judges each splat by its footprint (2x its largest axis) instead
  of only its centre. This also removes big blobs centred on the subject that
  spill far past its silhouette.
- `auto_tolerance` (with `check_psnr`): if the PSNR drops by more than
  `max_psnr_drop` (0.3 dB), it retries with `hull_tolerance` +0.15 (up to 0.85).
  `stats.auto` lists the tries.

### Retouch: fill in angles the orbit missed

A port of the app's Retouch ([docs](https://github.com/obraia/gausify-orbit)):

1. **Gausify Orbit Retouch Clip** renders the splat along a camera path that
   starts at a real photo and goes through angles the video didn't cover:
   - `wave`: a full loop, above then below the photos
   - `orbit_below` / `orbit_above`: a full loop at another height
   - `arc_below` / `arc_above`: a third of the orbit around `photo`

   `tilt` is the height change in degrees. Frames are rounded to 8n + 1 (LTX).
   `info` warns when the camera moves faster than 30 deg/s, which video models
   blur, and suggests a longer duration. The app's view-based motions (from the
   3D view, recorded motion) need its viewer and aren't ported.
2. Restore `reference` with any ComfyUI video workflow, e.g. LTX with an IC-LoRA
   (see the app's `assets/workflows/ltx-2.5-refinement.json`). The result needs the same
   number of frames and aspect ratio; the size may change.
3. **Gausify Orbit Retouch Apply** fine-tunes the splat from itself (Brush
   `init.ply`) on the photos plus some of the restored frames:
   - `restored_ratio` caps the restored frames at a fraction of the photo count
     (default 0.25). More frames pull the splat toward details the video model
     invented.
   - The fine-tune runs `steps` (default 3000) with a 10x lower position learning
     rate. Growth stops at half.
   - The restored frames take the render's alpha, so the subject stays a cut-out.

   It saves `<splat>_retouch.ply`. Put Clean Splat after it if you like.

### Recommended settings for people

[`workflows/gausify_orbit_person.json`](workflows/gausify_orbit_person.json) is
set up for a camera orbiting a person who stands still:

- **Frames**: `quality` preset (220 frames, 1920 px).
- **Masks**: SAM 2.1 (fp32) tracks the object in the centre of frame 0. If it picks the wrong object, set a `box`.
- **Poses**: `use_mask_for_poses` is on, so the background can't pull the
  cameras. Turn it off for a real orbit with a detailed static background around a
  person with little texture. Also: `pose_check = auto`, `fill_missing`,
  `sfm_max_side 1920` and `max_features 8000` (more matches on skin and plain
  clothes). If AKAZE still loses cameras, switch `estimator` to
  `aliked_lightglue` (see Pose estimators).
- **Train**: 30k steps, SH degree 2, 3M splats, 1920 px, SSIM, `reduce_haze`,
  `drop_inconsistent 0.1`.
  `psnr` is off, so every frame trains the splat (set `final_frames = all_frames`
  if you turn it on).
- **Train Advanced**: contribution pruning (`contrib_prune_min_views 3`) removes
  ghosts and doubled edges caused by small movements of the person. Opacity
  regularisers stay at 0 because they harden hair.
- **Clean Splat**: hull carving with its defaults; Preview3D shows the cleaned splat.

Everything else in that workflow stays at the app's defaults.

Masks: plug any node that produces one MASK per selected frame (white =
subject) into the Poses node. With masks, training is done on RGBA cut-outs
and the pose check becomes available.

## Masks

Three nodes cut the subject out of the Frame Select images (ports of the
masking methods of gausify-orbit); connect their `masks` to the Poses node.
`stats` is JSON: `count`, `meanCoverage`, `noSubject`, `backend`, `keyColor`.

| Method | How | License | Download |
|---|---|---|---|
| Solid Colour | Chroma key: distance to the background colour (median of the frame borders, or `color`), tolerance, softness, flood fill from the borders (`contiguous` keeps holes in the subject) | none (no model) | none |
| RMBG-1.4 | BRIA AI background removal, one frame at a time | **non-commercial** (BRIA RMBG-1.4); set `accept_noncommercial_license` (or `GAUSIFY_ACCEPT_RMBG_LICENSE=1` for the whole instance) | 176 MB |
| SAM 2.1 | Meta's SAM 2.1 tiny video tracking: mark the subject on `prompt_frame` with `box` (`x0,y0,x1,y1`) and/or `points` (`x,y,label;...`, label 1 = subject, 0 = background), all normalized 0-1; with neither, the object at the centre is tracked. Propagated forward, then backward; consistent across frames | Apache-2.0 | fp32 165 MB, fp16 84 MB |

Models are downloaded on first use (sha256-checked) into `$GAUSIFY_MODEL_DIR`,
else `ComfyUI/models/gausify/models` (else `~/.cache/gausify/models`). Use a local
file with `GAUSIFY_MODEL_RMBG=/path/model.onnx` (SAM 2.1:
`GAUSIFY_MODEL_SAM2_FP32_VISION_ENCODER`, ...; names in `gausify/models.json`).
`device` is `auto` (CUDA, else CPU), `cuda` or `cpu`.

Despill: with a green or blue screen, enable `despill` on the Train node. It
is meant for Solid Colour masks (as in the app, where only chroma-key masks
are despilled): it needs a saturated green or blue background in the masked-out area.

### Masks from other node packs

The Poses node takes any `MASK` batch with one mask per Frame Select image
(white = subject, soft values allowed; resized to the frames if needed). Put
the mask nodes after Frame Select, on its `IMAGE` output, so the masks match the
selected frames. With these packs, you get better edges than the built-in nodes
(the built-in SAM 2.1 tiny decodes 256x256 masks, RMBG-1.4 works at 1024x1024):

| Pack | Nodes | Use | Model license |
|---|---|---|---|
| [ComfyUI-segment-anything-2](https://github.com/kijai/ComfyUI-segment-anything-2) (kijai) | SAM 2.1 video segmentation, tiny / small / base_plus / large | Tracking with point prompts; large keeps arms and legs as the person turns | Apache-2.0 |
| [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) (1038lab) | BiRefNet (incl. BiRefNet-HR up to 2048 px and a matting variant), BEN2, RMBG-2.0, SAM2 | Per-frame cut-out, no prompt; often enough for one person on a calm background | BiRefNet MIT; **RMBG-2.0 non-commercial**; others: see the pack |
| [ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle) | MaskEdgeUltraDetail V2 (VITMatte, PyMatting, GuidedFilter) | Refines a coarse mask (SAM, BiRefNet) into a soft alpha for hair and edges | ViTMatte MIT |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | Grow Mask With Blur | Grows, shrinks or feathers a mask (e.g. one that cuts into the edge) | no model |

Suggested chains for people:

1. SAM 2.1 large (segment-anything-2) on every video frame → Frame Select `masks` → MaskEdgeUltraDetail V2 (VITMatte) on Frame Select's images and masks → Poses `masks`.
   Ready-made: [`workflows/gausify_orbit_person_sam2_vitmatte.json`](workflows/gausify_orbit_person_sam2_vitmatte.json)
   (the person workflow with these masks). **Gausify Orbit SAM2 Points** converts normalized points
   (`x,y,label;...`, default chest and hips at the centre) into the pixel coordinates kijai's
   `Sam2VideoSegmentationAddPoints` expects, so no point editor is needed. Connect its
   `coordinates_negative` only when you add background points (label 0).
2. Frame Select → BiRefNet-HR or BiRefNet matting (ComfyUI-RMBG) → Poses `masks`. Simplest, but frame by frame: check it picks the person in every frame.

Soft mattes become the alpha of the training frames, so hair keeps its
transparency. Clean Splat treats masks at 50% (value 128) as the edge, and
`mask_margin` protects the soft border. Check each pack's README for its own
install steps and model downloads.

ONNX Runtime: the nodes need `onnxruntime-gpu`. The plain `onnxruntime` package
conflicts with it (both provide the same module); if another node pack
installed it, `pip uninstall onnxruntime` first, then reinstall `onnxruntime-gpu`.

## Install

```bash
source /path/to/ComfyUI/venv/bin/activate  # ComfyUI's Python env, e.g. /comfy/mnt/venv
cd ComfyUI/custom_nodes
git clone https://github.com/michael-mueller-git/gausify-comfyui
gausify-comfyui/scripts/install.sh            # add --models and/or --colmap, see below
```

`scripts/install.sh` installs `requirements.txt` and repairs the two setups
that break the nodes: it keeps exactly one OpenCV build, between 4.12 and 5
(an installed contrib build stays contrib), and replaces plain `onnxruntime`
with `onnxruntime-gpu`. It then downloads `brush-cli` into
`models/gausify/bin`, and checks that OpenCV has AKAZE and that ONNX Runtime
can use CUDA. Options:

- `--models`: also downloads the ONNX models (SAM 2.1 fp32, RMBG-1.4 and the 3584-keypoint
  ALIKED-LightGlue, about 415 MB). With them in
  place, the check also tests that CUDA actually initialises.
- `--colmap`: also builds the COLMAP venv (see below).

Without an active venv it uses `venv/` or `.venv/` next to ComfyUI (or the base directory), or
`/comfy/mnt/venv`; `PYTHON=/path/to/python` selects the interpreter explicitly.
Running the script again is safe. Restart ComfyUI afterwards.

Manual install instead: `pip install -r gausify-comfyui/requirements.txt`.

If ComfyUI runs with `--base-directory` (e.g. Docker images using `/basedir`),
it loads custom nodes from `<base-directory>/custom_nodes`, not
`ComfyUI/custom_nodes`; clone there instead.

Keep a folder name other than `gausify` (that name is the library package
inside). Dependencies: numpy, scipy, opencv-python-headless (>= 4.12, < 5: OpenCV 5 drops `cv2.AKAZE_create`), pillow, onnx (splits the ALIKED-LightGlue model) and
onnxruntime-gpu. `brush-cli` is downloaded on first use into
`ComfyUI/models/gausify/bin` (override with `GAUSIFY_BIN_DIR`).

COLMAP (the `colmap` pose estimator) is not installed by `requirements.txt`.
`pycolmap-cuda12` 4.2.1 needs CUDA 12 libraries (`cuda-toolkit<13`), which
conflict with the CUDA 13 `cuda-toolkit` that recent torch builds pin, so
installing it into ComfyUI's venv would break that environment. Instead, the first
use of the `colmap` estimator creates an isolated venv and installs `pycolmap-cuda12`,
`numpy` and the CUDA 12 runtime wheels there (a few hundred MB, needs network and
pip access; your pip configuration, e.g. `PIP_INDEX_URL`, applies). COLMAP then runs
in a subprocess from that venv; ComfyUI never imports it. The venv is at
`$GAUSIFY_COLMAP_VENV` if set, else `ComfyUI/models/gausify/colmap-venv`
(else `~/.cache/gausify/colmap-venv`), and is rebuilt automatically if the pinned
packages change. `GAUSIFY_COLMAP_PIP_ARGS` adds arguments to its `pip install`
(e.g. `--index-url https://...`). The AKAZE estimator needs none of this.

### COLMAP environment: set up ahead of time

To build the COLMAP venv before the first run (e.g. to see pip's output, or on
a machine where ComfyUI has no network), run the setup script with ComfyUI's
venv active. It runs the node's own setup, so the venv is exactly what the node
expects, and then checks that COLMAP sees the GPU:

```bash
source /path/to/ComfyUI/venv/bin/activate
ComfyUI/custom_nodes/gausify-comfyui/scripts/setup-colmap-env.sh
```

It prints `cuda build: True | devices: 1` (or more) on success and exits with 3
if COLMAP sees no CUDA device. The venv goes to `$GAUSIFY_COLMAP_VENV`, else
`ComfyUI/models/gausify/colmap-venv`; pass a directory as the first argument to
choose another (then export `GAUSIFY_COLMAP_VENV` for ComfyUI too). `PYTHON=...`
selects the interpreter instead of the active venv. Use ComfyUI's Python: the venv
is stamped with its version and rebuilt if it differs. Don't create it with a plain
`python -m venv` + `pip install`; without the stamp the node replaces it.

### Upgrading from a version that installed pycolmap into ComfyUI's venv

Older versions put `pycolmap-cuda12` in ComfyUI's venv, which downgraded
torch's `cuda-toolkit` metapackage (and pinned opencv 4.12 / numpy < 2.3).
Restore torch's `cuda-toolkit` version (see `pip check`; torch 2.15 wants 13.2.2),
remove pycolmap, and reinstall:

```bash
pip install cuda-toolkit==13.2.2          # the version torch requires
pip uninstall -y pycolmap-cuda12
pip install -r requirements.txt
pip check
```

If other nodes needed newer opencv/numpy than 4.12/2.2, reinstall the versions you had.

Working files (frames, masks, COLMAP models) go to ComfyUI's temp folder,
keyed by content, so re-running with the same inputs reuses the poses.

## Development

```bash
pip install -r requirements.txt torch pytest ruff
ruff check .
pytest -m "not slow"
```

## License

MIT (see `LICENSE`). Ports gausify-orbit (MIT, © Bryan Diniz); uses Brush
(Apache-2.0) and COLMAP (BSD-3-Clause). See [THIRD_PARTY.md](THIRD_PARTY.md).
