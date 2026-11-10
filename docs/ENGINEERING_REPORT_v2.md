# Plan2CAD v2: engineering audit and upgrade report

Branch `v2`. The baseline is the code on `main` at commit `a85325e` (release v1.0.6).

All numbers in this report come from runs that were executed. The commands to reproduce them are in section 12.

---

## 1. Architecture and pipeline (as found)

```
image file
  └─ MainWindow.load_image ─ imageio.imread_gray / wall_view (colour) ─ deskew ─ clean_background
       └─ working image (temp PNG) + Project(scale)
DetectWorker (QThread) ─ pipeline.run(project, gray, params)
  1. trace (quick) → dominant angles
  2. dims.analyse: RapidOCR in every direction → tick search → scale vote → 2nd pass → clutter mask
  3. trace (x3 upsample, rule engine) + trace (x2, A-TRACE layer)
  4. walls: neural.segment (U-Net ONNX) → walls_from_prob → mask_walls vectoriser
            (+ rule solid walls not covered), join → prune → page filters → keep_building
  5. complete.complete_walls (missing walls at dimension ends), wall labels
  6. columns, doors (network openings + rule doors), windows (network)
  7. fit.fit_to_dimensions, chains.windows_from_chains, fit again
  8. (optional) furniture
Project (JSON .p2c) ─ dxf_export.export_dxf (layers, blocks, aligned dims, image underlay)
```

Model: a U-Net (channels 16-192), 4 classes (bg, wall, door, window), trained offline on CubiCasa5K with scan-like augmentation. It is exported to ONNX (opset 17) and run with onnxruntime on the CPU.
- **Input**: 1×1×H×W float32 = 1 − gray/255, resized so that a 0.18 m wall becomes about 7 px.
- **Inference**: tiles of 768 px with a 64 px overlap, softmax probabilities averaged over the overlaps.
- **Validation IoU** (CubiCasa val split, step 5950 of training): wall 0.58, door 0.39, window 0.56.
- **Limits**: the model was not validated independently on a labelled set of real scans or photos, because none exists. CubiCasa5K is CC BY-NC, so the model is for **personal / non-commercial use only**.

## 2. Confirmed bugs (baseline)

Each bug has a regression test in `tests/test_regressions.py`. **All 8 tests fail on the baseline** (`P2C_CODE=<baseline> pytest tests/test_regressions.py` → 8 failed) and pass on v2.

| # | File / function | Problem | Reproduction | Severity | Root cause | Fix |
|---|---|---|---|---|---|---|
| B1 | `cli.py` main | Every project was marked *calibrated*, even without `--scale`. The scale from the dimensions or the door estimate was then never used. | `python cli.py plan.jpg` → scale stays 0.03 | Major (wrong geometry) | `scale_calibrated=True` was hard-coded | `--scale` defaults to None; the project counts as calibrated only when a scale is given; the CLI prints the source of the scale |
| B2 | `ui/i18n.py` | The progress dialog showed raw keys `p_nn` and `p_scale` | Run detection | Minor | Labels missing for two pipeline stages | Labels added; the test checks every `say(...)` key |
| B3 | `model.Project.to_dict` | `fit_info`, `chain_info` and `completed` were silently lost on save | save → load | Major (data loss) | Fields not serialised | Serialised, numpy values converted |
| B4 | `model.Project.from_dict` | A file containing any unknown element key crashed with `TypeError` | Load a file written by a newer version | Major | `C(**o)` with unfiltered keys | Keys filtered to the dataclass fields |
| B5 | `engine/neural.segment` | The resize factor `7/(0.18/scale)` had no limit. A wrong scale (e.g. 0.2 m/px on a 2000 px image) requests a ~15 600 px square, i.e. several GB of float32 maps. | `segment(img, 0.2)` (computed: 15 556² × 4 classes × 4 B ≈ 3.9 GB, plus copies) | Critical (crash / out of memory) | No bounds | `input_factor()` clamps the factor to 0.12 to 3.0 and the longest network side to 6000 px; maps are returned as float16 |
| B6 | `dxf_export.export_dxf` | With `image_size=None` the flip height was 0, so every y became negative and the plan was mirrored | API call without an image size | Major | `H = 0` fallback | Height taken from the geometry's lowest point |
| B7 | `engine/pipeline.run` | The 4×H×W probability maps stayed attached to the project, and the UI worker kept that project alive, so hundreds of MB remained allocated after detection | Detect on a large image and watch memory | Major (memory) | `project._nn_prob` was never cleared | Cleared at the end of `run` |
| B8 | `ui/main_window.load_image` | A scale typed in the import dialog was saved without `scale_calibrated`, so the automatic estimate replaced it silently | Type a scale, then detect | Major | No calibration flag | "I know the scale" checkbox; the value is kept as calibrated |

**Checked and *not* bugs** (evidence in brackets):
- **EXIF orientation**: OpenCV's `imdecode` applies it [test `test_exif_orientation_applied`].
- **`clean_background` and thick filled walls**: it does not lighten them. 100% of wall pixels stay dark at 0.02, 0.01 and 0.006 m/px, thanks to its `max(bg, 200)` floor. The same floor does stop it from correcting strong shadows, which the new lighting step handles.

**Design weaknesses found with the benchmark** (not crashes, but they produce wrong geometry):
- **W1 (`detect.join_walls`)**: any free wall end was extended up to 0.6 m into the nearest wall without looking at the image. Furniture outlines therefore became walls attached to real walls.
- **W2 (network vectoriser)**: the centre line and thickness came only from the network mask. On thick walls the mask covers part of the band, so the centre lines sat on the inner face.
- **W3 (`dims.parse_value`)**: integers (cm / mm drawings) were always rejected, and comma decimals worked only by accident of the regex.
- **W4 (`fit`)**: a fixed 0.20 m pre-filter, unweighted least squares, and no record of why a dimension was dropped.

## 3. Files modified and modules created

New modules:
- `p2c/engine/preprocess.py`: orientation, page detection with perspective rectification (Zhang–He aspect recovery), deskew, flat-field lighting, paper level, scanner-border removal, noise estimate with non-local-means denoising. Every geometric step is a recorded 3×3 matrix.
- `p2c/engine/workflow.py`: one import path for the UI, the CLI and the benchmark.
- `p2c/engine/walls.py`: measures each wall from image cross-sections (centre, thickness, confidence), joins walls only when the gap shows evidence, logs every correction.
- `p2c/engine/validate.py`: validation with critical / major / minor / info severities, individual fixes that can be undone, and automatic fixes limited to safe cases.
- `bench/` (synthetic ground truth, runner, report) and `tests/` (70 tests).

Modified:
- `pipeline.py`: evidence stage, cancellation, validation, lazy rule trace, memory release.
- `neural.py`: bounded input, float16 maps, bounded-cost scale estimation, confidence on openings.
- `dims.py`: m/cm/mm readings, vote over readings, robust MAD calibration, status per dimension, Arabic-Indic digits.
- `fit.py`: confidence-weighted Huber IRLS, reason for every rejection, capped shifts.
- `model.py`: `conf` / `origin` on elements, unit / status on dimensions, persisted validation, corrections and transform; tolerant loading; new keys written only when used.
- `dxf_export.py`: real ARC entities for door swings, `P2C_TEXT` style (Arial), dimension text style, `verify_dxf()`.
- `geometry.py`: invalid wall outline repaired with `buffer(0)`.
- UI: import options (perspective, auto-orientation, denoise, "I know the scale"), original/processed comparison, validation report (show / fix / undo), origin and confidence overlay, cancellable detection with clear errors, DXF check after export, manual edits marked `manual`.
- `cli.py`: B1, plus the new import path.
- CI: runs a model-loading check and the pytest suite on Windows before building; `v2` builds an artifact, and only `main` publishes a release.

## 4. New dependencies

No new runtime dependency. `pytest` is used for tests only (`requirements-dev.txt`). Pillow is used by one optional test and is skipped when missing.

## 5. What each change is for

| Change | Why |
|---|---|
| Perspective rectification, only when a clear page quad with contrast ≥ 25 is found | Phone photos were the weakest case (photo_persp F1 0.80, IoU 0.49 at baseline). Without a reliable boundary nothing is warped, as the brief requires. |
| Aspect from the camera model, not edge lengths | Edge lengths gave a 2.8% aspect error; the camera model's mean error is 0.7% over 16 simulated photos (worst 8% at the strongest tilt). |
| Noise estimate from local std on flat paper | The Laplacian (Immerkær) estimator read about 0 on JPEG-compressed noise; the new one reads 4.9 against a true 5.2. |
| Wall cross-section measurement | Fixes W2: thick walls get their real centre and thickness; different walls keep different thicknesses (`test_different_walls_keep_different_thickness`). |
| Joins only with evidence | Fixes W1: a gap of more than 12 cm is closed only if ink or network probability is present in it. |
| Confidence and origin on each element | The user can see what is detected, auto-corrected, uncertain or manual, and review before export. |
| Validation with safe auto-fix only | Duplicates, zero-length and invalid entities are removed and logged. Everything else (gaps, thickness, isolated walls) is reported and fixed only on request, with undo. |
| Unit-aware dimensions and robust calibration | Integer cm/mm drawings become usable. When the unit is ambiguous the program reports it instead of guessing; misreads are rejected with a reason. |
| Huber IRLS fit | One wrong dimension can no longer pull the drawing (`test_fit_moves_wall_to_match_and_rejects_wrong_dimension`). |
| DXF arcs, text style, `verify_dxf` | Editable swings in AutoCAD; Arabic text renders with Arial; every export is re-opened and audited. |
| Lazy rule trace, vectorised validation | NN pipeline on plan 1 went from 18.8 s to 11.3 s with the same 205 walls. |

## 6. Tests

`python -m pytest -q tests` → **70 passed** (Linux; the Windows CI runs the same suite).

| File | Covers |
|---|---|
| `test_regressions.py` (8) | B1 to B8 |
| `test_preprocess.py` (13) | Clean scan untouched, rotated scan, perspective photo (angles and aspect), no warp without a page edge, mapping round trip, uneven lighting, noise, noise-free image not denoised, low resolution, exact manual rotation, size limit, EXIF |
| `test_dims_fit.py` (16) | Decimal / comma / explicit units / Arabic digits, integer ambiguity, unit vote, integer-only ambiguity flag, misread and unit-error rejection, fit with a wrong dimension, shift cap, dimension without walls |
| `test_geometry.py` (21) | Thickness and centre for filled and hollow walls at 0.10/0.20/0.30 m, angled walls, text and furniture beside a wall, single lines rejected, different thicknesses kept, dropping walls without evidence, evidence-gated joins, tiny gaps, L/T/X junctions, overshoot trimming, angled walls not forced to 90°, validation detect-and-fix, gap fix, overlap merge |
| `test_dxf.py` (6) | Units, layers, styles, blocks, entity counts, metre and y-up coordinates, arc against drawn swing, no duplicate outline, verifier catches a broken file |
| `test_compat.py` (3) | v1.0.x files load; files without new features keep the old shape; new fields round-trip |
| `test_ui_smoke.py` (3) | Offscreen import → detect → validation fix + undo → overlay → manual-edit marking → export + verify → save/load; cancellation leaves the drawing unchanged; phone photo rectified with comparison dialog |

The original `e2e.py` UI script also passes against v2 ("E2E OK").

## 7. Benchmark (before / after)

See section 13 (generated tables) and `bench_results/`.

## 8. Known limitations

- Synthetic plans are much easier than real scans; F1 on synthetic plans is already high at baseline. Real-plan accuracy is measured on **one** hand-traced plan only (plan 1). There is no labelled set of real phone photos, so "photo" results are simulated (pinhole camera, lighting, noise, JPEG).
- Automatic 90° orientation uses OCR text direction. It is optional (off by default) because it is slow, and it does nothing on drawings with little text.
- Rectification needs a visible page edge against a darker background. A photo cropped inside the sheet is only deskewed, not rectified.
- Furniture drawn with double parallel lines spaced like a wall can still be accepted as a wall; such walls usually get low confidence and show as *uncertain*.
- Door hinge and swing are a guess when no arc is visible; those doors are marked uncertain.
- The model was not retrained. Retraining needs a labelled set of the target drawings (see section 10).
- Windows build: CI runs the tests and builds the installer on Windows (see section 11). Opening files in AutoCAD itself was not possible here; DXF validity is checked with ezdxf's reader and auditor.

## 9. Remaining issues

- Plan 1 precision is still limited by site lines and page frames that the network labels as walls. They are now flagged *uncertain*, but not removed.
- The rule-based fallback (no model) is slower and noisier than the network path.
- `swing_conf` on doors is not persisted.

## 10. Retraining proposal (not executed)

- **Data**: 300+ real scans and phone photos of the user's drawing types, labelled with wall / door / window polygons (CubiCasa-style YOLO polygon text files, the same converter as `cc5k/prep.py`).
- **Strategy**: fine-tune the current U-Net from its weights, keeping the scan augmentation, and add perspective and lighting augmentation for photos.
- **Evaluation**: a held-out 20% of real drawings, using the metrics of this benchmark (wall F1, IoU, endpoint and thickness error, junction connectivity) and not pixel IoU alone.
- **Rule**: accept the new model only if it beats the current one on the held-out set.

## 11. Windows build verification

See the CI run on branch `v2` (status recorded in the final message). It runs the model-loading check, the pytest suite on windows-latest, PyInstaller and Inno Setup.

## 12. Reproducing the benchmark

```
pip install -r requirements.txt -r requirements-dev.txt
git worktree add ../p2c_baseline a85325e          # the baseline
BASELINE=../p2c_baseline sh bench/run_all.sh       # optional: P2C_REAL_DIR=<dir with work.png + wall_data.py>
cat bench_results/REPORT_baseline_vs_v2.md bench_results/REPORT_preprocessing.md
```

Cases (`bench/synth.py`):
- **Drawing styles**: `clean_filled`, `clean_hatched`, `clean_hollow`, `colour_cad`, `angled`, `thin_walls`.
- **Scans**: `scan_rot3`, `scan_noisy`, `scan_lowres`.
- **Photos**: `photo_persp`, `photo_shadow`, `photo_rot90`.

Each case runs with seeds 1 and 2, calibrated and uncalibrated.

## 13. Benchmark results
