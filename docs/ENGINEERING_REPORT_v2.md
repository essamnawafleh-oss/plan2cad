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

## 7. Benchmark (before / after): summary

Same cases, seeds and harness for both versions; 0 errored runs in each. Full tables are in section 13.

| Group (means) | Wall F1 | IoU | Precision | Recall | False walls / plan | Time (s) | Peak MB |
|---|---|---|---|---|---|---|---|
| Photos, calibrated (6 runs) | 0.831 → **0.973** | 0.721 → **0.906** | 0.788 → 0.953 | 0.937 → 0.994 | 3.67 → 1.00 | 30.0 → 8.4 | 2332 → 1533 |
| Photos, uncalibrated (6) | 0.810 → **0.976** | 0.742 → **0.887** | 0.749 → 0.959 | 0.929 → 0.994 | 4.67 → 0.50 | 32.2 → 11.6 | 2329 → 1533 |
| Scans, calibrated (18) | 0.951 → **0.965** | 0.783 → **0.869** | 0.915 → 0.946 | 0.992 → 0.985 | 1.83 → 1.61 | 11.0 → 9.6 | 1487 → 1490 |
| Scans, uncalibrated (18) | 0.954 → **0.973** | 0.782 → **0.880** | 0.914 → 0.955 | 0.998 → 0.992 | 1.06 → 0.72 | 12.9 → 12.7 | 1486 → 1490 |
| Real plan 1 (hand-traced) | 0.666 → 0.688 | 0.408 → 0.399 | 0.589 → 0.620 | 0.765 → 0.772 | 111 → 103 | 61.3 → 60.7 | 6022 → 6047 |

**Where v2 is worse (measured):**
- **Scans:**
  - Door F1 0.995 → 0.969 (calibrated) and 0.991 → 0.971 (uncalibrated).
  - Junction connectivity 0.98 → 0.95.
  - Recall 0.992 → 0.985.
  - Endpoint error 0.037 → 0.043 m (calibrated).

  The cause is the evidence-gated join: v2 refuses some joins that the old free 0.6 m extension made, so a few wall ends stay open and are reported as `open_gap` instead.
- **Real plan 1:**
  - Thickness error 0.076 → 0.107 m. Thickness is now measured from the drawn band. On this scan that band includes hatching and adjacent lines, while the hand-traced reference uses nominal 0.12 / 0.20 / 0.30 m values.
  - IoU 0.408 → 0.399, essentially unchanged.
- **Uncalibrated scale error:** improved only slightly (0.112 → 0.096 on photos, 0.086 → 0.085 on scans). Door-based estimation is still about ±10%; calibrate when no dimensions are written.

**Preprocessing configurations** (photos, calibrated, 6 runs; same code):

| Config | Wall F1 | IoU | Endpoint err (m) | Time (s) |
|---|---|---|---|---|
| default (perspective auto, lighting auto, denoise auto, deskew) | **0.973** | **0.906** | 0.042 | 9.0 |
| denoise always on | 0.973 | 0.902 | 0.042 | 8.9 |
| lighting off | 0.969 | 0.892 | 0.040 | 24.6 |
| perspective off | 0.915 | 0.808 | 0.101 | 13.1 |
| all preprocessing off | 0.928 | 0.695 | 0.120 | 33.5 |

The defaults are the best measured configuration. Forcing denoise on brings no gain, so it stays automatic. On scans all configurations give the same result, because the automatic steps correctly do nothing on clean scans.

Large image (6000×8000, segmentation only): peak RSS 2249 MB → 1326 MB. Time 6.6 s → 8.4 s, measured while the benchmark was also running, so the timing is not reliable.

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

CI run 42908726589 on branch `v2` (windows-latest): **success**.
- Smoke test ✔
- Model loads ✔ ("model ok")
- Unit tests ✔ ("70 passed, 1 warning in 17.26s")
- PyInstaller exe ✔
- Inno Setup installer ✔
- Artifact uploaded ✔
- Release: skipped by design on `v2`

The installed application was not launched on a real Windows desktop from here.

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

# Benchmark comparison

Files: `bench_results/baseline.json`, `bench_results/v2.json`

Runs with errors: baseline: 0, v2: 0

## photo drawings, calibrated

| Metric | baseline | v2 |
|---|---|---|
| runs | 6 | 6 |
| Wall F1 | 0.831 | 0.973 |
| Precision | 0.788 | 0.953 |
| Recall | 0.937 | 0.994 |
| IoU | 0.721 | 0.906 |
| Endpoint err (m) | 0.119 | 0.042 |
| Thickness err (m) | 0.039 | 0.000 |
| Junctions connected | 0.89 | 0.97 |
| Missing walls | 0.50 | 0.00 |
| False walls | 3.67 | 1.00 |
| Door F1 | 0.819 | 0.987 |
| Window F1 | 0.955 | 1.000 |
| Pipeline time (s) | 30.0 | 8.4 |
| NN time (s) | 0.4 | 0.3 |
| Import time (s) | 1.41 | 1.08 |
| DXF time (s) | 0.43 | 0.20 |
| Peak memory (MB) | 2332 | 1533 |

## photo drawings, uncalibrated

| Metric | baseline | v2 |
|---|---|---|
| runs | 6 | 6 |
| Wall F1 | 0.810 | 0.976 |
| Precision | 0.749 | 0.959 |
| Recall | 0.929 | 0.994 |
| IoU | 0.742 | 0.887 |
| Endpoint err (m) | 0.120 | 0.040 |
| Thickness err (m) | 0.037 | 0.021 |
| Junctions connected | 0.90 | 0.97 |
| Missing walls | 0.50 | 0.00 |
| False walls | 4.67 | 0.50 |
| Door F1 | 0.901 | 0.957 |
| Window F1 | 0.921 | 1.000 |
| Pipeline time (s) | 32.2 | 11.6 |
| NN time (s) | 3.4 | 3.2 |
| Import time (s) | 1.40 | 1.06 |
| DXF time (s) | 0.41 | 0.19 |
| Peak memory (MB) | 2329 | 1533 |
| Scale error (rel.) | 0.112 | 0.096 |

## real drawings, calibrated

| Metric | baseline | v2 |
|---|---|---|
| runs | 1 | 1 |
| Wall F1 | 0.666 | 0.688 |
| Precision | 0.589 | 0.620 |
| Recall | 0.765 | 0.772 |
| IoU | 0.408 | 0.399 |
| Endpoint err (m) | 0.296 | 0.302 |
| Thickness err (m) | 0.076 | 0.107 |
| Junctions connected | 0.65 | 0.65 |
| Missing walls | 38.00 | 34.00 |
| False walls | 111.00 | 103.00 |
| Pipeline time (s) | 61.3 | 60.7 |
| NN time (s) | 1.4 | 1.5 |
| Import time (s) | 0.34 | 0.33 |
| DXF time (s) | 0.76 | 0.67 |
| Peak memory (MB) | 6022 | 6047 |

## scan drawings, calibrated

| Metric | baseline | v2 |
|---|---|---|
| runs | 18 | 18 |
| Wall F1 | 0.951 | 0.965 |
| Precision | 0.915 | 0.946 |
| Recall | 0.992 | 0.985 |
| IoU | 0.783 | 0.869 |
| Endpoint err (m) | 0.037 | 0.043 |
| Thickness err (m) | 0.052 | 0.016 |
| Junctions connected | 0.98 | 0.95 |
| Missing walls | 0.00 | 0.06 |
| False walls | 1.83 | 1.61 |
| Door F1 | 0.995 | 0.969 |
| Window F1 | 0.962 | 0.978 |
| Pipeline time (s) | 11.0 | 9.6 |
| NN time (s) | 0.3 | 0.3 |
| Import time (s) | 0.11 | 0.14 |
| DXF time (s) | 0.22 | 0.20 |
| Peak memory (MB) | 1487 | 1490 |

## scan drawings, uncalibrated

| Metric | baseline | v2 |
|---|---|---|
| runs | 18 | 18 |
| Wall F1 | 0.954 | 0.973 |
| Precision | 0.914 | 0.955 |
| Recall | 0.998 | 0.992 |
| IoU | 0.782 | 0.880 |
| Endpoint err (m) | 0.040 | 0.039 |
| Thickness err (m) | 0.059 | 0.022 |
| Junctions connected | 0.99 | 0.97 |
| Missing walls | 0.00 | 0.00 |
| False walls | 1.06 | 0.72 |
| Door F1 | 0.991 | 0.971 |
| Window F1 | 0.938 | 0.970 |
| Pipeline time (s) | 12.9 | 12.7 |
| NN time (s) | 2.6 | 2.8 |
| Import time (s) | 0.12 | 0.15 |
| DXF time (s) | 0.21 | 0.21 |
| Peak memory (MB) | 1486 | 1490 |
| Scale error (rel.) | 0.086 | 0.085 |

## Per case (calibrated, wall F1 / IoU)

| Case | seed | baseline | v2 |
|---|---|---|---|
| angled | 1 | 0.944 / 0.796 | 0.962 / 0.886 |
| angled | 2 | 0.968 / 0.834 | 0.973 / 0.923 |
| clean_filled | 1 | 0.943 / 0.780 | 0.957 / 0.872 |
| clean_filled | 2 | 0.977 / 0.848 | 0.983 / 0.947 |
| clean_hatched | 1 | 0.926 / 0.748 | 0.952 / 0.776 |
| clean_hatched | 2 | 0.972 / 0.797 | 0.983 / 0.851 |
| clean_hollow | 1 | 0.900 / 0.681 | 0.933 / 0.774 |
| clean_hollow | 2 | 0.961 / 0.806 | 0.983 / 0.867 |
| colour_cad | 1 | 0.914 / 0.773 | 0.954 / 0.817 |
| colour_cad | 2 | 0.972 / 0.796 | 0.983 / 0.864 |
| photo_persp | 1 | 0.801 / 0.490 | 0.960 / 0.824 |
| photo_persp | 2 | 0.492 / 0.538 | 0.984 / 0.926 |
| photo_rot90 | 1 | 0.947 / 0.802 | 0.960 / 0.891 |
| photo_rot90 | 2 | 0.983 / 0.818 | 0.984 / 0.949 |
| photo_shadow | 1 | 0.953 / 0.820 | 0.965 / 0.911 |
| photo_shadow | 2 | 0.812 / 0.857 | 0.984 / 0.935 |
| real:plan1 | 0 | 0.666 / 0.408 | 0.688 / 0.399 |
| scan_lowres | 1 | 0.945 / 0.822 | 0.959 / 0.893 |
| scan_lowres | 2 | 0.978 / 0.827 | 0.984 / 0.884 |
| scan_noisy | 1 | 0.934 / 0.757 | 0.957 / 0.886 |
| scan_noisy | 2 | 0.978 / 0.794 | 0.984 / 0.962 |
| scan_rot3 | 1 | 0.950 / 0.827 | 0.964 / 0.887 |
| scan_rot3 | 2 | 0.978 / 0.859 | 0.965 / 0.910 |
| thin_walls | 1 | 0.952 / 0.676 | 0.961 / 0.826 |
| thin_walls | 2 | 0.928 / 0.666 | 0.928 / 0.826 |


### Preprocessing configurations

# Benchmark comparison

Files: `bench_results/pp_default.json`, `bench_results/pp_denoise_on.json`, `bench_results/pp_minimal.json`, `bench_results/pp_no_light.json`, `bench_results/pp_no_persp.json`

Runs with errors: pp_default: 0, pp_denoise_on: 0, pp_minimal: 0, pp_no_light: 0, pp_no_persp: 0

## photo drawings, calibrated

| Metric | pp_default | pp_denoise_on | pp_minimal | pp_no_light | pp_no_persp |
|---|---|---|---|---|---|
| runs | 6 | 6 | 6 | 6 | 6 |
| Wall F1 | 0.973 | 0.973 | 0.928 | 0.969 | 0.915 |
| Precision | 0.953 | 0.953 | 0.898 | 0.949 | 0.910 |
| Recall | 0.994 | 0.994 | 0.961 | 0.990 | 0.935 |
| IoU | 0.906 | 0.902 | 0.695 | 0.892 | 0.808 |
| Endpoint err (m) | 0.042 | 0.042 | 0.120 | 0.040 | 0.101 |
| Thickness err (m) | 0.000 | 0.000 | 0.015 | 0.009 | 0.004 |
| Junctions connected | 0.97 | 0.97 | 0.88 | 0.97 | 0.89 |
| Missing walls | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 |
| False walls | 1.00 | 1.00 | 2.33 | 1.33 | 2.17 |
| Door F1 | 0.987 | 0.974 | 0.827 | 0.972 | 0.954 |
| Window F1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.937 |
| Pipeline time (s) | 9.0 | 8.9 | 33.5 | 24.6 | 13.1 |
| NN time (s) | 0.3 | 0.3 | 0.4 | 0.3 | 0.4 |
| Import time (s) | 1.04 | 1.12 | 0.01 | 0.94 | 1.60 |
| DXF time (s) | 0.20 | 0.20 | 0.37 | 0.36 | 0.23 |
| Peak memory (MB) | 1536 | 1538 | 2097 | 2158 | 1882 |

## scan drawings, calibrated

| Metric | pp_default | pp_denoise_on | pp_minimal | pp_no_light | pp_no_persp |
|---|---|---|---|---|---|
| runs | 2 | 2 | 2 | 2 | 2 |
| Wall F1 | 0.970 | 0.970 | 0.970 | 0.970 | 0.970 |
| Precision | 0.948 | 0.947 | 0.948 | 0.948 | 0.948 |
| Recall | 0.994 | 0.994 | 0.994 | 0.994 | 0.994 |
| IoU | 0.924 | 0.914 | 0.924 | 0.924 | 0.924 |
| Endpoint err (m) | 0.035 | 0.035 | 0.035 | 0.035 | 0.035 |
| Thickness err (m) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Junctions connected | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 |
| Missing walls | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| False walls | 1.50 | 1.50 | 1.50 | 1.50 | 1.50 |
| Door F1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Window F1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Pipeline time (s) | 8.1 | 8.4 | 8.3 | 8.0 | 7.9 |
| NN time (s) | 0.3 | 0.3 | 0.3 | 0.3 | 0.2 |
| Import time (s) | 0.22 | 0.44 | 0.01 | 0.16 | 0.19 |
| DXF time (s) | 0.20 | 0.19 | 0.19 | 0.19 | 0.18 |
| Peak memory (MB) | 1431 | 1434 | 1426 | 1427 | 1431 |

## Per case (calibrated, wall F1 / IoU)

| Case | seed | pp_default | pp_denoise_on | pp_minimal | pp_no_light | pp_no_persp |
|---|---|---|---|---|---|---|
| photo_persp | 1 | 0.960 / 0.824 | 0.960 / 0.824 | 0.832 / 0.533 | 0.955 / 0.846 | 0.801 / 0.535 |
| photo_persp | 2 | 0.984 / 0.926 | 0.984 / 0.926 | 0.882 / 0.596 | 0.984 / 0.926 | 0.794 / 0.629 |
| photo_rot90 | 1 | 0.960 / 0.891 | 0.960 / 0.872 | 0.960 / 0.871 | 0.960 / 0.871 | 0.960 / 0.891 |
| photo_rot90 | 2 | 0.984 / 0.949 | 0.984 / 0.945 | 0.983 / 0.950 | 0.983 / 0.950 | 0.984 / 0.949 |
| photo_shadow | 1 | 0.965 / 0.911 | 0.965 / 0.911 | 0.936 / 0.588 | 0.968 / 0.857 | 0.965 / 0.911 |
| photo_shadow | 2 | 0.984 / 0.935 | 0.984 / 0.935 | 0.975 / 0.632 | 0.966 / 0.902 | 0.984 / 0.935 |
| scan_noisy | 1 | 0.957 / 0.886 | 0.956 / 0.877 | 0.957 / 0.886 | 0.957 / 0.886 | 0.957 / 0.886 |
| scan_noisy | 2 | 0.984 / 0.962 | 0.983 / 0.951 | 0.984 / 0.962 | 0.984 / 0.962 | 0.984 / 0.962 |
