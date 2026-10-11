# Food Recognition Inference Testbench (temporary)

Local experimental harness that chains the two **existing, untouched** checkpoints:

```
upload -> YOLO11s-seg (instances) -> mask-based crop -> [compare mode only] MobileNetV3 (Food-101) -> manual evaluation
```

It exists to answer: *is YOLO's own class prediction reliable enough to be the food identity (`yolo_only` mode), and does the
Food-101 classifier add anything on the regions YOLO extracts (`compare` mode)?*
It is **not** the dietary-monitoring application: no nutrition, load cell, database or chat.

## Modes

| Mode | What runs | Food identity |
|---|---|---|
| `yolo_only` | YOLO segmentation + mask crops. MobileNet is still loaded at startup but never run for the request (no classifier time, no 224x224 preprocessing). | YOLO's class (`yolo_class_name`, FoodSeg103 ingredient taxonomy) |
| `compare` *(default; the original pipeline)* | YOLO + MobileNet on the same masked crops, shown side by side with a comparison table | still YOLO's class; MobileNet's Food-101 dish label is attached as `mobilenet` for comparison |

Pick a mode with the radio buttons in the UI, the optional `mode` form field on `POST /api/inference`, or the server default
`TESTBENCH_DEFAULT_MODE` (`yolo_only` | `compare`; shipped value `compare` so nothing changes unless asked).
Verdict buttons judge each model separately (`model: "yolo" | "mobilenet"`), and the summary adds a YOLO accuracy block (also split
by YOLO confidence band) plus a head-to-head block restricted to segments judged for both models.

## Result schema (per segment)

`segment_id`, `yolo_class_id`, `yolo_class_name`, `yolo_confidence`, `primary_identity` (`source: "yolo"`), `box_xyxy`, `mask_area_px`,
`status` (`identified` in yolo_only, `classified` in compare, `skipped` = crop too small), `image_urls` with `masked`, `bbox`, `mask`
(lossless PNG, white = food, same extents as the crops and `box_xyxy`) and, in compare mode only, `model_input`, plus
`mobilenet` (`null` in yolo_only; otherwise `top1` and `predictions[]`), `feedback` (MobileNet verdict) and `yolo_feedback`.
`id`, `predictions` and `top1` are the original names and are kept for existing clients. Result level: `mode`, `identity_source`.

## Run it

From the repository root (`Electives 2`), using the project venv:

```powershell
.\venv\Scripts\python.exe -m pip install -r testbench\requirements-testbench.txt   # only if fastapi/uvicorn are missing
.\venv\Scripts\python.exe -m testbench                                              # http://127.0.0.1:8000
```

Open <http://127.0.0.1:8000>, drop one or more JPEG/PNG meal photos, review the crops, and mark each prediction
correct / incorrect / ambiguous. API docs: <http://127.0.0.1:8000/api/docs>. Results live in memory only.

Tests:

```powershell
.\venv\Scripts\python.exe -m pytest -c testbench/pytest.ini testbench/tests                 # everything (~25 s with GPU)
.\venv\Scripts\python.exe -m pytest -c testbench/pytest.ini testbench/tests -m "not live"   # no model loading
.\venv\Scripts\python.exe -m testbench.scripts.classifier_sanity_check --per-class 10       # classifier alone on Food-101 val
.\venv\Scripts\python.exe -m testbench.scripts.yolo_reliability_check --limit 300            # YOLO identity vs FoodSeg103 ground truth (0 = whole val split, 2135 images)
```

`yolo_reliability_check` runs the testbench's own segmentation path on FoodSeg103 validation photos and reports schema validity,
localisation (mask IoU >= 0.5, class ignored) separately from class correctness, a confidence-threshold sweep, pixel-area-weighted
class accuracy, same-region/different-class duplicates, top confusions and per-class precision/recall (`--out report.json`).

## Artifacts used (found by audit, nothing copied or modified)

| Role | File | Notes |
|---|---|---|
| Segmenter | `weights/yolo11s_food_best.pt` | byte-identical to `yolo_pipeline/runs/segment/food_seg_11s_v2/weights/best.pt`; task `segment`, 103 FoodSeg103 ingredient classes (names dict also has a harmless `-1: background` key) |
| Classifier | `weights/mobilenet_v3_food101_best.pt` | byte-identical to `mobilenet_pipeline/models/mobilenet_v3_food101_best.pt`; dict with `model_state_dict`, `class_names`, `num_classes=101`, `best_acc_top1=0.8311` |
| Class order | stored inside the classifier checkpoint | cross-checked against `mobilenet_pipeline/models/class_names.json` at startup; startup aborts if they differ |

> **The classifier checkpoint is MobileNetV3-*Large*, not Small.** `train_mobnet.py` builds `mobilenet_v3_large`; the weights
> load with `strict=True` into that architecture and fail for `mobilenet_v3_small`. The loader derives the variant from the
> tensor shapes, so it loads what the checkpoint really is.

## Preprocessing (identical to `train_mobnet.py` `data_transforms['val']`)

`masked RGB crop (outside mask = black) -> LetterboxPad(fill=0) -> Resize(224,224) -> ToTensor -> Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`.
No augmentation. `tests/test_preprocessing.py` imports the original training script and asserts bit-identical tensors.
Differences from training that are *inherent to the experiment*: training images were whole dishes; here the input is a
masked region with a black background (black was also the letterbox fill used in training).

## How a segment is produced

* Image decoded with Pillow, EXIF orientation applied, alpha flattened onto white, longest side capped at 1600 px
  (`TESTBENCH_MAX_IMAGE_SIDE`); that decoded array is the "original" for every later step.
* YOLO is given a **BGR** array (Ultralytics' convention for numpy input), `conf=0.25` (as in `yolo_pipeline/scripts/model_test.py`),
  `iou=0.7`, `imgsz=640`, and `retina_masks=True`, so masks arrive at original-image resolution and need no un-letterboxing.
  `retina_masks` only changes mask resolution, not detections. If a mask ever arrives at another resolution, the polygons
  (already in original coordinates) are rasterized instead of stretching the mask.
* Crop = YOLO box (outward-rounded, clamped) applied to the image with background pixels zeroed by the mask. The unmasked
  box crop and the 224x224 model input are kept for comparison. Crops with an empty mask, <100 mask px, or a side <10 px are
  listed as *skipped* with the reason and never reach the classifier.

## API

| Method & path | Purpose |
|---|---|
| `GET /api/health` | status, model/device info, store usage, limits, class list |
| `POST /api/inference` | multipart field `files` (1..N) + optional `mode` (`yolo_only` / `compare`); returns `items[]` (`completed` / `failed` / `rejected`) + `summary` |
| `GET /api/results` / `GET /api/results/{id}` | summaries / full detail incl. image URLs |
| `GET /api/results/{id}/images/{key}` | `original`, `overlay`, `seg{n}_masked`, `seg{n}_bbox`, `seg{n}_mask` (PNG), `seg{n}_input` (compare mode) |
| `DELETE /api/results` | clear everything |
| `PUT/DELETE /api/results/{id}/segments/{n}/feedback` | manual verdict (`correct` / `incorrect` + optional `actual_label` / `ambiguous`); body field / query param `model` = `mobilenet` (default) or `yolo` *(addition for section 8)* |
| `GET /api/evaluation/summary` | preliminary manual-evaluation metrics *(addition for section 8)* |

Errors are `{"error": {"code", "message"}}`; stack traces stay in the server log.

## Food identity confirmation + USDA nutrition (Expo integration)

Flow: `POST /api/inference` -> `result.regions[]` (duplicate-resolved foods, each `needs_confirmation`) -> user picks an identity ->
USDA reference per 100 g -> optional manual weight -> serving nutrients. Models only *suggest*; no confidence value confirms anything.

**Regions.** Segments whose *masks* overlap with IoU >= `TESTBENCH_DUPLICATE_MASK_IOU` (0.7) are one region (highest-confidence segment is primary).
Box overlap and containment never merge segments. All raw `segments` are still returned; each has `region_id` / `duplicate_of`.

| Endpoint | Purpose |
|---|---|
| `POST /api/inference` (multipart `files`, optional `mode`) | unchanged; `result.regions` is new |
| `GET /api/results/{rid}` | full result incl. `regions` |
| `PUT /api/results/{rid}/regions/{region}/identity` | confirm or correct. Body `{"label","taxonomy":"foodseg103"\|"food101"\|"custom","fdc_id"?,"weight_g"?}` |
| `DELETE /api/results/{rid}/regions/{region}/identity` | back to `needs_confirmation` |
| `PUT /api/results/{rid}/regions/{region}/portion` | `{"weight_g": 150}` (>0, <=5000) or `null` to clear |
| `GET /api/results/{rid}/regions/{region}/nutrition` | USDA reference + serving for one region |
| `GET /api/results/{rid}/nutrition` | meal total over confirmed regions that have a USDA record and a weight (each region counted once) |
| `GET /api/nutrition/foods` | labels that have USDA data |

`nutrition.status`: `needs_confirmation` | `ok` | `needs_usda_selection` (label has several records, e.g. raw/cooked: re-PUT identity with `fdc_id`) | `unmapped` (no USDA record; `per_100g` is null) | `invalid_selection`.
Nutrients USDA did not report are `null`, never 0. `custom` labels are never mapped. Weight is user-entered only (never from mask area).

Example (confirm, choose cooked record, enter weight):

    PUT /api/results/9aa98426c83c/regions/1/identity   {"label":"rice","taxonomy":"foodseg103","fdc_id":168878,"weight_g":150}
    -> {"region_id":1,"status":"confirmed","identity":{"label":"rice","taxonomy":"foodseg103","from_suggestion":true,...},
        "nutrition":{"status":"ok","usda_record":{"fdc_id":168878,"description":"...","data_source":"SR Legacy","preparation":"cooked"},
                     "per_100g":{"energy_kcal":...,"protein_g":...,"carbs_g":...,"fat_g":...},"weight_g":150,"serving":{...}}}

**Populating the USDA cache (one time, needs network + your key; the key stays in the backend environment):**

    $env:USDA_API_KEY = "<your key from api.data.gov>"     # DEMO_KEY is shared and rate-limited
    python -m testbench.scripts.usda_import search "broccoli raw"   # find FDC IDs
    python -m testbench.scripts.usda_import import                  # caches usda_curated.json -> testbench/data/nutrition.sqlite

Each curated FDC ID is checked against USDA's real description (`expect` in `usda_curated.json`); mismatches are skipped, never stored.
Until the import has run, every lookup returns `unmapped`.

## Settings (environment variables, all optional)

`TESTBENCH_DEVICE` (`auto`|`cpu`|`cuda`|`cuda:N`), `TESTBENCH_YOLO_CONF`, `TESTBENCH_YOLO_IOU`, `TESTBENCH_YOLO_IMGSZ`,
`TESTBENCH_DEFAULT_MODE` (`compare`), `TESTBENCH_MAX_SEGMENTS` (40), `TESTBENCH_MAX_UPLOAD_MB` (15), `TESTBENCH_MAX_FILES_PER_REQUEST` (20),
`TESTBENCH_MAX_IMAGE_MEGAPIXELS` (50), `TESTBENCH_MAX_IMAGE_SIDE` (1600), `TESTBENCH_MIN_MASK_PIXELS` (100),
`TESTBENCH_MIN_CROP_SIDE` (10), `TESTBENCH_MAX_RESULTS` (200), `TESTBENCH_MAX_STORE_MB` (512),
`TESTBENCH_HOST`, `TESTBENCH_PORT`, plus `TESTBENCH_YOLO_WEIGHTS`, `TESTBENCH_MOBILENET_WEIGHTS`, `TESTBENCH_CLASS_NAMES_JSON`.

## Layout

```
testbench/
  config.py            settings                      model_loader.py   load + validate checkpoints, device, warm-up
  ingest.py            upload validation/decoding    segmentation.py   YOLO inference, overlay
  crops.py             mask crops + preprocessing    classification.py MobileNet top-k
  preprocessing.py     copy of training val transform orchestrator.py  pipeline, stage timing, record assembly
  records.py           result records                result_store.py  bounded in-memory store
  evaluation.py        manual verdicts + summary     main.py          FastAPI routes + static UI
  static/              index.html, app.js, style.css
  scripts/             classifier_sanity_check.py, yolo_reliability_check.py
  tests/               unit/API tests (fakes for the two networks) + test_live_integration.py (real models)
```

## Timing definitions

`yolo_ms` = predict + masks to CPU; `crop_preprocess_ms` = mask crops + letterbox/normalize; `classify_ms` = MobileNet for all
segments in one batched pass; `inference_ms` = sum of those three; `total_ms` = wall clock including decode, overlay drawing and
JPEG encoding. CUDA is synchronized before and after every timed section. Models get one warm-up pass at startup.

## Known limitations

See the report delivered with this change; the important ones: taxonomy mismatch (ingredient segments vs dish classes),
softmax scores are uncalibrated, GPU softmax values vary ~1e-3 with batch size (TF32), and the first request after
startup can be slower. Everything is single-user, sequential and in memory.

## Plate analysis, total-weight allocation and meal log (Phase 2)

Per result (section F of the UI), all additive to the endpoints above:

| Endpoint | Purpose |
|---|---|
| `PUT/DELETE /api/results/{id}/plate` | calibrate: `rim_points` (3) or `center_x/center_y/radius_px`, optional `image_width/height` (rescaled with `PlateGeometry.scaled_to`) and `plate_diameter_mm` |
| `PUT/DELETE /api/results/{id}/total-weight` | one net weight, `source: "manual_simulation"` (a typed number standing in for the future ESP32 load cell) |
| `PUT /api/results/{id}/regions/{rid}/entity-count` | user-corrected item count (kept apart from detected segments) |
| `PUT /api/results/{id}/regions/{rid}/excluded` | mark a region "not food" (left out of area and weight) |
| `POST /api/results/{id}/regions/manual`, `DELETE .../regions/{rid}` | fallback for a food YOLO missed entirely: the user outlines it (polygon, image px) |
| `GET /api/results/{id}/plate-analysis` | areas, pie data, allocation, USDA nutrition, states, warnings (computed fresh from mask pixels) |
| `POST /api/results/{id}/meal` | finalize into SQLite (`accept_incomplete=true` stores an incomplete meal AS incomplete) |
| `GET /api/meals`, `GET /api/meals/{meal_id}` | saved meals |

States: `allocation_complete` / `allocation_partial` / `allocation_rejected` (`allocation_status` is `null` while inputs are missing), plus
`validation_status: "estimate_unvalidated"` (always: no accuracy margin has been established). A meal is `complete` only when the
allocation is complete AND every food has a selected USDA record. Segmentation area is not volume or mass, and high plate coverage
does not prove every food was detected: a completely missed food is not accounted for.

Backend configuration (env, all optional): `TESTBENCH_OVERLAP_POLICY` (`split`|`priority`), `TESTBENCH_OFF_PLATE_FLAG_FRACTION`,
`TESTBENCH_SHARED_FLAG_FRACTION`, `TESTBENCH_MIN_REGION_PX`, `TESTBENCH_STRICT_COVERAGE=1` (uncertain coverage rejects instead of
warning), `TESTBENCH_FOOD_COEFFICIENTS='{"rice": 1.0}'` (none shipped; only calibrated values belong here), `TESTBENCH_MEALS_DB`
(default `testbench/data/meals.sqlite`, created on first save; `nutrition.sqlite` is never written by the server).

Input rules for the new endpoints (Phase 3 audit): request bodies are validated strictly (no `true`/`"100"` coercion), and
`net_weight_g` must have at most 1 decimal place, because the estimates are rounded to 0.1 g and must add up exactly to the total.

## Camera and automatic plate localization (Phase 4)

Modules: `camera.py` (OpenCV acquisition only), `plate_detection.py` (classical OpenCV circle detection + evidence score),
`camera_calibration.py` (per-camera JSON store, `data/camera_state.json`), `plate_localization.py` (strategy), `camera_api.py` (routes).
The camera is never opened at startup, only when you press Test / Capture / Refresh in the UI (or call the endpoints).

Settings (env): `TESTBENCH_CAMERA_SOURCE` (index or `/dev/videoN`, default `0`), `TESTBENCH_CAMERA_BACKEND` (`auto` = DirectShow on Windows,
V4L2 on Linux), `TESTBENCH_CAMERA_WIDTH/HEIGHT` (requested 1280x720; the actual size is reported), `TESTBENCH_CAMERA_TIMEOUT_S`,
`TESTBENCH_PLATE_LOCALIZATION` = `fixed_calibration` | `automatic_detection` | `automatic_with_fallback` (default),
`TESTBENCH_CALIBRATION_VERIFY` / `TESTBENCH_CALIBRATION_MIN_SUPPORT`, `TESTBENCH_CAMERA_STATE_FILE`.

Endpoints: `GET /api/camera/status|devices|calibration|capture/image`, `POST /api/camera/test|capture|detect-plate|localize`,
`PUT /api/camera/selection|localization-mode|calibration`, `DELETE /api/camera/calibration`, and `POST /api/results/{id}/plate/auto`.
A captured frame is fetched from `/api/camera/capture/image` and posted to the unchanged `/api/inference`. `plate-analysis` now also returns
`plate_source` (`manual` | `saved_calibration` | `automatic_detection`).

A saved calibration is used only for frames from the same camera id, only if resolution/aspect are compatible (it is scaled), and only if the
saved rim still has edge evidence in the new image. Otherwise the detector runs (mode permitting); if that is not `detected` the API answers
`needs_manual` and the 3-point manual calibration is used. `evidence_score` is a heuristic, not a probability.

## Live camera preview (Phase 5)

`camera_preview.py`: while a preview runs, ONE background thread owns the ONE `cv2.VideoCapture` handle. Stills and probes of that camera
are served from its latest full-resolution frame (no second handle, no `camera_busy`); capturing a *different* camera while previewing is
refused with `camera_in_use`. The camera runs at `TESTBENCH_CAMERA_WIDTH x HEIGHT` (default 1920x1080); only the browser copy is downscaled
(`TESTBENCH_PREVIEW_WIDTH`, default 960) and rate-limited (`TESTBENCH_PREVIEW_FPS`, default 10, max 30; JPEG quality `TESTBENCH_PREVIEW_JPEG_QUALITY`).
JPEGs are only encoded while a browser is watching. The camera is released on Stop, on app shutdown, when the camera stops delivering frames,
and `TESTBENCH_PREVIEW_IDLE_TIMEOUT_S` (default 5 s) after the last stream viewer disconnects.

Endpoints: `POST /api/camera/preview/start` (optional `source`, `fps`, `preview_width`), `POST /api/camera/preview/stop`,
`GET /api/camera/preview/status`, `GET /api/camera/preview/stream` (multipart/x-mixed-replace MJPEG). `POST /api/camera/capture` now also
reports `from_preview`. UI: Start/Stop preview, live status, **Capture frame**, and **Capture & analyze** (the normal upload queue: YOLO + automatic plate localization).
Limitation: probing other cameras (Refresh camera list) while a 1080p preview streams shares USB bandwidth and can briefly glitch the stream.
