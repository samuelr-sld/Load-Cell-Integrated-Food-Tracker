# Food Recognition Inference Testbench (temporary)

Local experimental harness that chains the two **existing, untouched** checkpoints:

```
upload -> YOLO11s-seg (instances) -> mask-based crop -> MobileNetV3 (Food-101) -> top-5 + manual evaluation
```

It exists to answer one question: *can the Food-101 classifier correctly label the regions YOLO extracts?*
It is **not** the dietary-monitoring application: no nutrition, load cell, database or chat.

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
```

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
| `POST /api/inference` | multipart field `files` (1..N); returns `items[]` (`completed` / `failed` / `rejected`) + `summary` |
| `GET /api/results` / `GET /api/results/{id}` | summaries / full detail incl. image URLs |
| `GET /api/results/{id}/images/{key}` | `original`, `overlay`, `seg{n}_masked`, `seg{n}_bbox`, `seg{n}_input` |
| `DELETE /api/results` | clear everything |
| `PUT/DELETE /api/results/{id}/segments/{n}/feedback` | manual verdict (`correct` / `incorrect` + optional `actual_label` / `ambiguous`) *(addition for section 8)* |
| `GET /api/evaluation/summary` | preliminary manual-evaluation metrics *(addition for section 8)* |

Errors are `{"error": {"code", "message"}}`; stack traces stay in the server log.

## Settings (environment variables, all optional)

`TESTBENCH_DEVICE` (`auto`|`cpu`|`cuda`|`cuda:N`), `TESTBENCH_YOLO_CONF`, `TESTBENCH_YOLO_IOU`, `TESTBENCH_YOLO_IMGSZ`,
`TESTBENCH_MAX_SEGMENTS` (40), `TESTBENCH_MAX_UPLOAD_MB` (15), `TESTBENCH_MAX_FILES_PER_REQUEST` (20),
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
  scripts/             classifier_sanity_check.py
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
