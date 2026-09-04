# Traffic Vision

Vehicle and number-plate detection on the Kaggle
[Traffic vehicles Object Detection](https://www.kaggle.com/datasets/saumyapatel/traffic-vehicles-object-detection)
dataset: `Car`, `Number Plate`, `Blur Number Plate`, `Two Wheeler`, `Auto`, `Bus`, `Truck`.

Training runs on Colab. Inference runs in the browser — drop a video on a static
page and boxes are drawn on top. Nothing is uploaded anywhere.

## Usage

```bash
uv sync
uv run python src/extract_archive.py     # archive.zip -> data/raw
uv run python src/prepare_dataset.py     # data.yaml + annotation checks
# -> run notebooks/train_colab.ipynb on Colab (T4 GPU, ~35 min)
# -> put best.onnx in web/models/, best.pt at the repo root
uv run python web/serve.py               # http://localhost:8000
```

## Current model

`yolo11s`, 100 epochs at 640 px, measured on the 184 validation images:

| class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Car | 0.899 | 0.919 | 0.947 | 0.783 |
| Two Wheeler | 0.816 | 0.867 | 0.895 | 0.640 |
| Number Plate | 0.746 | 0.828 | 0.878 | 0.547 |
| Bus | 0.686 | 0.815 | 0.842 | 0.642 |
| Truck | 0.794 | 0.743 | 0.820 | 0.620 |
| Auto | 0.688 | 0.680 | 0.714 | 0.387 |
| Blur Number Plate | 0.617 | 0.764 | 0.719 | 0.383 |
| **all** | | | **0.831** | **0.572** |

```bash
uv run python src/evaluate.py --split val
```

## Dataset

The archive ships its own split. The `test` folder has no annotations — 267 raw
images and 18 videos — so it is set aside in `data/demo/` and two of those videos
are used as samples in the front-end.

| split | images | boxes |
|---|---|---|
| train | 732 | 9 153 |
| val | 184 | 1 980 |

`prepare_dataset.py` pairs each image with its `.txt` and validates every
annotation (field count, class id, `[0, 1]` bounds, degenerate boxes). Problems
are clipped or dropped and logged to `data/dataset/label_issues.log`.

## Front-end

`web/models/best.onnx` is loaded on startup. Drop a video, press *Start*.

The confidence threshold is live, classes can be toggled with live counts, and
latency, rate and box count are shown throughout. The NMS threshold is fixed at
the standard 0.45.

Inference is much slower than playback, so the video would otherwise run ahead of
the boxes by a full inference latency. The canvas therefore shows the frame the
boxes were computed from: picture and boxes swap together and never disagree, at
the cost of a picture that advances at inference pace.

Since that overlay hides the native video controls, the transport bar carries its
own seek slider. Seeking re-runs inference on the frame you land on.

The model is exported without NMS — onnxruntime-web only partially covers the NMS
operators — so `web/detector.js` does the letterbox, decodes the raw
`[1, 11, 8400]` output and runs per-class NMS in JavaScript.

WebGPU is tried first, falling back to WASM. `serve.py` sends
`Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy: credentialless`,
which unlocks multithreaded WASM; plain `python -m http.server` also works, but
single-threaded.

The 38 MB model takes ~1.2 s per frame on WASM/CPU, 30-60 ms on WebGPU. Each
visitor downloads it in full before the first detection.

`web/` is a static site with no build step, and `web/vercel.json` carries the
isolation headers. Point the Vercel project's *Root Directory* at `web` and push;
`.vercelignore` keeps `serve.py` out.

`best.onnx` and the sample videos are committed on purpose: a Git deployment can
only serve what the repository contains. Do not move them to Git LFS — Vercel
clones without LFS and would deploy the pointer files instead of the model.

After deploying, check the model actually landed:

```bash
curl -sI https://<your-domain>/models/best.onnx | head -1   # expect 200
```

## Layout

```
best.pt                       PyTorch weights (evaluation, re-export)
notebooks/train_colab.ipynb   training + ONNX export
src/
├── config.py                 classes and paths
├── extract_archive.py        archive.zip -> data/raw
├── prepare_dataset.py        data.yaml + annotation validation
├── evaluate.py               per-class mAP
└── export_onnx.py            best.pt -> web/models/best.onnx
web/
├── index.html, style.css
├── detector.js               letterbox, decode, NMS
├── app.js                    interface
├── serve.py                  local static server (cross-origin isolation, ranges)
├── vercel.json               the same headers, for deployment
├── models/best.onnx
└── demo/                     sample videos
```

## Notes

- `override-dependencies` in `pyproject.toml` drops `opencv-python` (GUI build,
  needs `libxcb`/`libGL`) in favour of `opencv-python-headless`, which provides
  the same `cv2` module. Required on NixOS and in containers without X11.
- Torch is installed CPU-only. For a CUDA machine, swap the `pytorch-cpu` index
  for `https://download.pytorch.org/whl/cu124`.
- Ultralytics YOLO11 is AGPL-3.0.
