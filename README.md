# Traffic Vision

Vehicle and number-plate detection on the Kaggle
[Traffic vehicles Object Detection](https://www.kaggle.com/datasets/saumyapatel/traffic-vehicles-object-detection)
dataset: `Car`, `Number Plate`, `Blur Number Plate`, `Two Wheeler`, `Auto`, `Bus`, `Truck`.

Two pieces. `notebooks/train_colab.ipynb` trains on Colab and exports `best.onnx`.
`web/` is a static page where you drop a video and boxes are drawn on top.

Inference runs entirely in the browser. No video is uploaded anywhere.

## Training

Open `notebooks/train_colab.ipynb` in Colab, set *Runtime → Change runtime type →
T4 GPU*, upload `archive.zip` and run the cells. About 35 minutes.

The notebook is self-contained: unpack, `data.yaml`, train, per-class mAP,
confusion matrix, ONNX export, download.

ONNX sizes: `yolo11n` 10 MB, `yolo11s` 38 MB, `yolo11m` 80 MB. The front-end
downloads the whole file, so the choice shows up as page load time.

The export uses `nms=False` and `dynamic=False`: onnxruntime-web only partially
covers the NMS operators, so `web/detector.js` does the letterbox, decodes the raw
`[1, 11, 8400]` output and runs per-class NMS in JavaScript.

Drop the resulting `best.onnx` into `web/models/`.

## Current model

`yolo11s`, 100 epochs at 640 px, on the 184 validation images:

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

Trained on 732 images and 9 153 boxes, validated on 184 and 1 980. The archive's
`test` folder carries no annotations, just 267 raw images and 18 videos. Two of
those videos are `web/demo/sample1.mp4` and `sample2.mp4`.

The two plate classes are small objects, seen only by the finest of the three
detection maps. Their mAP responds to `IMGSZ` far more than to `EPOCHS`.

## Front-end

```bash
cd web && python3 -m http.server      # http://localhost:8000
```

The model is loaded on startup. Drop a video, press *Start*. The confidence
threshold is live, classes can be toggled with live counts, and latency, rate and
box count are shown throughout.

**Frames and boxes stay in step.** Inference is much slower than playback, so the
video would run ahead of the boxes by a full inference latency. The canvas shows
the frame the boxes were computed from, and the two swap together. The picture
therefore advances at inference pace, roughly one step per second on CPU.

**Backends.** WebGPU is tried first, then WASM. Each runs a warm-up pass before
being accepted: a backend can create a session and then hang on the first run,
which would leave the page stuck with no error. A hang also locks the runtime, so
recovery reloads the page pinned to WASM.

**A local `python3 -m http.server` is degraded**, in two visible ways: no
isolation headers, so WASM runs single-threaded (1471 ms per frame instead of
737), and no range requests, so the seek bar cannot move. `vercel dev` serves
`web/` with the production headers.

## Deployment

`web/` is a static site with no build step. Point the Vercel project's *Root
Directory* at `web` and push.

`web/vercel.json` sends `Cross-Origin-Opener-Policy` and
`Cross-Origin-Embedder-Policy: credentialless`. Without them the browser does not
expose `SharedArrayBuffer` at all, so WASM cannot use more than one thread. They
are not optional.

`best.onnx` and the sample videos are committed on purpose: a Git deployment can
only serve what the repository contains. Git LFS is a trap here: Vercel clones
without it and would deploy the pointer files instead of the model.

Each visitor downloads the 38 MB model before the first detection.

## Layout

```
notebooks/train_colab.ipynb   training + ONNX export
web/
├── index.html, style.css
├── detector.js               letterbox, decode, NMS
├── app.js                    interface
├── vercel.json               cross-origin isolation headers
├── models/best.onnx
└── demo/                     sample videos
```

Ultralytics YOLO11 is AGPL-3.0.
