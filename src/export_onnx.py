"""Export best.pt to web/models/best.onnx for the browser front-end.

Exported without NMS: onnxruntime-web only partially covers the NMS operators,
so decoding and non-maximum suppression happen in JavaScript (web/detector.js).
"""
import argparse
import shutil
import sys
from pathlib import Path

from config import CLASS_NAMES, ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12, help="12 for wasm compatibility")
    parser.add_argument("--dest", type=Path, default=ROOT / "web" / "models" / "best.onnx")
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"Weights not found: {args.weights}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    exported = Path(YOLO(args.weights).export(
        format="onnx", imgsz=args.imgsz, opset=args.opset,
        simplify=True, dynamic=False, nms=False, half=False,
    ))
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported, args.dest)

    import onnxruntime as ort

    session = ort.InferenceSession(str(args.dest), providers=["CPUExecutionProvider"])
    out = session.get_outputs()[0]
    print(f"\n{args.dest}  ({args.dest.stat().st_size / 1e6:.1f} MB)")
    print(f"input  {session.get_inputs()[0].shape}")
    print(f"output {out.shape}")

    expected = 4 + len(CLASS_NAMES)
    if isinstance(out.shape[1], int) and out.shape[1] != expected:
        print(f"WARNING: {out.shape[1]} output rows, expected {expected} "
              f"(4 box + {len(CLASS_NAMES)} classes).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
