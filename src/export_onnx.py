"""Exporte un best.pt en ONNX pour le front (web/models/best.onnx).

L'export se fait sans NMS : onnxruntime-web ne couvre que partiellement les
operateurs de NMS, le front fait donc lui-meme le decodage et la suppression
des non-maxima (voir web/app.js).
"""
import argparse
import shutil
import sys
from pathlib import Path

from config import CLASS_NAMES, ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dest", type=Path, default=ROOT / "web" / "models" / "best.onnx")
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"Poids introuvables : {args.weights}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    path = Path(YOLO(args.weights).export(
        format="onnx", imgsz=args.imgsz, opset=args.opset,
        simplify=True, dynamic=False, nms=False, half=False,
    ))

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, args.dest)
    print(f"\nONNX : {args.dest}  ({args.dest.stat().st_size / 1e6:.1f} Mo)")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.dest), providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    print(f"entree  {inp.name} {inp.shape}")
    print(f"sortie  {out.name} {out.shape}")
    expected = 4 + len(CLASS_NAMES)
    if isinstance(out.shape[1], int) and out.shape[1] != expected:
        print(f"ATTENTION : {out.shape[1]} lignes en sortie, {expected} attendues "
              f"(4 + {len(CLASS_NAMES)} classes).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
