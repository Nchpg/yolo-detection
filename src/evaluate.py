"""Report per-class mAP for a trained model on a given split."""
import argparse
import sys
from pathlib import Path

from config import CLASS_NAMES, DATA_YAML, ROOT, RUNS_DIR


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto | cpu | 0 | mps")
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"Weights not found: {args.weights}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    metrics = YOLO(args.weights).val(
        data=str(args.data), split=args.split, imgsz=args.imgsz, batch=args.batch,
        device=pick_device(args.device), plots=True,
        project=str(RUNS_DIR), name=f"eval-{args.split}", exist_ok=True,
    )

    # class_result(i) indexes the classes actually evaluated, in the order of
    # ap_class_index -- not the absolute class id.
    order = {int(c): i for i, c in enumerate(metrics.box.ap_class_index)}

    print(f"\n{'class':<20}{'P':>8}{'R':>8}{'mAP50':>10}{'mAP50-95':>10}")
    for cls_id, name in enumerate(CLASS_NAMES):
        if cls_id in order:
            p, r, ap50, ap = metrics.box.class_result(order[cls_id])
            print(f"{name:<20}{p:>8.3f}{r:>8.3f}{ap50:>10.3f}{ap:>10.3f}")
        else:
            print(f"{name:<20}{'-':>8}{'-':>8}{'-':>10}{'-':>10}")

    print(f"\nmAP50 {metrics.box.map50:.4f}   mAP50-95 {metrics.box.map:.4f}")
    print(f"Plots and confusion matrix: {metrics.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
