"""Evalue un modele entraine sur un split donne (val par defaut, ou test)."""
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


def per_class_rows(metrics, class_names):
    """Associe chaque classe a ses metriques.

    metrics.box.class_result(i) s'indexe sur les classes reellement evaluees
    (ordre de metrics.box.ap_class_index), pas sur l'id absolu de la classe.
    """
    order = {int(c): i for i, c in enumerate(metrics.box.ap_class_index)}
    for cls_id, name in enumerate(class_names):
        yield name, metrics.box.class_result(order[cls_id]) if cls_id in order else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"Poids introuvables : {args.weights}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    metrics = YOLO(args.weights).val(
        data=str(args.data), split=args.split, imgsz=args.imgsz, batch=args.batch,
        conf=args.conf, iou=args.iou, device=pick_device(args.device), plots=True,
        project=str(RUNS_DIR), name=f"eval-{args.split}", exist_ok=True,
    )

    print(f"\n--- Resultats sur le split '{args.split}' ---")
    print(f"  {'classe':<20}{'P':>8}{'R':>8}{'mAP50':>10}{'mAP50-95':>10}")
    for name, res in per_class_rows(metrics, CLASS_NAMES):
        if res is None:
            print(f"  {name:<20}{'-':>8}{'-':>8}{'-':>10}{'-':>10}")
        else:
            p, r, ap50, ap = res
            print(f"  {name:<20}{p:>8.3f}{r:>8.3f}{ap50:>10.3f}{ap:>10.3f}")
    print(f"\n  mAP50 : {metrics.box.map50:.4f}   mAP50-95 : {metrics.box.map:.4f}")
    print(f"  Graphiques et matrice de confusion : {metrics.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
