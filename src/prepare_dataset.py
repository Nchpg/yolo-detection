"""Construit une arborescence YOLO exploitable par Ultralytics a partir de data/raw.

L'archive Kaggle fournit deja un decoupage :
  Traffic Dataset/images/{train,val,test}  +  Traffic Dataset/labels/{train,val}
Le dossier `test` ne contient aucune annotation (images brutes + 18 videos) : il
est traite comme un jeu de demonstration, pas comme un split evalue.

Ce script :
  - retrouve chaque paire image / label ou qu'elle se trouve,
  - reprend le decoupage existant quand il y en a un (sinon hash deterministe),
  - valide et nettoie les annotations (nb de champs, id de classe, bornes 0-1),
  - isole les images sans annotation et les videos dans data/demo,
  - genere data/dataset/data.yaml.
"""
import argparse
import hashlib
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from config import CLASS_NAMES, DATASET_DIR, DEMO_DIR, IMAGE_EXTS, RAW_DIR, VIDEO_EXTS

SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}


def find_files(root: Path, exts: set[str]) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def label_for(image: Path, root: Path) -> Path | None:
    """Cherche le .txt correspondant a une image (conventions YOLOv5 / LabelImg)."""
    candidates = [image.with_suffix(".txt")]

    parts = list(image.relative_to(root).parts)
    for i, part in enumerate(parts[:-1]):
        if part.lower() in {"images", "image", "img", "jpegimages"}:
            for repl in ("labels", "label", "annotations"):
                alt = list(parts)
                alt[i] = repl
                candidates.append(root.joinpath(*alt).with_suffix(".txt"))

    for parent in image.parents:
        if parent == root.parent:
            break
        for name in ("labels", "label", "annotations"):
            candidates.append(parent / name / f"{image.stem}.txt")

    for c in candidates:
        if c.is_file():
            return c
    return None


def split_from_path(rel: Path) -> str | None:
    for part in rel.parts[:-1]:
        alias = SPLIT_ALIASES.get(part.lower())
        if alias:
            return alias
    return None


def hash_split(key: str, ratios: tuple[float, float, float], seed: int) -> str:
    """Split deterministe : la meme image tombe toujours dans le meme sous-ensemble."""
    x = int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if x < ratios[0]:
        return "train"
    if x < ratios[0] + ratios[1]:
        return "val"
    return "test"


def parse_label(path: Path, n_classes: int) -> tuple[list[str], list[str]]:
    """Retourne (lignes valides, messages d'anomalie)."""
    lines, errors = [], []
    for lineno, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        fields = raw.split()
        if len(fields) != 5:
            errors.append(f"{path}:{lineno} attendu 5 champs, recu {len(fields)}")
            continue
        try:
            cls = int(float(fields[0]))
            box = [float(v) for v in fields[1:]]
        except ValueError:
            errors.append(f"{path}:{lineno} valeur non numerique")
            continue
        if not 0 <= cls < n_classes:
            errors.append(f"{path}:{lineno} classe {cls} hors de [0,{n_classes - 1}]")
            continue
        if any(not 0.0 <= v <= 1.0 for v in box):
            box = [min(max(v, 0.0), 1.0) for v in box]
            errors.append(f"{path}:{lineno} coordonnees hors bornes, recadrees")
        if box[2] <= 0 or box[3] <= 0:
            errors.append(f"{path}:{lineno} boite de taille nulle, ignoree")
            continue
        lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in box))
    return lines, errors


def place(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=DATASET_DIR)
    parser.add_argument("--demo", type=Path, default=DEMO_DIR)
    parser.add_argument("--split-mode", choices=("auto", "existing", "hash"), default="auto",
                        help="auto = reprend le decoupage du dataset s'il existe")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="mode hash uniquement")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="mode hash uniquement")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()

    if not args.raw.exists():
        print(f"{args.raw} introuvable. Lancez d'abord src/extract_archive.py", file=sys.stderr)
        return 1

    ratios = (args.train_ratio, args.val_ratio, max(1.0 - args.train_ratio - args.val_ratio, 0.0))
    if args.train_ratio + args.val_ratio > 1.0 + 1e-9:
        print("train-ratio + val-ratio doit etre <= 1", file=sys.stderr)
        return 1

    images = find_files(args.raw, IMAGE_EXTS)
    if not images:
        print(f"Aucune image trouvee sous {args.raw}", file=sys.stderr)
        return 1

    use_existing = args.split_mode == "existing" or (
        args.split_mode == "auto"
        and sum(1 for p in images if split_from_path(p.relative_to(args.raw))) > 0.9 * len(images)
    )
    print(f"{len(images)} images trouvees sous {args.raw}")
    print("Decoupage : " + ("repris du dataset" if use_existing else f"hash deterministe {ratios}"))

    for d in (args.out, args.demo):
        if d.exists():
            shutil.rmtree(d)
    for split in SPLITS:
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (args.demo / "images").mkdir(parents=True, exist_ok=True)
    (args.demo / "videos").mkdir(parents=True, exist_ok=True)

    stats = {s: Counter() for s in SPLITS}
    counts = Counter()
    unlabeled = 0
    errors: list[str] = []
    used_names: dict[str, int] = defaultdict(int)

    for image in images:
        rel = image.relative_to(args.raw)
        label_path = label_for(image, args.raw)
        lines: list[str] = []
        if label_path is not None:
            lines, errs = parse_label(label_path, len(CLASS_NAMES))
            errors.extend(errs)

        # noms uniques : plusieurs sous-dossiers peuvent contenir le meme nom de fichier
        stem = image.stem.replace(" ", "_")
        used_names[stem] += 1
        if used_names[stem] > 1:
            stem = f"{stem}_{used_names[stem] - 1}"

        if label_path is None or not lines:
            unlabeled += 1
            place(image, args.demo / "images" / f"{stem}{image.suffix.lower()}", args.mode)
            continue

        split = split_from_path(rel) if use_existing else None
        if split is None:
            split = hash_split(str(rel), ratios, args.seed)

        place(image, args.out / "images" / split / f"{stem}{image.suffix.lower()}", args.mode)
        (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        counts[split] += 1
        for line in lines:
            stats[split][int(line.split()[0])] += 1

    for video in find_files(args.raw, VIDEO_EXTS):
        place(video, args.demo / "videos" / video.name.replace(" ", "_"), args.mode)

    data = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    if counts["test"]:
        data["test"] = "images/test"
    yaml_path = args.out / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    print("\n--- Repartition ---")
    for split in SPLITS:
        if counts[split]:
            print(f"  {split:5s} : {counts[split]:5d} images, {sum(stats[split].values()):6d} boites")
    n_videos = len(list((args.demo / "videos").iterdir()))
    print(f"  sans annotation -> {args.demo / 'images'} : {unlabeled} images")
    print(f"  videos          -> {args.demo / 'videos'} : {n_videos}")

    print("\n--- Instances par classe ---")
    active = [s for s in SPLITS if counts[s]]
    print(f"  {'classe':<20}" + "".join(f"{s:>9}" for s in active) + f"{'total':>9}")
    for i, name in enumerate(CLASS_NAMES):
        row = [stats[s][i] for s in active]
        print(f"  {name:<20}" + "".join(f"{v:>9d}" for v in row) + f"{sum(row):>9d}")

    if errors:
        log = args.out / "label_issues.log"
        log.write_text("\n".join(errors))
        print(f"\n{len(errors)} anomalies d'annotation corrigees/ignorees -> {log}")

    print(f"\ndata.yaml genere : {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
