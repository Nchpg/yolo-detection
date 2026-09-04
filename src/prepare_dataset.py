"""Turn data/raw into a YOLO layout Ultralytics can train on.

The archive already ships a split -- images/{train,val,test} with labels for
train and val only. The test folder holds raw images and 18 videos, so it is
kept aside as demo material rather than treated as an evaluated split.

Beyond copying files around, this validates every annotation: field count,
class id range, coordinates within [0, 1] and degenerate boxes.
"""
import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from config import CLASS_NAMES, DATASET_DIR, DEMO_DIR, IMAGE_EXTS, RAW_DIR, VIDEO_EXTS

SPLITS = ("train", "val", "test")
ALIASES = {"train": "train", "val": "val", "valid": "val", "test": "test"}


def find_files(root: Path, exts: set[str]) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def label_for(image: Path, root: Path) -> Path | None:
    """Locate the .txt next to the image, or under a parallel labels/ folder."""
    sibling = image.with_suffix(".txt")
    if sibling.is_file():
        return sibling

    parts = list(image.relative_to(root).parts)
    for i, part in enumerate(parts[:-1]):
        if part.lower() == "images":
            parts[i] = "labels"
            candidate = root.joinpath(*parts).with_suffix(".txt")
            return candidate if candidate.is_file() else None
    return None


def split_of(rel: Path) -> str | None:
    for part in rel.parts[:-1]:
        if part.lower() in ALIASES:
            return ALIASES[part.lower()]
    return None


def parse_label(path: Path, n_classes: int) -> tuple[list[str], list[str]]:
    """Return (valid lines, problem reports)."""
    lines, problems = [], []
    for lineno, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 5:
            problems.append(f"{path}:{lineno} expected 5 fields, got {len(fields)}")
            continue
        try:
            cls = int(float(fields[0]))
            box = [float(v) for v in fields[1:]]
        except ValueError:
            problems.append(f"{path}:{lineno} non-numeric value")
            continue
        if not 0 <= cls < n_classes:
            problems.append(f"{path}:{lineno} class {cls} outside [0,{n_classes - 1}]")
            continue
        if any(not 0.0 <= v <= 1.0 for v in box):
            box = [min(max(v, 0.0), 1.0) for v in box]
            problems.append(f"{path}:{lineno} coordinates out of range, clipped")
        if box[2] <= 0 or box[3] <= 0:
            problems.append(f"{path}:{lineno} zero-sized box, dropped")
            continue
        lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in box))
    return lines, problems


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=DATASET_DIR)
    parser.add_argument("--demo", type=Path, default=DEMO_DIR)
    args = parser.parse_args()

    if not args.raw.exists():
        print(f"{args.raw} not found. Run src/extract_archive.py first.", file=sys.stderr)
        return 1

    images = find_files(args.raw, IMAGE_EXTS)
    if not images:
        print(f"No images under {args.raw}", file=sys.stderr)
        return 1
    print(f"{len(images)} images found under {args.raw}")

    for d in (args.out, args.demo):
        if d.exists():
            shutil.rmtree(d)
    (args.demo / "images").mkdir(parents=True)
    (args.demo / "videos").mkdir(parents=True)

    stats = {s: Counter() for s in SPLITS}
    counts = Counter()
    unlabelled = 0
    problems: list[str] = []
    seen: dict[str, int] = defaultdict(int)

    for image in images:
        label_path = label_for(image, args.raw)
        lines: list[str] = []
        if label_path is not None:
            lines, found = parse_label(label_path, len(CLASS_NAMES))
            problems.extend(found)

        # the same basename can appear in several folders
        stem = image.stem.replace(" ", "_")
        seen[stem] += 1
        if seen[stem] > 1:
            stem = f"{stem}_{seen[stem] - 1}"

        if not lines:
            unlabelled += 1
            link(image, args.demo / "images" / f"{stem}{image.suffix.lower()}")
            continue

        split = split_of(image.relative_to(args.raw)) or "train"
        link(image, args.out / "images" / split / f"{stem}{image.suffix.lower()}")
        label_out = args.out / "labels" / split / f"{stem}.txt"
        label_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.write_text("\n".join(lines) + "\n")

        counts[split] += 1
        for line in lines:
            stats[split][int(line.split()[0])] += 1

    for video in find_files(args.raw, VIDEO_EXTS):
        link(video, args.demo / "videos" / video.name.replace(" ", "_"))

    data = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": dict(enumerate(CLASS_NAMES)),
    }
    if counts["test"]:
        data["test"] = "images/test"
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    active = [s for s in SPLITS if counts[s]]
    print()
    for split in active:
        print(f"  {split:5s} {counts[split]:5d} images, {sum(stats[split].values()):6d} boxes")
    n_videos = len(list((args.demo / "videos").iterdir()))
    print(f"  demo  {unlabelled:5d} unlabelled images, {n_videos} videos -> {args.demo}")

    print(f"\n  {'class':<20}" + "".join(f"{s:>9}" for s in active) + f"{'total':>9}")
    for i, name in enumerate(CLASS_NAMES):
        row = [stats[s][i] for s in active]
        print(f"  {name:<20}" + "".join(f"{v:>9d}" for v in row) + f"{sum(row):>9d}")

    if problems:
        log = args.out / "label_issues.log"
        log.write_text("\n".join(problems))
        print(f"\n  {len(problems)} annotation problems clipped or dropped -> {log}")

    print(f"\n  {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
