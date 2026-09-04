"""Extrait l'archive Kaggle (archive.zip) dans data/raw."""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from config import RAW_DIR, ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=ROOT / "archive.zip")
    parser.add_argument("--dest", type=Path, default=RAW_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.zip.exists():
        print(f"{args.zip} introuvable.", file=sys.stderr)
        return 1
    if args.dest.exists():
        if not args.force:
            print(f"{args.dest} existe deja (utilisez --force pour ecraser).")
            return 0
        shutil.rmtree(args.dest)

    args.dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as z:
        members = z.namelist()
        print(f"Extraction de {len(members)} entrees vers {args.dest} ...")
        z.extractall(args.dest)
    print("Termine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
