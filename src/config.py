"""Shared paths and class list."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Class order as annotated in the Kaggle dataset (LabelImg, YOLOv5 format).
CLASS_NAMES = [
    "Car",
    "Number Plate",
    "Blur Number Plate",
    "Two Wheeler",
    "Auto",
    "Bus",
    "Truck",
]

RAW_DIR = ROOT / "data" / "raw"          # unpacked archive
DATASET_DIR = ROOT / "data" / "dataset"  # generated YOLO layout
DEMO_DIR = ROOT / "data" / "demo"        # unlabelled images and videos
DATA_YAML = DATASET_DIR / "data.yaml"
RUNS_DIR = ROOT / "runs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
