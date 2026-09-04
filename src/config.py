"""Configuration commune au projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Ordre des classes tel que decrit par le dataset Kaggle
# (labels LabelImg au format YOLOv5, indices 0..6).
CLASS_NAMES = [
    "Car",
    "Number Plate",
    "Blur Number Plate",
    "Two Wheeler",
    "Auto",
    "Bus",
    "Truck",
]

RAW_DIR = ROOT / "data" / "raw"          # copie brute telechargee depuis Kaggle
DATASET_DIR = ROOT / "data" / "dataset"  # arborescence YOLO generee
DEMO_DIR = ROOT / "data" / "demo"      # images non annotees + videos de test
DATA_YAML = DATASET_DIR / "data.yaml"
RUNS_DIR = ROOT / "runs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
