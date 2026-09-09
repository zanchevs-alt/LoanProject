"""Load the trained loan model and expose JSON-friendly model data."""

import json
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE_PATH = BASE_DIR / "svm_pipeline_model.pkl"
METADATA_FILE_PATH = BASE_DIR / "model_metadata.json"
_pipeline = None
_metadata = None


def _ensure_loaded():
    global _pipeline, _metadata
    if _pipeline is None:
        if not MODEL_FILE_PATH.exists():
            raise FileNotFoundError("Run train_model.py before starting the API.")
        _pipeline = joblib.load(MODEL_FILE_PATH)
    if _metadata is None:
        if not METADATA_FILE_PATH.exists():
            raise FileNotFoundError("Run train_model.py before starting the API.")
        _metadata = json.loads(METADATA_FILE_PATH.read_text(encoding="utf-8"))


def get_model_info():
    _ensure_loaded()
    return _metadata["model_info"]


def get_features():
    _ensure_loaded()
    return _metadata["features"]


def get_samples():
    _ensure_loaded()
    return _metadata["samples"]


def get_metrics():
    _ensure_loaded()
    return _metadata["metrics"]


def get_margins_distribution():
    _ensure_loaded()
    return _metadata["margins_on_test_set"]


def get_model_file_info():
    _ensure_loaded()
    return _metadata["model_file"]


def get_classes():
    _ensure_loaded()
    return _metadata["classes"]


def get_full_model_overview():
    return {"model_info": get_model_info(), "features": get_features(), "classes": get_classes(), "model_file": get_model_file_info()}


def compute_margin(input_df: pd.DataFrame):
    _ensure_loaded()
    return _pipeline.decision_function(input_df).tolist()