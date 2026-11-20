"""Data module (shim re-exporting src/datasets/data).

This package exists so that existing imports of src.data.* continue to work
while the canonical location is src/datasets/ (loads, splits, augmentation).

For new code prefer from src.datasets.data import ....
"""
from src.datasets.data import *
