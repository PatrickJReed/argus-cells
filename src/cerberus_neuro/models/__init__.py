"""Model zoo for cerberus-neuro / argus-cells.

Re-exports the disease-classifier models so callers can import them from one
place. :class:`BaselineDiseaseClassifier` (ResNet34, 6-channel) lives in
``cerberus_neuro.model`` and is re-exported here for symmetry with the
transformer-family :class:`ArgusCCT`.
"""

from __future__ import annotations

from cerberus_neuro.model import BaselineDiseaseClassifier

from .cct import ArgusCCT

__all__ = ["ArgusCCT", "BaselineDiseaseClassifier"]
