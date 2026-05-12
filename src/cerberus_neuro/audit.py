"""Donor-structure audit utilities for Phase 0 of argus-cells.

Pure-pandas operations on the manifest DataFrame returned by
:func:`cerberus_neuro.data.build_manifest`. No S3, no torch, no PyTorch
dependencies — these utilities run on Colab Free.
"""
from __future__ import annotations

import pandas as pd


def donor_counts_by_condition(manifest: pd.DataFrame) -> dict[str, int]:
    """Count unique donor lines (Metadata_line_ID) per Metadata_line_condition.

    Parameters
    ----------
    manifest
        DataFrame with at least ``Metadata_line_ID`` and
        ``Metadata_line_condition`` columns.

    Returns
    -------
    Dict mapping each condition value to the number of unique line_IDs
    observed under that condition. Empty manifest returns an empty dict.
    """
    if len(manifest) == 0:
        return {}
    return (
        manifest.groupby("Metadata_line_condition")["Metadata_line_ID"]
        .nunique()
        .to_dict()
    )
