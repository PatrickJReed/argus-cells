"""NeuroPainting data pipeline (cpg0038-tegtmeyer-neuropainting).

Two entry points:

- :func:`build_manifest` — concatenate per-plate ``load_data.csv`` URL columns
  with biological labels from per-plate ``platemap.txt`` files, returning one
  row per (plate, well, site).
- :class:`NeuroPaintingDataset` — IterableDataset streaming
  ``(brightfield[1, h, w], fluorescence[5, h, w], cell_type, line_condition)``
  tuples. v0 strategy: random ``crop_size`` crops from each loaded 2160×2160
  site, same coordinates across all 6 channels. Crops are rejected unless
  they contain at least ``min_cells_per_crop`` CellProfiler cell centroids
  (default 1) so we don't train on background-only tiles. Per-cell crops via
  the segmentation outlines under ``publication_data/`` remain a v1 option.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import boto3
import numpy as np
import pandas as pd
import torch
from botocore import UNSIGNED
from botocore.config import Config
from PIL import Image
from torch.utils.data import IterableDataset

BUCKET = "cellpainting-gallery"
ROOT_PREFIX = "cpg0038-tegtmeyer-neuropainting/"
WORKSPACE_PREFIX = ROOT_PREFIX + "broad/workspace/"
IMAGES_PREFIX = ROOT_PREFIX + "broad/images/"

# Default v0 batch set excludes the 63× neurons batch (different magnification
# breaks the cell-type-classification task; reserved for v1 resolution-invariance
# work).
BATCHES_V0 = [
    "NCP_ASTROCYTES_1",
    "NCP_NEURONS_2_20x",
    "NCP_PROGENITORS_1",
    "NCP_STEM_1",
]

# Brightfield is the model input; the five fluorescence channels are virtual-
# staining targets. This order is what the dataset stacks into the channel axis.
CHANNEL_INPUT = "OrigBrightfield"
CHANNELS_FLUORESCENCE = ["OrigDNA", "OrigMito", "OrigAGP", "OrigER", "OrigRNA"]
CHANNEL_ORDER = [CHANNEL_INPUT] + CHANNELS_FLUORESCENCE

CELL_TYPES = ["stem", "progen", "neuron", "astro"]
CELL_TYPE_TO_IDX = {c: i for i, c in enumerate(CELL_TYPES)}

LINE_CONDITIONS = ["control", "deletion"]
LINE_CONDITION_TO_IDX = {c: i for i, c in enumerate(LINE_CONDITIONS)}


def _s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _list_recursive(s3, prefix: str) -> list[tuple[str, int]]:
    paginator = s3.get_paginator("list_objects_v2")
    out = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, PaginationConfig={"PageSize": 1000}):
        out.extend((o["Key"], o["Size"]) for o in page.get("Contents", []))
    return out


def _download(s3, key: str, local: Path) -> Path:
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, key, str(local))
    return local


def _key_from_url(url: str) -> str:
    prefix = f"s3://{BUCKET}/"
    return url[len(prefix):] if url.startswith(prefix) else url.lstrip("/")


def build_manifest(cache_dir: Path, batches: list[str] | None = None) -> pd.DataFrame:
    """Build a per-(plate, well, site) manifest.

    Joins per-plate ``load_data.csv`` (URL_* image columns + Metadata_Site) with
    per-plate ``platemap.txt`` (Metadata_cell_type, Metadata_line_ID,
    Metadata_line_condition, Metadata_line_source).

    Parameters
    ----------
    cache_dir
        Local directory used to cache downloaded CSVs. Files land at
        ``cache_dir / <s3_key>``.
    batches
        Subset of batches to include. Defaults to :data:`BATCHES_V0` (excludes
        the 63× neurons batch).

    Returns
    -------
    DataFrame indexed by (Metadata_Plate, Metadata_Well, Metadata_Site) with
    columns: URL_OrigBrightfield, URL_OrigDNA, URL_OrigMito, URL_OrigAGP,
    URL_OrigER, URL_OrigRNA, Metadata_cell_type, Metadata_line_ID,
    Metadata_line_condition, Metadata_line_source, batch.
    """
    cache_dir = Path(cache_dir)
    s3 = _s3_client()
    batches = batches if batches is not None else BATCHES_V0

    platemap_frames = []
    for batch in batches:
        for key, _ in _list_recursive(s3, f"{WORKSPACE_PREFIX}metadata/{batch}/platemap/"):
            if not key.endswith(".txt"):
                continue
            local = _download(s3, key, cache_dir / key)
            df = pd.read_csv(local, sep="\t")
            df["batch"] = batch
            platemap_frames.append(df)
    pmap = pd.concat(platemap_frames, ignore_index=True).drop_duplicates(
        subset=["Metadata_Plate", "Metadata_Well"]
    )

    keep_cols = ["Metadata_Plate", "Metadata_Well", "Metadata_Site"] + [
        f"URL_{c}" for c in CHANNEL_ORDER
    ]
    load_frames = []
    for batch in batches:
        for key, _ in _list_recursive(s3, f"{WORKSPACE_PREFIX}load_data_csv/{batch}/"):
            if not key.endswith("/load_data.csv"):
                continue
            local = _download(s3, key, cache_dir / key)
            df = pd.read_csv(local, usecols=keep_cols)
            df["batch"] = batch
            load_frames.append(df)
    loads = pd.concat(load_frames, ignore_index=True).drop_duplicates(
        subset=["Metadata_Plate", "Metadata_Well", "Metadata_Site"]
    )

    pmap_cols = [
        "Metadata_Plate", "Metadata_Well",
        "Metadata_cell_type", "Metadata_line_ID",
        "Metadata_line_condition", "Metadata_line_source",
    ]
    return loads.merge(pmap[pmap_cols], on=["Metadata_Plate", "Metadata_Well"], how="inner")


def subset_manifest(
    manifest: pd.DataFrame,
    wells_per_cell_type: int | None = None,
    sites_per_well: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Deterministic subsampling for Colab-Free-friendly v0 runs.

    ``wells_per_cell_type`` is split evenly across line_condition (control vs
    deletion). ``sites_per_well`` caps the per-well site count. Both are
    optional; passing neither returns the input.
    """
    rng = np.random.default_rng(seed)
    if wells_per_cell_type is not None:
        kept = []
        per_cond = max(1, wells_per_cell_type // 2)
        for _, ct_df in manifest.groupby("Metadata_cell_type"):
            wells = ct_df[
                ["Metadata_Plate", "Metadata_Well", "Metadata_line_condition"]
            ].drop_duplicates()
            for _, cond_df in wells.groupby("Metadata_line_condition"):
                idx = rng.choice(len(cond_df), size=min(per_cond, len(cond_df)), replace=False)
                kept.append(cond_df.iloc[idx][["Metadata_Plate", "Metadata_Well"]])
        kept_keys = pd.concat(kept).drop_duplicates().reset_index(drop=True)
        manifest = manifest.merge(kept_keys, on=["Metadata_Plate", "Metadata_Well"], how="inner")
    if sites_per_well is not None:
        manifest = (
            manifest.groupby(["Metadata_Plate", "Metadata_Well"], group_keys=False)
            .apply(lambda g: g.sample(min(sites_per_well, len(g)), random_state=seed))
            .reset_index(drop=True)
        )
    return manifest


def _load_image(s3, url: str, cache_dir: Path) -> np.ndarray:
    key = _key_from_url(url)
    local = _download(s3, key, cache_dir / key)
    with Image.open(local) as im:
        return np.asarray(im)


_CENTROID_COL_CANDIDATES = [
    ("AreaShape_Center_Y", "AreaShape_Center_X"),
    ("Location_Center_Y",  "Location_Center_X"),
    ("Center_Y",           "Center_X"),
]


def load_cell_centroids(
    s3,
    batch: str,
    plate: str,
    well: str,
    site,
    cache_dir: Path,
) -> np.ndarray:
    """Return an ``Nx2`` array of ``(y, x)`` cell centroids in image-pixel space.

    Reads CellProfiler ``Cells.csv`` from
    ``workspace/analysis/<batch>/<plate>/analysis/<plate>-<well>-<site>/Cells.csv``.
    Tries common centroid column-name variants (CP version dependent).
    """
    site_str = str(site)
    key = (
        f"{WORKSPACE_PREFIX}analysis/{batch}/{plate}/analysis/"
        f"{plate}-{well}-{site_str}/Cells.csv"
    )
    local = _download(s3, key, Path(cache_dir) / key)
    df = pd.read_csv(local)
    for y_col, x_col in _CENTROID_COL_CANDIDATES:
        if y_col in df.columns and x_col in df.columns:
            return np.stack([df[y_col].to_numpy(), df[x_col].to_numpy()], axis=1)
    raise KeyError(
        f"No centroid columns in {key}; first columns: {list(df.columns)[:20]}"
    )


def crop_cell_count(centroids: np.ndarray | None, y: int, x: int, size: int) -> int:
    """How many centroids fall inside the ``[y, y+size) x [x, x+size)`` crop."""
    if centroids is None or len(centroids) == 0:
        return 0
    inside = (
        (centroids[:, 0] >= y) & (centroids[:, 0] < y + size) &
        (centroids[:, 1] >= x) & (centroids[:, 1] < x + size)
    )
    return int(inside.sum())


@dataclass
class NeuroPaintingDataset(IterableDataset):
    """IterableDataset streaming (brightfield, fluorescence, cell_type, line_condition).

    Each yielded sample is a tuple of:

    - ``brightfield`` — float32 tensor, shape ``(1, crop_size, crop_size)``,
      values in ``[0, 1]``.
    - ``fluorescence`` — float32 tensor, shape ``(5, crop_size, crop_size)``,
      values in ``[0, 1]``, channel order ``[DNA, Mito, AGP, ER, RNA]``.
    - ``cell_type`` — int in ``[0, 4)`` (index into :data:`CELL_TYPES`).
    - ``line_condition`` — int in ``{0, 1}`` (control vs deletion).

    For each (plate, well, site) row in the manifest, ``crops_per_site``
    independent random crops are yielded. Same crop coordinates are applied to
    all 6 channels so brightfield input and fluorescence target stay aligned.

    When ``min_cells_per_crop > 0``, the per-site CellProfiler ``Cells.csv``
    centroids are loaded and crops are rejected (up to ``max_crop_attempts``
    tries each) unless they contain at least ``min_cells_per_crop`` cell
    centroids. Sites where every attempt fails skip the remaining requested
    crops; sites missing a Cells.csv are dropped entirely.
    """

    manifest: pd.DataFrame
    cache_dir: Path
    crop_size: int = 256
    crops_per_site: int = 4
    min_cells_per_crop: int = 1
    max_crop_attempts: int = 32
    shuffle: bool = True
    seed: int = 0

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, int, int]]:
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else 0
        n_workers = worker.num_workers if worker else 1

        rng = np.random.default_rng(self.seed + worker_id)
        s3 = _s3_client()
        cache = Path(self.cache_dir)

        rows = self.manifest.iloc[worker_id::n_workers].reset_index(drop=True)
        if self.shuffle:
            rows = rows.sample(frac=1.0, random_state=self.seed + worker_id).reset_index(drop=True)

        for _, row in rows.iterrows():
            try:
                channels = np.stack(
                    [_load_image(s3, row[f"URL_{c}"], cache) for c in CHANNEL_ORDER],
                    axis=0,
                )
                centroids = None
                if self.min_cells_per_crop > 0:
                    centroids = load_cell_centroids(
                        s3,
                        row["batch"], row["Metadata_Plate"],
                        row["Metadata_Well"], row["Metadata_Site"],
                        cache,
                    )
            except Exception:
                continue

            channels = channels.astype(np.float32) / 65535.0
            ct = CELL_TYPE_TO_IDX[row["Metadata_cell_type"]]
            cond = LINE_CONDITION_TO_IDX[row["Metadata_line_condition"]]

            _, h, w = channels.shape
            for _ in range(self.crops_per_site):
                y, x = -1, -1
                for _ in range(self.max_crop_attempts):
                    yy = int(rng.integers(0, h - self.crop_size + 1))
                    xx = int(rng.integers(0, w - self.crop_size + 1))
                    n_cells = crop_cell_count(centroids, yy, xx, self.crop_size)
                    if self.min_cells_per_crop <= 0 or n_cells >= self.min_cells_per_crop:
                        y, x = yy, xx
                        break
                if y < 0:
                    continue
                crop = channels[:, y:y + self.crop_size, x:x + self.crop_size]
                yield (
                    torch.from_numpy(crop[:1].copy()),
                    torch.from_numpy(crop[1:].copy()),
                    ct,
                    cond,
                )
