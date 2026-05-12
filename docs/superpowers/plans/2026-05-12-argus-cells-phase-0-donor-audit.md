# argus-cells Phase 0 — Donor Structure Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit the donor structure of `cpg0038-tegtmeyer-neuropainting` to determine whether the argus-cells donor-confound analysis can proceed (≥3 donor lines per condition) and produce the crop-budget recommendation that scopes Phase 2 training.

**Architecture:** Add a small `audit` module to the existing `cerberus_neuro` package with pure-pandas utility functions that operate on the manifest DataFrame produced by `build_manifest()`. Drive the analysis from a new notebook (`notebooks/03_donor_audit.ipynb`) that consumes those utilities and produces tables + plots on Colab Free. Write the findings, the imbalance metrics, and the explicit gate decision to `docs/superpowers/results/2026-05-12-phase-0-donor-audit.md`.

**Tech Stack:** Python 3.10+, pandas, numpy, matplotlib, pytest, Jupyter. No GPU required — Colab Free is sufficient.

**Spec:** [`docs/superpowers/specs/2026-05-12-argus-cells-design.md`](../specs/2026-05-12-argus-cells-design.md), §3 Phase 0 prerequisite + §5 Phase 0 row.

---

## File Structure

- Create: `src/cerberus_neuro/audit.py` — donor-structure audit utilities, pure pandas, no S3/torch dependencies.
- Create: `tests/__init__.py` — empty, marks tests as a package.
- Create: `tests/test_audit.py` — unit tests for audit utilities with a synthetic fixture.
- Create: `notebooks/03_donor_audit.ipynb` — the audit notebook, consumes `cerberus_neuro.audit` and `cerberus_neuro.data.build_manifest`.
- Create: `docs/superpowers/results/2026-05-12-phase-0-donor-audit.md` — audit report with explicit gate decision.
- Modify: `pyproject.toml` (line 3) and `src/cerberus_neuro/__init__.py` (line 1) — version bump `0.0.22` → `0.0.23` so Colab `pip install --upgrade` re-resolves the new `audit` symbol.

---

## Task 1: Add tests scaffold + version bump

**Files:**
- Create: `tests/__init__.py`
- Modify: `pyproject.toml:3`
- Modify: `src/cerberus_neuro/__init__.py:1`

- [ ] **Step 1: Create empty tests package**

```bash
mkdir -p tests && touch tests/__init__.py
```

- [ ] **Step 2: Bump version in `pyproject.toml`**

Open `pyproject.toml`, change line 3:

```toml
version = "0.0.23"
```

- [ ] **Step 3: Bump version in `__init__.py`**

Open `src/cerberus_neuro/__init__.py`, change line 1:

```python
__version__ = "0.0.23"
```

- [ ] **Step 4: Verify pytest discovery works**

```bash
pip install -e ".[dev]" && pytest tests/ -v
```

Expected: `no tests ran` (or `collected 0 items`), exit code 0. This confirms pytest is installed and discovers the (empty) tests directory without error.

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py pyproject.toml src/cerberus_neuro/__init__.py
git commit -m "test: scaffold tests/ package and bump version to 0.0.23 for Phase 0 audit module"
```

---

## Task 2: Add `audit.py` with `donor_counts_by_condition()` (TDD)

**Files:**
- Create: `src/cerberus_neuro/audit.py`
- Create: `tests/test_audit.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_audit.py`:

```python
"""Tests for cerberus_neuro.audit donor-structure utilities."""
from __future__ import annotations

import pandas as pd
import pytest

from cerberus_neuro.audit import donor_counts_by_condition


@pytest.fixture
def tiny_manifest() -> pd.DataFrame:
    """Synthetic manifest mirroring the real cpg0038 schema.

    3 donors per condition (D1-D3 control, D4-D6 deletion).
    4 cell types per donor (stem, progen, neuron, astro).
    2 wells per (donor, cell_type), 4 sites per well.
    Total rows: 6 donors x 4 cell types x 2 wells x 4 sites = 192.
    """
    rows = []
    donors = [
        ("D1", "control"), ("D2", "control"), ("D3", "control"),
        ("D4", "deletion"), ("D5", "deletion"), ("D6", "deletion"),
    ]
    for donor_id, condition in donors:
        for cell_type in ["stem", "progen", "neuron", "astro"]:
            for well_idx in range(2):
                well = f"A{well_idx + 1:02d}"
                for site in range(4):
                    rows.append({
                        "Metadata_Plate": f"plate_{donor_id}_{cell_type}",
                        "Metadata_Well": well,
                        "Metadata_Site": site + 1,
                        "Metadata_cell_type": cell_type,
                        "Metadata_line_ID": donor_id,
                        "Metadata_line_condition": condition,
                        "Metadata_line_source": "synthetic",
                        "batch": f"NCP_{cell_type.upper()}_1",
                    })
    return pd.DataFrame(rows)


def test_donor_counts_by_condition_returns_correct_counts(tiny_manifest):
    counts = donor_counts_by_condition(tiny_manifest)
    assert counts == {"control": 3, "deletion": 3}


def test_donor_counts_by_condition_empty_returns_empty():
    empty = pd.DataFrame(columns=["Metadata_line_ID", "Metadata_line_condition"])
    assert donor_counts_by_condition(empty) == {}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_audit.py -v
```

Expected: `ImportError: cannot import name 'donor_counts_by_condition' from 'cerberus_neuro.audit'`.

- [ ] **Step 3: Write the minimal implementation**

Create `src/cerberus_neuro/audit.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_audit.py -v
```

Expected: `2 passed`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add src/cerberus_neuro/audit.py tests/test_audit.py
git commit -m "audit: donor_counts_by_condition() with tests"
```

---

## Task 3: Add `donor_well_table()` (TDD)

**Files:**
- Modify: `src/cerberus_neuro/audit.py`
- Modify: `tests/test_audit.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_audit.py`:

```python
from cerberus_neuro.audit import donor_well_table


def test_donor_well_table_shape(tiny_manifest):
    table = donor_well_table(tiny_manifest)
    # 6 donors x 4 cell types per donor = 24 (donor, cell_type) groups.
    assert len(table) == 24
    assert set(table.columns) == {"cell_type", "line_condition", "line_ID", "n_wells"}


def test_donor_well_table_well_counts(tiny_manifest):
    table = donor_well_table(tiny_manifest)
    # Each (donor, cell_type) combo has 2 wells in the fixture.
    assert (table["n_wells"] == 2).all()


def test_donor_well_table_donor_coverage(tiny_manifest):
    table = donor_well_table(tiny_manifest)
    # Every donor appears in every cell type.
    assert set(table["line_ID"]) == {"D1", "D2", "D3", "D4", "D5", "D6"}
    assert set(table["cell_type"]) == {"stem", "progen", "neuron", "astro"}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_audit.py -v
```

Expected: `ImportError: cannot import name 'donor_well_table' from 'cerberus_neuro.audit'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `src/cerberus_neuro/audit.py`:

```python
def donor_well_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Build a (cell_type, line_condition, line_ID) -> well-count table.

    One row per unique (cell_type, line_condition, line_ID) triple, with
    ``n_wells`` = count of distinct (Metadata_Plate, Metadata_Well) pairs
    falling under that triple.

    Returns
    -------
    DataFrame with columns: ``cell_type``, ``line_condition``, ``line_ID``,
    ``n_wells``. Sorted by (cell_type, line_condition, line_ID).
    """
    well_keys = (
        manifest[
            [
                "Metadata_cell_type",
                "Metadata_line_condition",
                "Metadata_line_ID",
                "Metadata_Plate",
                "Metadata_Well",
            ]
        ]
        .drop_duplicates()
    )
    counts = (
        well_keys.groupby(
            ["Metadata_cell_type", "Metadata_line_condition", "Metadata_line_ID"]
        )
        .size()
        .reset_index(name="n_wells")
    )
    counts = counts.rename(
        columns={
            "Metadata_cell_type": "cell_type",
            "Metadata_line_condition": "line_condition",
            "Metadata_line_ID": "line_ID",
        }
    )
    return counts.sort_values(["cell_type", "line_condition", "line_ID"]).reset_index(drop=True)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_audit.py -v
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/cerberus_neuro/audit.py tests/test_audit.py
git commit -m "audit: donor_well_table() with tests"
```

---

## Task 4: Add `imbalance_metric()` (TDD)

**Files:**
- Modify: `src/cerberus_neuro/audit.py`
- Modify: `tests/test_audit.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_audit.py`:

```python
import math

from cerberus_neuro.audit import imbalance_metric


def test_imbalance_metric_perfect_balance(tiny_manifest):
    """Equal wells per donor → CV = 0 for every (cell_type, line_condition)."""
    table = donor_well_table(tiny_manifest)
    imbalance = imbalance_metric(table)
    # 4 cell types x 2 conditions = 8 groups.
    assert len(imbalance) == 8
    for key, val in imbalance.items():
        assert val["cv"] == 0
        assert val["n_donors"] == 3


def test_imbalance_metric_imbalanced_high_cv():
    """One donor has 10 wells, others have 1 — CV should be > 1.0."""
    table = pd.DataFrame(
        [
            {"cell_type": "stem", "line_condition": "control", "line_ID": "D1", "n_wells": 10},
            {"cell_type": "stem", "line_condition": "control", "line_ID": "D2", "n_wells": 1},
            {"cell_type": "stem", "line_condition": "control", "line_ID": "D3", "n_wells": 1},
        ]
    )
    imbalance = imbalance_metric(table)
    assert imbalance[("stem", "control")]["cv"] > 1.0
    assert imbalance[("stem", "control")]["n_donors"] == 3


def test_imbalance_metric_single_donor_returns_nan():
    """Single donor in a group: CV is undefined → NaN."""
    table = pd.DataFrame(
        [{"cell_type": "stem", "line_condition": "control", "line_ID": "D1", "n_wells": 5}]
    )
    imbalance = imbalance_metric(table)
    assert math.isnan(imbalance[("stem", "control")]["cv"])
    assert imbalance[("stem", "control")]["n_donors"] == 1
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_audit.py -v
```

Expected: `ImportError: cannot import name 'imbalance_metric' from 'cerberus_neuro.audit'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `src/cerberus_neuro/audit.py`:

```python
import math


def imbalance_metric(table: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    """Donor-balance coefficient of variation per (cell_type, line_condition).

    For each (cell_type, line_condition) group, computes the coefficient of
    variation (std/mean) of per-donor ``n_wells``. CV=0 means perfectly
    balanced donor representation; higher CV means one or two donors
    dominate the group.

    Single-donor groups return ``cv=NaN`` (CV is undefined with N=1).

    Parameters
    ----------
    table
        Output of :func:`donor_well_table`. Must have columns
        ``cell_type``, ``line_condition``, ``line_ID``, ``n_wells``.

    Returns
    -------
    Dict keyed by ``(cell_type, line_condition)`` tuples, with values
    ``{"cv": float, "n_donors": int}``.
    """
    out: dict[tuple[str, str], dict[str, float]] = {}
    for (cell_type, condition), group in table.groupby(["cell_type", "line_condition"]):
        wells = group["n_wells"].to_numpy()
        n_donors = int(len(wells))
        if n_donors <= 1:
            cv = math.nan
        else:
            mean = float(wells.mean())
            cv = float(wells.std() / mean) if mean > 0 else math.nan
        out[(cell_type, condition)] = {"cv": cv, "n_donors": n_donors}
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_audit.py -v
```

Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/cerberus_neuro/audit.py tests/test_audit.py
git commit -m "audit: imbalance_metric() with tests"
```

---

## Task 5: Add `crop_budget_estimate()` (TDD)

**Files:**
- Modify: `src/cerberus_neuro/audit.py`
- Modify: `tests/test_audit.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_audit.py`:

```python
from cerberus_neuro.audit import crop_budget_estimate


def test_crop_budget_estimate_returns_expected_shape(tiny_manifest):
    budget = crop_budget_estimate(tiny_manifest, crops_per_site=10)
    # 6 donors x 4 cell types x 2 wells x 4 sites = 192 rows
    # 6 donors x 4 cell types x 2 wells = 48 wells.
    assert budget["n_sites"] == 192
    assert budget["n_wells"] == 48
    assert budget["crops_per_site"] == 10
    assert budget["max_crops_upper_bound"] == 1920


def test_crop_budget_estimate_zero_crops_per_site(tiny_manifest):
    budget = crop_budget_estimate(tiny_manifest, crops_per_site=0)
    assert budget["max_crops_upper_bound"] == 0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_audit.py -v
```

Expected: `ImportError: cannot import name 'crop_budget_estimate' from 'cerberus_neuro.audit'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `src/cerberus_neuro/audit.py`:

```python
def crop_budget_estimate(manifest: pd.DataFrame, crops_per_site: int) -> dict[str, int]:
    """Naive upper-bound estimate of total crops yielded by the pipeline.

    Real yield is typically 60-90% of this upper bound after the
    CellProfiler ``min_cells_per_crop`` filter is applied at dataset
    iteration time. Use this estimate as an *upper bound* for Phase 2
    crop-budget planning; multiply by the empirical yield ratio from
    Phase 0.5 / Phase 1 for a realistic number.

    Parameters
    ----------
    manifest
        DataFrame with columns ``Metadata_Plate``, ``Metadata_Well`` (one
        row per (plate, well, site) tuple).
    crops_per_site
        Configurable number of crops yielded per site by the
        CellProfiler-centroid tile selector.

    Returns
    -------
    Dict with ``n_sites``, ``n_wells``, ``crops_per_site``, and
    ``max_crops_upper_bound`` keys.
    """
    n_sites = int(len(manifest))
    n_wells = int(
        manifest.groupby(["Metadata_Plate", "Metadata_Well"]).ngroups
    )
    return {
        "n_sites": n_sites,
        "n_wells": n_wells,
        "crops_per_site": int(crops_per_site),
        "max_crops_upper_bound": n_sites * int(crops_per_site),
    }
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_audit.py -v
```

Expected: `10 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/cerberus_neuro/audit.py tests/test_audit.py
git commit -m "audit: crop_budget_estimate() with tests"
```

---

## Task 6: Add audit notebook scaffold

**Files:**
- Create: `notebooks/03_donor_audit.ipynb`

- [ ] **Step 1: Create the notebook by copying the existing template**

Use `notebooks/00_environment_smoke.ipynb` as the structural template (markdown header + Open-In-Colab badge + pip install + Drive mount). Create `notebooks/03_donor_audit.ipynb` with the following cell sequence:

**Cell 1 — markdown header:**

```markdown
# Phase 0 — Donor Structure Audit of cpg0038

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PatrickJReed/cerberus-neuro/blob/main/notebooks/03_donor_audit.ipynb)

**Goal:** Tabulate donor lines per condition, per cell type. Compute imbalance metrics.
Estimate the crop budget for Phase 2. Produce inputs for the Phase 0 audit report.

**Spec:** `docs/superpowers/specs/2026-05-12-argus-cells-design.md`, §3 + §5.
**Plan:** `docs/superpowers/plans/2026-05-12-argus-cells-phase-0-donor-audit.md`.
```

**Cell 2 — pip install:**

```python
!pip install -q --upgrade git+https://github.com/PatrickJReed/cerberus-neuro.git@main
```

**Cell 3 — Drive mount (for manifest caching):**

```python
from google.colab import drive

drive.mount("/content/drive")
```

**Cell 4 — imports + cache dir:**

```python
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from cerberus_neuro.data import build_manifest
from cerberus_neuro.audit import (
    donor_counts_by_condition,
    donor_well_table,
    imbalance_metric,
    crop_budget_estimate,
)

pd.set_option("display.max_rows", 200)

CACHE_DIR = Path("/content/drive/MyDrive/cerberus-neuro/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: Save the notebook with cleared outputs**

Notebook cells contain code only; cell outputs are intentionally empty at commit time. The user will run the notebook on Colab; Colab autosave-to-GitHub will push the executed notebook (with outputs) back.

- [ ] **Step 3: Verify the notebook opens cleanly**

```bash
jupyter nbconvert --to script notebooks/03_donor_audit.ipynb --stdout | head -30
```

Expected: the four cells render as Python source without `SyntaxError`. The shell `!pip install` line will render as `get_ipython().system(...)`. That's fine.

- [ ] **Step 4: Commit**

```bash
git add notebooks/03_donor_audit.ipynb
git commit -m "phase0: 03_donor_audit notebook scaffold (cells 1-4: header, install, drive, imports)"
```

---

## Task 7: Add the audit-execution cells to the notebook

**Files:**
- Modify: `notebooks/03_donor_audit.ipynb`

- [ ] **Step 1: Add cell 5 (build manifest)**

```python
# Cell 5 — Build the manifest from cpg0038 platemaps + load_data CSVs.
# This is identical to what 01_data_exploration.ipynb does; ~1-3 min.
manifest = build_manifest(cache_dir=CACHE_DIR)
print(f"Manifest rows (sites): {len(manifest):,}")
print(f"Unique wells: {manifest.groupby(['Metadata_Plate', 'Metadata_Well']).ngroups:,}")
print(f"Cell types: {sorted(manifest['Metadata_cell_type'].unique())}")
print(f"Conditions: {sorted(manifest['Metadata_line_condition'].unique())}")
print(f"Donors (line_IDs): {sorted(manifest['Metadata_line_ID'].unique())}")
manifest.head()
```

- [ ] **Step 2: Add cell 6 (donor counts per condition)**

```python
# Cell 6 — Donor counts per condition. Gate check: N >= 3 per condition.
counts = donor_counts_by_condition(manifest)
print("Donor counts per condition:")
for cond, n in counts.items():
    flag = "OK" if n >= 3 else "BLOCKER"
    print(f"  {cond}: {n} donor lines  [{flag}]")
```

- [ ] **Step 3: Add cell 7 (donor x cell_type x condition table)**

```python
# Cell 7 — Full per-(cell_type, condition, donor) well-count table.
well_table = donor_well_table(manifest)
print(f"Total (cell_type, condition, donor) groups: {len(well_table)}")
print(well_table.to_string(index=False))
```

- [ ] **Step 4: Add cell 8 (imbalance metric)**

```python
# Cell 8 — Per-(cell_type, condition) donor-balance CV.
# CV = 0 is perfectly balanced; high CV means one donor dominates.
imbalance = imbalance_metric(well_table)
imb_df = pd.DataFrame(
    [
        {"cell_type": k[0], "line_condition": k[1], "cv": v["cv"], "n_donors": v["n_donors"]}
        for k, v in imbalance.items()
    ]
)
print(imb_df.to_string(index=False))
```

- [ ] **Step 5: Add cell 9 (crop budget estimate, three scenarios)**

```python
# Cell 9 — Crop budget upper-bound at three candidate crops_per_site values.
# Phase 0.5 / Phase 1 yielded ~16k crops at crops_per_site=10. Phase 2 target
# is ~50-100k. Try crops_per_site = 10, 30, 50.
for cps in [10, 30, 50]:
    budget = crop_budget_estimate(manifest, crops_per_site=cps)
    print(
        f"  crops_per_site={cps:>3}: "
        f"sites={budget['n_sites']:,}, "
        f"wells={budget['n_wells']:,}, "
        f"upper-bound crops={budget['max_crops_upper_bound']:,}"
    )
```

- [ ] **Step 6: Save notebook with cleared outputs and commit**

```bash
git add notebooks/03_donor_audit.ipynb
git commit -m "phase0: 03_donor_audit cells 5-9 (manifest, counts, table, imbalance, budget)"
```

---

## Task 8: Add visualization cell to the notebook

**Files:**
- Modify: `notebooks/03_donor_audit.ipynb`

- [ ] **Step 1: Add cell 10 (donor balance heatmap)**

```python
# Cell 10 — Heatmap of wells per (cell_type, donor), faceted by condition.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, condition in zip(axes, ["control", "deletion"]):
    sub = well_table[well_table["line_condition"] == condition]
    pivot = sub.pivot_table(
        index="cell_type",
        columns="line_ID",
        values="n_wells",
        fill_value=0,
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Wells per (cell_type, donor) — {condition}")
    fig.colorbar(im, ax=ax, label="n_wells")
plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Add cell 11 (summary block — copy these numbers into the audit report)**

```python
# Cell 11 — Summary block for the audit report. Paste this into the
# `Quantitative findings` section of docs/superpowers/results/2026-05-12-phase-0-donor-audit.md
print("=" * 60)
print("PHASE 0 AUDIT SUMMARY")
print("=" * 60)
print()
print("Donor counts per condition:")
for cond, n in counts.items():
    flag = "OK" if n >= 3 else "BLOCKER"
    print(f"  {cond}: {n} donor lines  [{flag}]")
print()
print("Per-(cell_type, condition) imbalance (donor-balance CV):")
for (ct, cond), val in imbalance.items():
    print(f"  ({ct:>6}, {cond:>8}): CV={val['cv']:.3f}, N_donors={val['n_donors']}")
print()
print("Crop budget at crops_per_site=30:")
budget = crop_budget_estimate(manifest, crops_per_site=30)
print(f"  upper-bound crops: {budget['max_crops_upper_bound']:,}")
print(f"  realistic (60% yield): {int(budget['max_crops_upper_bound'] * 0.60):,}")
print(f"  realistic (90% yield): {int(budget['max_crops_upper_bound'] * 0.90):,}")
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_donor_audit.ipynb
git commit -m "phase0: 03_donor_audit cells 10-11 (heatmap, summary block)"
```

---

## Task 9: Run the audit on Colab

**Files:**
- Modify (via Colab autosave): `notebooks/03_donor_audit.ipynb`

This task is human-driven, not automated. The user executes it on Colab.

- [ ] **Step 1: Push the latest commits to GitHub**

```bash
git push origin main
```

- [ ] **Step 2: User opens `notebooks/03_donor_audit.ipynb` on Colab**

Open the notebook in Colab via the badge in the notebook header. Runtime: Colab Free CPU is sufficient (no GPU needed).

- [ ] **Step 3: User runs all cells**

Runtime → Run all. Expected wall-clock: 3-5 minutes total, dominated by the manifest build in cell 5.

- [ ] **Step 4: User confirms autosave wrote the executed notebook back to GitHub**

Verify on github.com that `notebooks/03_donor_audit.ipynb` now has cell outputs populated. If the user's Colab autosave-to-GitHub is paused, manually save via File → Save a copy in GitHub.

- [ ] **Step 5: Pull the executed notebook locally**

```bash
git pull origin main
```

---

## Task 10: Write the audit report with explicit gate decision

**Files:**
- Create: `docs/superpowers/results/2026-05-12-phase-0-donor-audit.md`

- [ ] **Step 1: Copy the summary block from cell 11 of the executed notebook**

Open the executed `notebooks/03_donor_audit.ipynb` and locate the cell-11 stdout output (the `PHASE 0 AUDIT SUMMARY` block).

- [ ] **Step 2: Create the audit report**

Create `docs/superpowers/results/2026-05-12-phase-0-donor-audit.md`:

```markdown
# v0 Phase 0 Result — Donor Structure Audit

**Spec:** [`docs/superpowers/specs/2026-05-12-argus-cells-design.md`](../specs/2026-05-12-argus-cells-design.md), §3 + §5.
**Plan:** [`docs/superpowers/plans/2026-05-12-argus-cells-phase-0-donor-audit.md`](../plans/2026-05-12-argus-cells-phase-0-donor-audit.md).
**Notebook:** [`notebooks/03_donor_audit.ipynb`](../../../notebooks/03_donor_audit.ipynb).
**Run date:** 2026-05-XX
**Hardware:** Colab Free CPU.

## Quantitative findings

(Paste the `PHASE 0 AUDIT SUMMARY` stdout block from cell 11 here, formatted as a code block.)

```
PHASE 0 AUDIT SUMMARY
============================================================

Donor counts per condition:
  control: N donor lines  [OK/BLOCKER]
  deletion: M donor lines  [OK/BLOCKER]

Per-(cell_type, condition) imbalance (donor-balance CV):
  (...)

Crop budget at crops_per_site=30:
  upper-bound crops: ...
  realistic (60% yield): ...
  realistic (90% yield): ...
```

## Interpretation

(One short paragraph each:)

- **Donor coverage adequacy** — does each condition meet the N ≥ 3 gate?
- **Imbalance severity** — are any (cell_type, condition) groups dominated by a single donor (CV > 1.0)?
- **Crop budget** — at the chosen `crops_per_site` value, do we have enough crops for CCT-from-scratch (target ~50-100k)?

## Gate decision

One of:

- **PROCEED to Phase 1.** Donor structure meets the N ≥ 3 gate per condition. Imbalance is tolerable. Crop budget is sufficient at `crops_per_site=X`. Phase 1 begins with the predecessor's 0.73 baseline as the harness target.
- **ESCALATE to user.** Donor structure does not meet the N ≥ 3 gate, OR imbalance is severe (CV > 1.5 in any group), OR crop budget is insufficient at the maximum reasonable `crops_per_site`. Specific blocker(s): [list]. Recommended scope changes for user review: [list].

## Implications for Phase 1 + Phase 2 design

- Phase 1 (build interpretability harness on existing 0.73 baseline) is/is-not unblocked.
- Phase 2 crop-budget recommendation: `crops_per_site = X` to hit ~Y thousand crops.
- Phase 2 donor-probe configuration: N donors per condition observed; expected linear-probe random-baseline accuracy = 1/N.
```

- [ ] **Step 3: Fill in the report with the actual numbers from the notebook output**

Replace the placeholder values (`N`, `M`, `X`, `Y`, and the `[OK/BLOCKER]` flags) with the real numbers from the Colab run. Replace `2026-05-XX` with the actual run date. Pick the recommended `crops_per_site` based on what hits the ~50-100k target.

- [ ] **Step 4: Make the explicit gate decision**

Pick one of the two options under `## Gate decision` and delete the other. The decision MUST be unambiguous: PROCEED or ESCALATE. If ESCALATE, list specific blockers.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/results/2026-05-12-phase-0-donor-audit.md
git commit -m "phase0: donor audit report with gate decision (PROCEED|ESCALATE)"
```

---

## Task 11: Push and announce Phase 0 completion

**Files:** None (git operations only).

- [ ] **Step 1: Push all Phase 0 commits**

```bash
git push origin main
```

- [ ] **Step 2: Verify final state**

```bash
git log --oneline -15
ls docs/superpowers/results/
pytest tests/ -v
```

Expected:
- 11+ new commits since the spec commit (`777993a`).
- `2026-05-12-phase-0-donor-audit.md` present in results dir.
- `pytest tests/ -v` shows `10 passed`.

- [ ] **Step 3: Announce Phase 0 completion to the user**

Print the gate decision (PROCEED vs ESCALATE) and recommend whether to invoke `writing-plans` for the Phase 1 implementation plan.

If PROCEED → recommend: write Phase 1 plan next (interpretability harness build on the existing 0.73 baseline).

If ESCALATE → recommend: stop the planning workflow, surface the blocker for user decision, do not auto-advance to Phase 1.

---

## Out of scope for this plan (deferred to subsequent Phase plans)

- Phase 1: building the interpretability harness (GradCAM, attention rollout, IG, channel ablation, donor probe, cell-type stratification). Plan written after Phase 0 result lands.
- Phase 2: training Argus-RN34 + Argus-CCT production models. Plan written after Phase 1 harness validation lands.
- Phase 3: running the full attribution analysis on production models and writing the biology output.
- Phase 4: argus-cells repo migration, HF Hub artifact publication, README polish.

Each subsequent phase gets its own focused implementation plan once the predecessor phase completes.
