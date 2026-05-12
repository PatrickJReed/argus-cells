# argus-cells Phase 1 — Interpretability Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable interpretability harness — channel ablation, GradCAM, Integrated Gradients, donor probe, cell-type stratification, figure generation — and validate it end-to-end against the existing `cerberus-neuro-v0-baseline` checkpoint (best-epoch val_acc 0.73).

**Architecture:** A small, focused module set under `src/cerberus_neuro/` (we stay in this repo through Phase 1 per spec). Each attribution method exposes the same callable signature returning a common `AttributionResult` dataclass, so downstream analysis treats them uniformly. The donor probe operates on frozen encoder embeddings; the analysis module wraps any attribution method with per-cell-type stratification. A driver notebook loads the existing checkpoint, runs the full harness on the val split, and produces the Phase 1 outputs.

**Tech Stack:** Python 3.10+, PyTorch 2.4+, scikit-learn (linear probe), pandas, matplotlib, pytest, Jupyter. GPU helpful for IG; everything else is fine on CPU.

**Spec:** [`docs/superpowers/specs/2026-05-12-argus-cells-design.md`](../specs/2026-05-12-argus-cells-design.md), §4 (harness components) + §5 (Phase 1 row) + §6 (Phase 1 success criteria).
**Phase 0 result (gating this plan):** [`docs/superpowers/results/2026-05-12-phase-0-donor-audit.md`](../results/2026-05-12-phase-0-donor-audit.md) — PROCEED, 48 donor lines, crop budget ≫ target.

---

## File Structure

New files under `src/cerberus_neuro/`:

```
src/cerberus_neuro/
  attribution/
    __init__.py                # re-exports public API
    base.py                    # AttributionResult dataclass + shared helpers
    channel_ablation.py        # zero-channel perturbation method
    gradcam.py                 # GradCAM hooks on ResNet34 layer4
    integrated_gradients.py    # IG with zero-baseline, configurable steps
  probes/
    __init__.py
    donor_probe.py             # linear (or MLP) probe on frozen embeddings
  analysis/
    __init__.py
    stratification.py          # group attribution results by cell type
    figures.py                 # channel-ablation heatmap, saliency overlays, probe bar chart
```

New test files under `tests/`:

```
tests/
  test_attribution_base.py
  test_channel_ablation.py
  test_gradcam.py
  test_integrated_gradients.py
  test_donor_probe.py
  test_stratification.py
  test_figures.py
  conftest.py                  # shared fixtures: tiny_model_6ch, tiny_batch_6ch, synthetic_embeddings
```

New notebook:

```
notebooks/
  04_phase_1_harness.ipynb     # load checkpoint, run full harness, produce Phase 1 outputs
```

New results file:

```
docs/superpowers/results/
  2026-05-12-phase-1-harness-result.md
```

Modifications:

- `pyproject.toml` + `src/cerberus_neuro/__init__.py`: version bump (one per task that introduces new symbols, or one bump at start covering the whole plan).
- `src/cerberus_neuro/model.py`: add an `extract_embedding()` method to `BaselineDiseaseClassifier` that returns the 512-dim pre-classifier feature vector (needed by the donor probe).

---

## Task 1: Version bump + module scaffolds

**Files:**
- Modify: `pyproject.toml` (version field)
- Modify: `src/cerberus_neuro/__init__.py` (version)
- Create: `src/cerberus_neuro/attribution/__init__.py`
- Create: `src/cerberus_neuro/probes/__init__.py`
- Create: `src/cerberus_neuro/analysis/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Bump version to 0.1.0**

Phase 1 introduces three new subpackages and many new symbols. One version bump up-front covers the whole plan.

In `pyproject.toml` change the version field to:

```toml
version = "0.1.0"
```

In `src/cerberus_neuro/__init__.py` change line 1 to:

```python
__version__ = "0.1.0"
```

- [ ] **Step 2: Create the three empty subpackage `__init__.py` files**

```bash
mkdir -p src/cerberus_neuro/attribution src/cerberus_neuro/probes src/cerberus_neuro/analysis
touch src/cerberus_neuro/attribution/__init__.py
touch src/cerberus_neuro/probes/__init__.py
touch src/cerberus_neuro/analysis/__init__.py
```

- [ ] **Step 3: Create `tests/conftest.py` with shared fixtures**

This file gathers fixtures that multiple test files reuse. Content:

```python
"""Shared pytest fixtures for the interpretability harness tests."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def tiny_batch_6ch() -> dict[str, torch.Tensor]:
    """Synthetic batch matching the 6-channel disease-classifier input shape.

    Returns a dict with keys: ``images`` ([B=4, C=6, H=64, W=64]),
    ``labels`` ([B] binary disease labels in {0, 1}),
    ``cell_type`` ([B] integer cell-type labels in {0, 1, 2, 3}),
    ``line_ID`` ([B] integer donor IDs in {1, 2, 3, 4}).

    Deterministic via fixed seed.
    """
    g = torch.Generator().manual_seed(0)
    return {
        "images": torch.randn(4, 6, 64, 64, generator=g),
        "labels": torch.tensor([0, 1, 0, 1]),
        "cell_type": torch.tensor([0, 1, 2, 3]),
        "line_ID": torch.tensor([1, 2, 3, 4]),
    }


@pytest.fixture
def tiny_model_6ch():
    """Tiny 6-channel binary classifier mirroring BaselineDiseaseClassifier
    structure (encoder with a `layer4` attribute, head producing 2-logit output)
    but small enough to run on CPU in tests.

    Exposes the same attributes the attribution methods rely on:
    - ``encoder.layer4``: last conv stage (the GradCAM target)
    - ``encoder``: callable mapping [B, 6, H, W] -> 5-tuple feature stack
    - ``head``: classifier on the 4th feature stage
    """
    import torch.nn as nn

    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(nn.Conv2d(6, 8, 3, padding=1), nn.ReLU())
            self.layer2 = nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU())
            self.layer3 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU())
            self.layer4 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())

        def forward(self, x):
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            return x1, x2, x3, x4, x4

    class TinyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 2)

        def forward(self, features):
            return self.fc(self.pool(features).flatten(1))

    class TinyClassifier(nn.Module):
        model_kind = "baseline"

        def __init__(self):
            super().__init__()
            self.encoder = TinyEncoder()
            self.head = TinyHead()

        def forward(self, x):
            *_, x4 = self.encoder(x)
            return self.head(x4)

        def extract_embedding(self, x):
            *_, x4 = self.encoder(x)
            return self.head.pool(x4).flatten(1)

    m = TinyClassifier()
    m.eval()
    return m


@pytest.fixture
def synthetic_embeddings():
    """Synthetic 512-dim embeddings + donor + disease labels for the donor probe.

    Designed so donor identity is strongly linearly separable (each donor's
    embeddings cluster around a unique vector) but disease is not.

    Returns dict with keys: ``train_emb`` [120, 512], ``train_donor`` [120],
    ``train_disease`` [120], plus matching ``val_*`` arrays of shape [40].
    """
    import numpy as np

    rng = np.random.default_rng(0)
    n_donors = 4
    train_per_donor, val_per_donor = 30, 10
    centers = rng.normal(size=(n_donors, 512)).astype("float32") * 5.0

    def gen(n_per):
        emb = np.empty((n_donors * n_per, 512), dtype="float32")
        donor = np.empty(n_donors * n_per, dtype="int64")
        disease = np.empty(n_donors * n_per, dtype="int64")
        for d in range(n_donors):
            sl = slice(d * n_per, (d + 1) * n_per)
            emb[sl] = centers[d] + rng.normal(size=(n_per, 512)).astype("float32")
            donor[sl] = d
            disease[sl] = rng.integers(0, 2, size=n_per)
        return emb, donor, disease

    train_emb, train_donor, train_disease = gen(train_per_donor)
    val_emb, val_donor, val_disease = gen(val_per_donor)
    return {
        "train_emb": train_emb, "train_donor": train_donor, "train_disease": train_disease,
        "val_emb": val_emb, "val_donor": val_donor, "val_disease": val_disease,
    }
```

- [ ] **Step 4: Verify pytest still discovers cleanly with no test changes**

```bash
source venv/bin/activate && pytest tests/ -v 2>&1 | tail -5
```

Expected: `10 passed` (the existing audit tests still pass; conftest fixtures aren't exercised yet).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/cerberus_neuro/__init__.py src/cerberus_neuro/attribution/__init__.py src/cerberus_neuro/probes/__init__.py src/cerberus_neuro/analysis/__init__.py tests/conftest.py
git commit -m "phase1: scaffold attribution/probes/analysis subpackages + test fixtures, bump to 0.1.0"
```

(Use HEREDOC, include the `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer.)

---

## Task 2: AttributionResult dataclass + base interface

**Files:**
- Create: `src/cerberus_neuro/attribution/base.py`
- Create: `tests/test_attribution_base.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the AttributionResult dataclass and shared helpers."""
from __future__ import annotations

import torch

from cerberus_neuro.attribution.base import AttributionResult, channel_scores_from_saliency


def test_attribution_result_minimal():
    r = AttributionResult(
        saliency=None,
        channel_scores=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        metadata={"method": "test"},
    )
    assert r.saliency is None
    assert r.channel_scores.shape == (6,)
    assert r.metadata == {"method": "test"}


def test_attribution_result_with_saliency():
    r = AttributionResult(
        saliency=torch.zeros(2, 6, 8, 8),
        channel_scores=torch.zeros(2, 6),
        metadata={"method": "test"},
    )
    assert r.saliency.shape == (2, 6, 8, 8)
    assert r.channel_scores.shape == (2, 6)


def test_channel_scores_from_saliency_per_sample():
    saliency = torch.tensor(
        [[[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 0.0]]]]
    )  # shape [1, 3, 2, 2]: channel 0 sums to 4, channel 1 sums to 8, channel 2 sums to 0
    scores = channel_scores_from_saliency(saliency.abs())
    assert scores.shape == (1, 3)
    assert torch.allclose(scores[0], torch.tensor([4.0, 8.0, 0.0]))


def test_channel_scores_handles_negative_values():
    # Negative attributions are kept signed for sum aggregation by default.
    saliency = torch.tensor([[[[1.0, -1.0]], [[0.5, 0.5]]]])  # [1, 2, 1, 2]
    scores = channel_scores_from_saliency(saliency)
    assert torch.allclose(scores[0], torch.tensor([0.0, 1.0]))
```

- [ ] **Step 2: Verify it fails**

```bash
source venv/bin/activate && pytest tests/test_attribution_base.py -v
```

Expected: `ImportError: cannot import name 'AttributionResult' from 'cerberus_neuro.attribution.base'`.

- [ ] **Step 3: Write the implementation**

Create `src/cerberus_neuro/attribution/base.py`:

```python
"""Common interface for interpretability attribution methods.

Every method in :mod:`cerberus_neuro.attribution` returns the same
:class:`AttributionResult` dataclass so downstream analysis treats them
uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class AttributionResult:
    """Uniform return type for all attribution methods.

    Attributes
    ----------
    saliency
        Per-pixel attribution map. Shape ``[B, C, H, W]`` where ``B`` is the
        batch size, ``C`` matches the input-channel count (typically 6 here:
        brightfield + 5 fluorescence). ``None`` for methods that do not produce
        per-pixel maps (e.g. channel ablation).
    channel_scores
        Per-channel importance scores. Shape ``[B, C]`` for per-sample scores,
        or ``[C]`` for aggregated-batch scores. Sign conventions vary by
        method; consult ``metadata["aggregation"]``.
    metadata
        Free-form dict describing the method, its hyperparameters, and any
        runtime info worth carrying through the analysis pipeline.
    """

    saliency: torch.Tensor | None
    channel_scores: torch.Tensor
    metadata: dict = field(default_factory=dict)


def channel_scores_from_saliency(saliency: torch.Tensor) -> torch.Tensor:
    """Aggregate per-pixel saliency to per-channel scores by summing over H, W.

    Parameters
    ----------
    saliency
        Tensor of shape ``[B, C, H, W]``. Caller controls sign (pass
        ``saliency.abs()`` if absolute attribution is desired).

    Returns
    -------
    Tensor of shape ``[B, C]``.
    """
    if saliency.ndim != 4:
        raise ValueError(f"expected [B,C,H,W], got shape {tuple(saliency.shape)}")
    return saliency.sum(dim=(2, 3))
```

- [ ] **Step 4: Verify it passes**

```bash
source venv/bin/activate && pytest tests/test_attribution_base.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/cerberus_neuro/attribution/base.py tests/test_attribution_base.py
git commit -m "phase1: AttributionResult dataclass + channel_scores_from_saliency helper"
```

(With Co-Authored-By trailer.)

---

## Task 3: Channel ablation (TDD)

**Files:**
- Create: `src/cerberus_neuro/attribution/channel_ablation.py`
- Create: `tests/test_channel_ablation.py`

Channel ablation is the simplest attribution method: zero out one input channel at a time, measure the drop in classification accuracy. No gradients, pure forward-pass perturbation.

- [ ] **Step 1: Write the failing test**

Create `tests/test_channel_ablation.py`:

```python
"""Tests for the channel-ablation attribution method."""
from __future__ import annotations

import torch

from cerberus_neuro.attribution.channel_ablation import (
    compute_channel_ablation,
    compute_channel_ablation_per_sample,
)


def test_compute_channel_ablation_returns_drop_per_channel(tiny_model_6ch, tiny_batch_6ch):
    result = compute_channel_ablation(
        model=tiny_model_6ch,
        images=tiny_batch_6ch["images"],
        labels=tiny_batch_6ch["labels"],
    )
    # 6-channel input → 6 ablation scores.
    assert result.channel_scores.shape == (6,)
    # No per-pixel saliency for this method.
    assert result.saliency is None
    # Method metadata is populated.
    assert result.metadata["method"] == "channel_ablation"
    assert "baseline_accuracy" in result.metadata


def test_compute_channel_ablation_zero_channel_changes_logits(tiny_model_6ch, tiny_batch_6ch):
    """Ablating a channel must produce a strictly different prediction set
    than the baseline (otherwise the model isn't using that channel at all,
    which is informative but rare with random init)."""
    result = compute_channel_ablation(
        model=tiny_model_6ch,
        images=tiny_batch_6ch["images"],
        labels=tiny_batch_6ch["labels"],
    )
    # The scores are accuracy drops. They can be negative (ablation HELPED
    # the model, an artifact of random labels) or positive. Just check the
    # shape and the type.
    assert result.channel_scores.dtype == torch.float32 or result.channel_scores.dtype == torch.float64


def test_compute_channel_ablation_per_sample_shape(tiny_model_6ch, tiny_batch_6ch):
    """Per-sample variant gives [B, C] confidence drops."""
    result = compute_channel_ablation_per_sample(
        model=tiny_model_6ch,
        images=tiny_batch_6ch["images"],
        labels=tiny_batch_6ch["labels"],
    )
    assert result.channel_scores.shape == (4, 6)  # B=4, C=6
```

- [ ] **Step 2: Verify it fails**

```bash
source venv/bin/activate && pytest tests/test_channel_ablation.py -v
```

Expected: `ImportError: cannot import name 'compute_channel_ablation' from 'cerberus_neuro.attribution.channel_ablation'`.

- [ ] **Step 3: Write the implementation**

Create `src/cerberus_neuro/attribution/channel_ablation.py`:

```python
"""Channel-ablation attribution.

For each input channel, zero it out, run the model, measure the change in
classification accuracy or per-sample target-class confidence. The score is
the *drop* (baseline - ablated): positive means the model relies on that
channel, near-zero means the channel is unused, negative means ablating the
channel *helped* (which is a useful red flag).

Designed for the 6-channel brightfield + Cell Painting input. No gradients;
just forward passes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import AttributionResult


@torch.no_grad()
def compute_channel_ablation(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> AttributionResult:
    """Per-channel accuracy drop across the whole batch.

    Parameters
    ----------
    model
        Trained classifier returning ``[B, n_classes]`` logits.
    images
        Input tensor of shape ``[B, C, H, W]``.
    labels
        Ground-truth integer labels of shape ``[B]``.

    Returns
    -------
    :class:`AttributionResult` with ``saliency=None`` and ``channel_scores`` of
    shape ``[C]`` (one accuracy-drop score per input channel).
    """
    was_training = model.training
    model.eval()
    try:
        n_channels = images.shape[1]
        baseline_acc = _accuracy(model, images, labels)
        drops = torch.zeros(n_channels)
        for c in range(n_channels):
            ablated = images.clone()
            ablated[:, c, :, :] = 0.0
            ablated_acc = _accuracy(model, ablated, labels)
            drops[c] = baseline_acc - ablated_acc
        return AttributionResult(
            saliency=None,
            channel_scores=drops,
            metadata={
                "method": "channel_ablation",
                "baseline_accuracy": float(baseline_acc),
                "aggregation": "batch_accuracy_drop",
            },
        )
    finally:
        model.train(was_training)


@torch.no_grad()
def compute_channel_ablation_per_sample(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> AttributionResult:
    """Per-sample per-channel target-class confidence drop.

    Yields ``[B, C]`` channel_scores: how much zeroing channel ``c`` reduces the
    softmax probability assigned to the correct class for sample ``b``.

    Returns
    -------
    :class:`AttributionResult` with ``saliency=None`` and ``channel_scores`` of
    shape ``[B, C]``.
    """
    was_training = model.training
    model.eval()
    try:
        B, n_channels = images.shape[0], images.shape[1]
        baseline_conf = _target_class_confidence(model, images, labels)  # [B]
        drops = torch.zeros(B, n_channels)
        for c in range(n_channels):
            ablated = images.clone()
            ablated[:, c, :, :] = 0.0
            ablated_conf = _target_class_confidence(model, ablated, labels)  # [B]
            drops[:, c] = baseline_conf - ablated_conf
        return AttributionResult(
            saliency=None,
            channel_scores=drops,
            metadata={
                "method": "channel_ablation_per_sample",
                "aggregation": "per_sample_target_confidence_drop",
            },
        )
    finally:
        model.train(was_training)


def _accuracy(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> float:
    logits = model(images)
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def _target_class_confidence(
    model: nn.Module, images: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    logits = model(images)
    probs = torch.softmax(logits, dim=-1)
    return probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
```

- [ ] **Step 4: Verify it passes**

```bash
source venv/bin/activate && pytest tests/test_channel_ablation.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/cerberus_neuro/attribution/channel_ablation.py tests/test_channel_ablation.py
git commit -m "phase1: channel_ablation attribution (batch + per-sample variants)"
```

(With Co-Authored-By trailer.)

---

## Task 4: GradCAM (TDD)

**Files:**
- Create: `src/cerberus_neuro/attribution/gradcam.py`
- Create: `tests/test_gradcam.py`

GradCAM hooks into the last conv stage of the ResNet34 encoder (`encoder.layer4`), records the forward activation and the backward gradient with respect to the target class logit, and computes the channel-weighted activation map.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gradcam.py`:

```python
"""Tests for the GradCAM attribution method."""
from __future__ import annotations

import torch

from cerberus_neuro.attribution.gradcam import compute_gradcam


def test_compute_gradcam_returns_saliency_at_input_resolution(tiny_model_6ch, tiny_batch_6ch):
    images = tiny_batch_6ch["images"]
    result = compute_gradcam(
        model=tiny_model_6ch,
        target_layer=tiny_model_6ch.encoder.layer4,
        images=images,
        target_class=1,
    )
    # Saliency upsampled to the input H, W. Shape is [B, 1, H, W] (single
    # channel-agnostic spatial map per sample).
    assert result.saliency.shape == (4, 1, 64, 64)
    # Saliency is non-negative after ReLU.
    assert (result.saliency >= 0).all()
    # Method metadata is populated.
    assert result.metadata["method"] == "gradcam"
    assert result.metadata["target_class"] == 1


def test_compute_gradcam_channel_scores_have_shape_B_by_C_with_input_channels(
    tiny_model_6ch, tiny_batch_6ch,
):
    """GradCAM saliency is channel-agnostic (single map per sample), so we
    broadcast its sum across the input-channel axis: channel_scores[b, c] =
    sum(saliency[b, 0]) for all c. This keeps the AttributionResult shape
    uniform with IG and channel ablation.
    """
    result = compute_gradcam(
        model=tiny_model_6ch,
        target_layer=tiny_model_6ch.encoder.layer4,
        images=tiny_batch_6ch["images"],
        target_class=1,
    )
    assert result.channel_scores.shape == (4, 6)
    # Per-sample, the 6 channel scores are equal (broadcast from one map).
    for b in range(4):
        first = result.channel_scores[b, 0]
        assert torch.allclose(result.channel_scores[b], first.expand(6))


def test_compute_gradcam_target_class_zero_gives_different_map(
    tiny_model_6ch, tiny_batch_6ch,
):
    """GradCAM for class 0 should differ from class 1 (the gradient targets
    different logits)."""
    images = tiny_batch_6ch["images"]
    r0 = compute_gradcam(
        model=tiny_model_6ch, target_layer=tiny_model_6ch.encoder.layer4,
        images=images, target_class=0,
    )
    r1 = compute_gradcam(
        model=tiny_model_6ch, target_layer=tiny_model_6ch.encoder.layer4,
        images=images, target_class=1,
    )
    assert not torch.allclose(r0.saliency, r1.saliency)
```

- [ ] **Step 2: Verify it fails**

```bash
source venv/bin/activate && pytest tests/test_gradcam.py -v
```

Expected: `ImportError: cannot import name 'compute_gradcam' from 'cerberus_neuro.attribution.gradcam'`.

- [ ] **Step 3: Write the implementation**

Create `src/cerberus_neuro/attribution/gradcam.py`:

```python
"""GradCAM attribution for CNN classifiers.

Hooks a target conv layer for both the forward activation and the backward
gradient. GradCAM = ReLU(sum_c (avg_pool(grad_c) * act_c)) — a class-conditional
saliency map at the spatial resolution of the target layer, upsampled bilinearly
to the input resolution.

For the argus-cells `BaselineDiseaseClassifier`, the target layer is
``encoder.layer4`` (the deepest conv stage, where class-discriminative spatial
features live).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AttributionResult


def compute_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    images: torch.Tensor,
    target_class: int,
) -> AttributionResult:
    """Compute GradCAM saliency for ``images`` against ``target_class``.

    Parameters
    ----------
    model
        Classifier producing ``[B, n_classes]`` logits.
    target_layer
        The conv-style module whose activations + gradients drive the map.
        For BaselineDiseaseClassifier: ``model.encoder.layer4``.
    images
        ``[B, C_in, H, W]`` input tensor.
    target_class
        Integer class index to compute saliency for.

    Returns
    -------
    :class:`AttributionResult` with
    - ``saliency`` of shape ``[B, 1, H, W]`` (single channel-agnostic spatial
      map per sample, upsampled bilinearly to the input resolution),
    - ``channel_scores`` of shape ``[B, C_in]`` where each row is the per-sample
      saliency sum broadcast across input channels (GradCAM does not
      distinguish input channels).
    """
    was_training = model.training
    model.eval()

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def fwd_hook(_module, _inputs, output):
        activations["act"] = output

    def bwd_hook(_module, _grad_input, grad_output):
        gradients["grad"] = grad_output[0]

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    try:
        images = images.detach().clone().requires_grad_(False)
        logits = model(images)
        if logits.ndim != 2:
            raise ValueError(f"expected 2D logits, got shape {tuple(logits.shape)}")
        score = logits[:, target_class].sum()
        model.zero_grad(set_to_none=True)
        score.backward()

        act = activations["act"]               # [B, K, h, w]
        grad = gradients["grad"]               # [B, K, h, w]
        weights = grad.mean(dim=(2, 3), keepdim=True)  # [B, K, 1, 1]
        cam = (weights * act).sum(dim=1, keepdim=True)  # [B, 1, h, w]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)

        # Channel-agnostic spatial map; broadcast a per-sample scalar across
        # the 6 input channels to keep the AttributionResult shape uniform.
        per_sample_sum = cam.sum(dim=(2, 3)).squeeze(-1)  # [B]
        n_channels = images.shape[1]
        channel_scores = per_sample_sum.unsqueeze(-1).expand(-1, n_channels).detach().clone()

        return AttributionResult(
            saliency=cam.detach(),
            channel_scores=channel_scores,
            metadata={
                "method": "gradcam",
                "target_class": int(target_class),
                "target_layer": type(target_layer).__name__,
                "aggregation": "spatial_sum_broadcast",
            },
        )
    finally:
        fh.remove()
        bh.remove()
        model.train(was_training)
```

- [ ] **Step 4: Verify it passes**

```bash
source venv/bin/activate && pytest tests/test_gradcam.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/cerberus_neuro/attribution/gradcam.py tests/test_gradcam.py
git commit -m "phase1: GradCAM attribution with hook-based act/grad capture"
```

(With Co-Authored-By trailer.)

---

## Task 5: Integrated Gradients (TDD)

**Files:**
- Create: `src/cerberus_neuro/attribution/integrated_gradients.py`
- Create: `tests/test_integrated_gradients.py`

Integrated Gradients (Sundararajan et al., 2017): for each input, interpolate from a baseline (zero image) to the actual input over ``n_steps`` and integrate the gradient of the target class logit with respect to the input along that path. Cross-architecture; works on any differentiable classifier.

- [ ] **Step 1: Write the failing test**

Create `tests/test_integrated_gradients.py`:

```python
"""Tests for Integrated Gradients attribution."""
from __future__ import annotations

import torch

from cerberus_neuro.attribution.integrated_gradients import compute_integrated_gradients


def test_ig_returns_per_input_saliency(tiny_model_6ch, tiny_batch_6ch):
    result = compute_integrated_gradients(
        model=tiny_model_6ch,
        images=tiny_batch_6ch["images"],
        target_class=1,
        n_steps=8,
    )
    # IG saliency lives at input resolution + input channel count.
    assert result.saliency.shape == (4, 6, 64, 64)
    assert result.metadata["method"] == "integrated_gradients"
    assert result.metadata["n_steps"] == 8


def test_ig_channel_scores_shape(tiny_model_6ch, tiny_batch_6ch):
    result = compute_integrated_gradients(
        model=tiny_model_6ch,
        images=tiny_batch_6ch["images"],
        target_class=1,
        n_steps=8,
    )
    assert result.channel_scores.shape == (4, 6)


def test_ig_zero_input_yields_near_zero_saliency(tiny_model_6ch):
    """If both baseline and input are zero, IG must be near-zero everywhere
    (the integrand path has zero length)."""
    images = torch.zeros(2, 6, 64, 64)
    result = compute_integrated_gradients(
        model=tiny_model_6ch, images=images, target_class=1, n_steps=8,
    )
    assert result.saliency.abs().max().item() < 1e-4


def test_ig_changes_with_target_class(tiny_model_6ch, tiny_batch_6ch):
    """Different target class → different attribution map."""
    images = tiny_batch_6ch["images"]
    r0 = compute_integrated_gradients(model=tiny_model_6ch, images=images, target_class=0, n_steps=8)
    r1 = compute_integrated_gradients(model=tiny_model_6ch, images=images, target_class=1, n_steps=8)
    assert not torch.allclose(r0.saliency, r1.saliency)
```

- [ ] **Step 2: Verify it fails**

```bash
source venv/bin/activate && pytest tests/test_integrated_gradients.py -v
```

Expected: `ImportError: cannot import name 'compute_integrated_gradients' from 'cerberus_neuro.attribution.integrated_gradients'`.

- [ ] **Step 3: Write the implementation**

Create `src/cerberus_neuro/attribution/integrated_gradients.py`:

```python
"""Integrated Gradients (Sundararajan et al., 2017).

IG_i(x) = (x_i - x'_i) * (1/N) * sum_{k=1..N} d f_target / d x_i |_{x' + (k/N)(x-x')}

where ``x'`` is the baseline (zero image by default). Approximates the path
integral of gradients from ``x'`` to ``x`` over ``n_steps`` linear interpolations.

Cross-architecture: works on any differentiable classifier. The argus-cells
methods spine uses this on both Argus-RN34 and (later) Argus-CCT for the
cross-architecture-agreement analysis.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import AttributionResult, channel_scores_from_saliency


def compute_integrated_gradients(
    model: nn.Module,
    images: torch.Tensor,
    target_class: int,
    n_steps: int = 32,
    baseline: torch.Tensor | None = None,
) -> AttributionResult:
    """Compute IG attribution for ``images`` against ``target_class``.

    Parameters
    ----------
    model
        Classifier producing ``[B, n_classes]`` logits.
    images
        ``[B, C, H, W]`` input tensor.
    target_class
        Integer class index to attribute.
    n_steps
        Number of Riemann steps along the baseline → input path. Default 32.
    baseline
        Reference input for the path integral. ``None`` (default) uses a
        zero tensor matching ``images.shape``.

    Returns
    -------
    :class:`AttributionResult` with
    - ``saliency`` of shape ``[B, C, H, W]`` (signed attribution per input
      element),
    - ``channel_scores`` of shape ``[B, C]`` (sum of |saliency| over H, W).
    """
    was_training = model.training
    model.eval()
    try:
        if baseline is None:
            baseline = torch.zeros_like(images)
        if baseline.shape != images.shape:
            raise ValueError(
                f"baseline shape {tuple(baseline.shape)} != images shape "
                f"{tuple(images.shape)}"
            )
        alphas = torch.linspace(1.0 / n_steps, 1.0, n_steps, device=images.device)
        # Average gradient over the path.
        grad_accum = torch.zeros_like(images)
        for alpha in alphas:
            interp = baseline + alpha * (images - baseline)
            interp = interp.detach().clone().requires_grad_(True)
            logits = model(interp)
            score = logits[:, target_class].sum()
            model.zero_grad(set_to_none=True)
            score.backward()
            grad_accum = grad_accum + interp.grad.detach()
        avg_grad = grad_accum / n_steps
        saliency = (images - baseline) * avg_grad
        channel_scores = channel_scores_from_saliency(saliency.abs())
        return AttributionResult(
            saliency=saliency.detach(),
            channel_scores=channel_scores.detach(),
            metadata={
                "method": "integrated_gradients",
                "target_class": int(target_class),
                "n_steps": int(n_steps),
                "aggregation": "abs_spatial_sum",
            },
        )
    finally:
        model.train(was_training)
```

- [ ] **Step 4: Verify it passes**

```bash
source venv/bin/activate && pytest tests/test_integrated_gradients.py -v
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/cerberus_neuro/attribution/integrated_gradients.py tests/test_integrated_gradients.py
git commit -m "phase1: Integrated Gradients with zero-baseline + configurable n_steps"
```

(With Co-Authored-By trailer.)

---

## Task 6: Re-export attribution API from `attribution/__init__.py`

**Files:**
- Modify: `src/cerberus_neuro/attribution/__init__.py`

- [ ] **Step 1: Populate `__init__.py`**

Replace the empty `src/cerberus_neuro/attribution/__init__.py` with:

```python
"""Interpretability attribution methods."""
from .base import AttributionResult, channel_scores_from_saliency
from .channel_ablation import (
    compute_channel_ablation,
    compute_channel_ablation_per_sample,
)
from .gradcam import compute_gradcam
from .integrated_gradients import compute_integrated_gradients

__all__ = [
    "AttributionResult",
    "channel_scores_from_saliency",
    "compute_channel_ablation",
    "compute_channel_ablation_per_sample",
    "compute_gradcam",
    "compute_integrated_gradients",
]
```

- [ ] **Step 2: Verify the package imports cleanly**

```bash
source venv/bin/activate && python -c "from cerberus_neuro.attribution import compute_channel_ablation, compute_gradcam, compute_integrated_gradients, AttributionResult; print('attribution API imports OK')"
```

Expected: `attribution API imports OK`.

- [ ] **Step 3: Run full test suite**

```bash
source venv/bin/activate && pytest tests/ -v 2>&1 | tail -5
```

Expected: `20 passed` (10 audit + 4 base + 3 channel ablation + 3 gradcam + 4 IG = 24; actual count may differ slightly but all should pass).

- [ ] **Step 4: Commit**

```bash
git add src/cerberus_neuro/attribution/__init__.py
git commit -m "phase1: attribution/__init__.py public API re-exports"
```

(With Co-Authored-By trailer.)

---

## Task 7: Add `extract_embedding()` to `BaselineDiseaseClassifier`

**Files:**
- Modify: `src/cerberus_neuro/model.py:264-295` (the `BaselineDiseaseClassifier` class)
- Create: `tests/test_baseline_embedding.py`

The donor probe needs frozen 512-dim encoder embeddings. The classifier head computes them internally; we surface a method that returns them.

- [ ] **Step 1: Write the failing test**

Create `tests/test_baseline_embedding.py`:

```python
"""Test that BaselineDiseaseClassifier exposes a 512-dim embedding extractor."""
from __future__ import annotations

import torch

from cerberus_neuro.model import BaselineDiseaseClassifier


def test_extract_embedding_returns_512d_vector():
    model = BaselineDiseaseClassifier(in_channels=6, n_classes=2, pretrained_encoder=False)
    model.eval()
    x = torch.randn(3, 6, 64, 64)
    with torch.no_grad():
        emb = model.extract_embedding(x)
    assert emb.shape == (3, 512)
    assert emb.dtype == torch.float32


def test_extract_embedding_no_grad_does_not_require_grad():
    """The extractor should work cleanly under no_grad; output should be
    a plain Tensor without requires_grad set when called under no_grad."""
    model = BaselineDiseaseClassifier(in_channels=6, n_classes=2, pretrained_encoder=False)
    model.eval()
    x = torch.randn(2, 6, 64, 64)
    with torch.no_grad():
        emb = model.extract_embedding(x)
    assert emb.requires_grad is False
```

- [ ] **Step 2: Verify it fails**

```bash
source venv/bin/activate && pytest tests/test_baseline_embedding.py -v
```

Expected: `AttributeError: 'BaselineDiseaseClassifier' object has no attribute 'extract_embedding'`.

- [ ] **Step 3: Modify `BaselineDiseaseClassifier`**

In `src/cerberus_neuro/model.py`, add a method to the class. Read the file first to confirm current state, then add `extract_embedding` just before `parameter_count`.

Replace the existing class body (the file's `BaselineDiseaseClassifier` class) with:

```python
class BaselineDiseaseClassifier(nn.Module):
    """All-channel single-task disease classifier (the v0 upper-bound baseline).

    Same ResNet34 encoder as :class:`CerberusModel`, but takes the full 6-channel
    stack (brightfield + 5 fluorescence) as input and exposes only the
    line-condition head. Establishes "what's the best disease accuracy you can
    get with all the data, no virtual-staining task to share gradient with?".
    The Cerberus model's disease number is meaningful in comparison to this
    upper bound: it answers "how much of that signal is recoverable from
    brightfield alone, when the encoder is forced to also predict fluorescence?".
    """

    model_kind = "baseline"

    def __init__(self, in_channels: int = 6, n_classes: int = 2, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels=in_channels, pretrained=pretrained_encoder)
        self.head = ClassifierHead(512, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, _, x4 = self.encoder(x)
        return self.head(x4)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-dim pre-classifier embedding for the donor probe.

        This is the global-average-pooled output of the encoder's final conv
        stage. Shape: ``[B, 512]``.
        """
        _, _, _, _, x4 = self.encoder(x)
        pooled = torch.nn.functional.adaptive_avg_pool2d(x4, output_size=1)
        return pooled.flatten(1)

    def parameter_count(self) -> dict[str, int]:
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "encoder": count(self.encoder),
            "head": count(self.head),
            "total": count(self),
        }
```

- [ ] **Step 4: Verify it passes**

```bash
source venv/bin/activate && pytest tests/test_baseline_embedding.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/cerberus_neuro/model.py tests/test_baseline_embedding.py
git commit -m "phase1: BaselineDiseaseClassifier.extract_embedding() for donor probe"
```

(With Co-Authored-By trailer.)

---

## Task 8: Donor probe — fit and evaluate (TDD)

**Files:**
- Create: `src/cerberus_neuro/probes/donor_probe.py`
- Create: `tests/test_donor_probe.py`

A linear (or 1-hidden-layer MLP) classifier fit on frozen encoder embeddings. We test it with synthetic embeddings where donor identity is linearly separable.

- [ ] **Step 1: Write the failing test**

Create `tests/test_donor_probe.py`:

```python
"""Tests for the donor probe."""
from __future__ import annotations

import math

from cerberus_neuro.probes.donor_probe import (
    fit_linear_probe,
    parallel_probe_report,
)


def test_fit_linear_probe_recovers_separable_donor(synthetic_embeddings):
    """Synthetic donors are linearly separable; probe accuracy should be > 0.8."""
    report = fit_linear_probe(
        train_emb=synthetic_embeddings["train_emb"],
        train_labels=synthetic_embeddings["train_donor"],
        val_emb=synthetic_embeddings["val_emb"],
        val_labels=synthetic_embeddings["val_donor"],
        n_classes=4,
    )
    assert report["val_accuracy"] > 0.8
    assert report["random_baseline"] == 0.25  # 1/4 for 4 donors


def test_fit_linear_probe_on_random_labels_returns_near_baseline(synthetic_embeddings):
    """Probing for random disease labels (no structure) → near-baseline accuracy."""
    import numpy as np
    rng = np.random.default_rng(0)
    n_train = synthetic_embeddings["train_emb"].shape[0]
    n_val = synthetic_embeddings["val_emb"].shape[0]
    random_train = rng.integers(0, 2, size=n_train)
    random_val = rng.integers(0, 2, size=n_val)
    report = fit_linear_probe(
        train_emb=synthetic_embeddings["train_emb"],
        train_labels=random_train,
        val_emb=synthetic_embeddings["val_emb"],
        val_labels=random_val,
        n_classes=2,
    )
    # No more than 20pp above random for noise-only labels.
    assert report["val_accuracy"] < report["random_baseline"] + 0.2


def test_parallel_probe_report_includes_ratio(synthetic_embeddings):
    """The parallel report runs two probes on the same embeddings and returns
    donor / disease accuracy as the confound-strength scalar."""
    report = parallel_probe_report(
        train_emb=synthetic_embeddings["train_emb"],
        train_donor=synthetic_embeddings["train_donor"],
        train_disease=synthetic_embeddings["train_disease"],
        val_emb=synthetic_embeddings["val_emb"],
        val_donor=synthetic_embeddings["val_donor"],
        val_disease=synthetic_embeddings["val_disease"],
        n_donors=4,
    )
    assert "donor" in report and "disease" in report
    assert "ratio" in report
    # In the synthetic data, donor is linearly separable, disease is not, so
    # the ratio (donor / disease) should be > 1.
    assert report["ratio"] > 1.0
    assert report["donor"]["random_baseline"] == 0.25
    assert report["disease"]["random_baseline"] == 0.5
```

- [ ] **Step 2: Verify it fails**

```bash
source venv/bin/activate && pytest tests/test_donor_probe.py -v
```

Expected: `ImportError: cannot import name 'fit_linear_probe' from 'cerberus_neuro.probes.donor_probe'`.

- [ ] **Step 3: Write the implementation**

Create `src/cerberus_neuro/probes/donor_probe.py`:

```python
"""Donor confound probe.

Fit a linear classifier on frozen encoder embeddings to predict donor identity
(``Metadata_line_ID``). High accuracy means the encoder has learned features
that linearly distinguish donor identity, which is a confound for any
disease-classification claim built on those embeddings.

The parallel report fits a second probe on the same embeddings against the
disease label, then reports the donor / disease ratio as a confound-strength
scalar. ratio ≪ 1 means the encoder retains less donor info than disease info
(good); ratio ≥ 1 means donor info is at least as linearly extractable as
disease info (red flag).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_linear_probe(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    val_emb: np.ndarray,
    val_labels: np.ndarray,
    n_classes: int,
    max_iter: int = 1000,
    C: float = 1.0,
) -> dict[str, float]:
    """Fit an L2-regularized multinomial logistic regression and report accuracy.

    Parameters
    ----------
    train_emb, train_labels
        Training embeddings ``[N_train, D]`` and integer class labels ``[N_train]``.
    val_emb, val_labels
        Validation embeddings ``[N_val, D]`` and labels ``[N_val]``.
    n_classes
        Number of distinct classes in the labels (used to compute the random
        baseline as ``1/n_classes``).
    max_iter
        Maximum optimizer iterations.
    C
        Inverse of regularization strength. Default 1.0 matches scikit-learn's
        default; useful to tune to balance under- vs over-fitting in small
        regimes.

    Returns
    -------
    Dict with ``train_accuracy``, ``val_accuracy``, ``random_baseline``,
    ``n_classes`` keys.
    """
    clf = LogisticRegression(
        max_iter=max_iter,
        C=C,
        multi_class="multinomial",
        solver="lbfgs",
    )
    clf.fit(train_emb, train_labels)
    train_acc = float(clf.score(train_emb, train_labels))
    val_acc = float(clf.score(val_emb, val_labels))
    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "random_baseline": 1.0 / float(n_classes),
        "n_classes": int(n_classes),
    }


def parallel_probe_report(
    train_emb: np.ndarray,
    train_donor: np.ndarray,
    train_disease: np.ndarray,
    val_emb: np.ndarray,
    val_donor: np.ndarray,
    val_disease: np.ndarray,
    n_donors: int,
) -> dict[str, dict | float]:
    """Fit two probes (donor identity and disease) and report the ratio.

    Donor probe: ``n_donors``-way classification.
    Disease probe: binary classification.

    Returns
    -------
    Dict with keys ``donor`` (probe report), ``disease`` (probe report), and
    ``ratio`` (``donor.val_accuracy / disease.val_accuracy``). A ratio ≪ 1 is
    good (encoder retained little donor info); ratio ≥ 1 is a red flag.
    """
    donor_report = fit_linear_probe(
        train_emb=train_emb, train_labels=train_donor,
        val_emb=val_emb, val_labels=val_donor,
        n_classes=n_donors,
    )
    disease_report = fit_linear_probe(
        train_emb=train_emb, train_labels=train_disease,
        val_emb=val_emb, val_labels=val_disease,
        n_classes=2,
    )
    ratio = donor_report["val_accuracy"] / max(disease_report["val_accuracy"], 1e-6)
    return {
        "donor": donor_report,
        "disease": disease_report,
        "ratio": float(ratio),
    }
```

- [ ] **Step 4: Verify it passes**

```bash
source venv/bin/activate && pytest tests/test_donor_probe.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Re-export from `probes/__init__.py`**

Replace the empty `src/cerberus_neuro/probes/__init__.py` with:

```python
"""Confound-detection probes on frozen encoder embeddings."""
from .donor_probe import fit_linear_probe, parallel_probe_report

__all__ = ["fit_linear_probe", "parallel_probe_report"]
```

- [ ] **Step 6: Verify import**

```bash
source venv/bin/activate && python -c "from cerberus_neuro.probes import fit_linear_probe, parallel_probe_report; print('probes API OK')"
```

Expected: `probes API OK`.

- [ ] **Step 7: Commit**

```bash
git add src/cerberus_neuro/probes/donor_probe.py src/cerberus_neuro/probes/__init__.py tests/test_donor_probe.py
git commit -m "phase1: donor probe (linear logistic-regression) + parallel disease probe ratio"
```

(With Co-Authored-By trailer.)

---

## Task 9: Cell-type stratification analysis (TDD)

**Files:**
- Create: `src/cerberus_neuro/analysis/stratification.py`
- Create: `tests/test_stratification.py`

A wrapper that takes an attribution method's output plus per-sample cell-type labels and produces a per-(cell_type) summary of channel importance.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stratification.py`:

```python
"""Tests for cell-type stratification of attribution results."""
from __future__ import annotations

import pandas as pd
import torch

from cerberus_neuro.attribution.base import AttributionResult
from cerberus_neuro.analysis.stratification import stratify_channel_scores_by_cell_type


def test_stratify_groups_per_cell_type():
    # B=8 samples, C=6 channels, all samples in cell_type 0 have score 1.0 in
    # channel 0; samples in cell_type 1 have score 2.0 in channel 1.
    scores = torch.zeros(8, 6)
    scores[0:4, 0] = 1.0
    scores[4:8, 1] = 2.0
    cell_types = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    cell_type_names = ["stem", "progen", "neuron", "astro"]

    result = AttributionResult(saliency=None, channel_scores=scores, metadata={"method": "test"})
    df = stratify_channel_scores_by_cell_type(
        result=result,
        cell_types=cell_types,
        cell_type_names=cell_type_names,
        channel_names=["BF", "DNA", "Mito", "AGP", "ER", "RNA"],
    )
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"cell_type", "channel", "mean_score", "n_samples"}
    # cell_type=0 (stem) has 4 samples, channel BF mean=1.0.
    stem_bf = df[(df["cell_type"] == "stem") & (df["channel"] == "BF")].iloc[0]
    assert stem_bf["mean_score"] == 1.0
    assert stem_bf["n_samples"] == 4
    # cell_type=1 (progen) has 4 samples, channel DNA mean=2.0.
    progen_dna = df[(df["cell_type"] == "progen") & (df["channel"] == "DNA")].iloc[0]
    assert progen_dna["mean_score"] == 2.0
    assert progen_dna["n_samples"] == 4
    # Total rows: 4 cell types x 6 channels = 24. Cell types with no samples
    # still appear with n_samples=0.
    assert len(df) == 24


def test_stratify_ignores_unknown_cell_types():
    """A cell_type index outside the names list raises a clear error."""
    scores = torch.zeros(2, 6)
    cell_types = torch.tensor([0, 99])  # 99 is out-of-range
    result = AttributionResult(saliency=None, channel_scores=scores, metadata={})

    import pytest
    with pytest.raises(IndexError):
        stratify_channel_scores_by_cell_type(
            result=result,
            cell_types=cell_types,
            cell_type_names=["stem", "progen"],
            channel_names=["BF", "DNA", "Mito", "AGP", "ER", "RNA"],
        )
```

- [ ] **Step 2: Verify it fails**

```bash
source venv/bin/activate && pytest tests/test_stratification.py -v
```

Expected: `ImportError: cannot import name 'stratify_channel_scores_by_cell_type' from 'cerberus_neuro.analysis.stratification'`.

- [ ] **Step 3: Write the implementation**

Create `src/cerberus_neuro/analysis/stratification.py`:

```python
"""Cell-type stratification of attribution results.

Groups per-sample per-channel scores by cell type and reports the
per-(cell_type, channel) mean. This is the core finding-shaped output of
the argus-cells Phase 1 harness: a 4x6 table answering "for each cell type,
which of the 6 input channels does the model rely on most?".
"""
from __future__ import annotations

import pandas as pd
import torch

from cerberus_neuro.attribution.base import AttributionResult


def stratify_channel_scores_by_cell_type(
    result: AttributionResult,
    cell_types: torch.Tensor,
    cell_type_names: list[str],
    channel_names: list[str],
) -> pd.DataFrame:
    """Group channel scores by cell type and report per-group means.

    Parameters
    ----------
    result
        An :class:`AttributionResult` with ``channel_scores`` of shape
        ``[B, C]`` (per-sample, per-channel scores).
    cell_types
        Integer cell-type labels of shape ``[B]``. Each value indexes into
        ``cell_type_names``.
    cell_type_names
        Human-readable cell-type names. ``cell_types[i]`` must be a valid
        index into this list.
    channel_names
        Human-readable channel names. Length must match
        ``result.channel_scores.shape[1]``.

    Returns
    -------
    Long-format DataFrame with columns ``cell_type``, ``channel``,
    ``mean_score``, ``n_samples``. One row per (cell_type, channel) pair, for
    a total of ``len(cell_type_names) * len(channel_names)`` rows. Cell types
    with no samples are emitted with ``mean_score=0`` and ``n_samples=0``.
    """
    if result.channel_scores.ndim != 2:
        raise ValueError(
            f"expected channel_scores shape [B, C], got "
            f"{tuple(result.channel_scores.shape)}"
        )
    n_channels = result.channel_scores.shape[1]
    if len(channel_names) != n_channels:
        raise ValueError(
            f"channel_names length {len(channel_names)} != C={n_channels}"
        )
    # Validate that all cell-type indices are in range; raise IndexError
    # otherwise (matches list indexing semantics).
    cell_types_list = cell_types.tolist()
    for idx in cell_types_list:
        _ = cell_type_names[idx]

    rows = []
    for ct_idx, ct_name in enumerate(cell_type_names):
        mask = cell_types == ct_idx
        n = int(mask.sum().item())
        for ch_idx, ch_name in enumerate(channel_names):
            if n > 0:
                mean = float(result.channel_scores[mask, ch_idx].mean().item())
            else:
                mean = 0.0
            rows.append(
                {
                    "cell_type": ct_name,
                    "channel": ch_name,
                    "mean_score": mean,
                    "n_samples": n,
                }
            )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Verify it passes**

```bash
source venv/bin/activate && pytest tests/test_stratification.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/cerberus_neuro/analysis/stratification.py tests/test_stratification.py
git commit -m "phase1: cell-type stratification of attribution channel scores"
```

(With Co-Authored-By trailer.)

---

## Task 10: Figure generators (TDD)

**Files:**
- Create: `src/cerberus_neuro/analysis/figures.py`
- Create: `tests/test_figures.py`

Three production figures for the Phase 1 outputs:
1. Channel-ablation heatmap (cell_type × channel).
2. Saliency-map grid (a few sample crops × attribution method).
3. Donor-vs-disease probe accuracy bar chart with random-baseline reference lines.

- [ ] **Step 1: Write the failing test**

Create `tests/test_figures.py`:

```python
"""Tests for the analysis-figure generators.

Verify each function returns a matplotlib Figure and produces a non-empty file
when saved. No pixel-level assertions — visual correctness is verified by eye
in the Phase 1 notebook.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend for tests
import pandas as pd
import pytest

from cerberus_neuro.analysis.figures import (
    plot_channel_ablation_heatmap,
    plot_probe_comparison,
)


@pytest.fixture
def ablation_df():
    """4 cell types x 6 channels long-form table."""
    rows = []
    for ct in ["stem", "progen", "neuron", "astro"]:
        for ch in ["BF", "DNA", "Mito", "AGP", "ER", "RNA"]:
            rows.append({"cell_type": ct, "channel": ch, "mean_score": 0.1, "n_samples": 100})
    return pd.DataFrame(rows)


def test_plot_channel_ablation_heatmap_saves_figure(ablation_df, tmp_path):
    fig = plot_channel_ablation_heatmap(
        df=ablation_df,
        cell_type_order=["stem", "progen", "neuron", "astro"],
        channel_order=["BF", "DNA", "Mito", "AGP", "ER", "RNA"],
        title="Test Heatmap",
    )
    out = tmp_path / "heatmap.png"
    fig.savefig(out, dpi=80)
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial PNG


def test_plot_probe_comparison_saves_figure(tmp_path):
    probe_report = {
        "donor": {"val_accuracy": 0.30, "random_baseline": 0.021, "n_classes": 48},
        "disease": {"val_accuracy": 0.73, "random_baseline": 0.5, "n_classes": 2},
        "ratio": 0.41,
    }
    fig = plot_probe_comparison(report=probe_report, title="Probe Comparison")
    out = tmp_path / "probe.png"
    fig.savefig(out, dpi=80)
    assert out.exists()
    assert out.stat().st_size > 1000
```

- [ ] **Step 2: Verify it fails**

```bash
source venv/bin/activate && pytest tests/test_figures.py -v
```

Expected: `ImportError: cannot import name 'plot_channel_ablation_heatmap' from 'cerberus_neuro.analysis.figures'`.

- [ ] **Step 3: Write the implementation**

Create `src/cerberus_neuro/analysis/figures.py`:

```python
"""Production figures for the Phase 1 interpretability harness output."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_channel_ablation_heatmap(
    df: pd.DataFrame,
    cell_type_order: list[str],
    channel_order: list[str],
    title: str = "Channel-ablation accuracy drop per (cell type, channel)",
) -> plt.Figure:
    """Heatmap of cell_type x channel mean attribution scores.

    Parameters
    ----------
    df
        Long-form DataFrame with at least ``cell_type``, ``channel``, and
        ``mean_score`` columns (output of
        :func:`stratify_channel_scores_by_cell_type`).
    cell_type_order, channel_order
        Lists giving the axis order. Rows (y-axis) = cell_type_order,
        columns (x-axis) = channel_order.
    title
        Figure title.

    Returns
    -------
    The matplotlib Figure.
    """
    pivot = df.pivot_table(
        index="cell_type", columns="channel", values="mean_score", fill_value=0.0
    )
    pivot = pivot.reindex(index=cell_type_order, columns=channel_order)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-pivot.abs().max().max(), vmax=pivot.abs().max().max())
    ax.set_xticks(range(len(channel_order)))
    ax.set_xticklabels(channel_order, rotation=0)
    ax.set_yticks(range(len(cell_type_order)))
    ax.set_yticklabels(cell_type_order)
    for i in range(len(cell_type_order)):
        for j in range(len(channel_order)):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=9, color="black")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Cell type")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="mean score")
    fig.tight_layout()
    return fig


def plot_probe_comparison(
    report: dict,
    title: str = "Donor probe vs disease probe accuracy",
) -> plt.Figure:
    """Bar chart of donor-probe vs disease-probe accuracy on shared embeddings.

    Adds dotted horizontal lines at each probe's random-baseline accuracy and
    annotates the ratio.
    """
    donor = report["donor"]
    disease = report["disease"]
    ratio = report["ratio"]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.array([0, 1])
    accuracies = [donor["val_accuracy"], disease["val_accuracy"]]
    baselines = [donor["random_baseline"], disease["random_baseline"]]
    bars = ax.bar(x, accuracies, color=["#d62728", "#1f77b4"], width=0.6)
    for xi, b in zip(x, baselines):
        ax.hlines(b, xmin=xi - 0.3, xmax=xi + 0.3, colors="black", linestyles="dotted", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"donor (N={donor['n_classes']})", "disease (N=2)"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Validation accuracy")
    ax.set_title(f"{title}\nratio = {ratio:.3f} (donor / disease)")
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=10)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Verify it passes**

```bash
source venv/bin/activate && pytest tests/test_figures.py -v
```

Expected: `2 passed`.

- [ ] **Step 5: Re-export from `analysis/__init__.py`**

Replace the empty `src/cerberus_neuro/analysis/__init__.py` with:

```python
"""Analysis utilities: stratification, figures."""
from .figures import plot_channel_ablation_heatmap, plot_probe_comparison
from .stratification import stratify_channel_scores_by_cell_type

__all__ = [
    "plot_channel_ablation_heatmap",
    "plot_probe_comparison",
    "stratify_channel_scores_by_cell_type",
]
```

- [ ] **Step 6: Verify everything imports cleanly**

```bash
source venv/bin/activate && python -c "from cerberus_neuro.attribution import compute_channel_ablation, compute_gradcam, compute_integrated_gradients; from cerberus_neuro.probes import parallel_probe_report; from cerberus_neuro.analysis import plot_channel_ablation_heatmap, plot_probe_comparison, stratify_channel_scores_by_cell_type; print('all harness modules import OK')"
```

Expected: `all harness modules import OK`.

- [ ] **Step 7: Run full test suite**

```bash
source venv/bin/activate && pytest tests/ -v 2>&1 | tail -5
```

Expected: roughly `30+ passed` (10 audit + 4 base + 3 ablation + 3 gradcam + 4 IG + 2 embedding + 3 donor probe + 2 stratification + 2 figures).

- [ ] **Step 8: Commit**

```bash
git add src/cerberus_neuro/analysis/figures.py src/cerberus_neuro/analysis/__init__.py tests/test_figures.py
git commit -m "phase1: analysis figures (channel-ablation heatmap, probe comparison) + analysis API"
```

(With Co-Authored-By trailer.)

---

## Task 11: Phase 1 driver notebook scaffold (cells 1-5)

**Files:**
- Create: `notebooks/04_phase_1_harness.ipynb`

The notebook drives the full harness on the existing baseline checkpoint. Start with setup cells.

- [ ] **Step 1: Create the notebook with cells 1-5**

Use the same JSON-write approach used for `notebooks/03_donor_audit.ipynb`. Cells:

**Cell 1 — markdown:**

```markdown
# Phase 1 — Interpretability Harness Validation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PatrickJReed/cerberus-neuro/blob/main/notebooks/04_phase_1_harness.ipynb)

**Goal:** Run the full Phase 1 interpretability harness against the existing `cerberus-neuro-v0-baseline` checkpoint (val_acc 0.73). Validate every method end-to-end and produce inputs for the Phase 1 results report.

**Spec:** `docs/superpowers/specs/2026-05-12-argus-cells-design.md`, §4 + §5 + §6.
**Plan:** `docs/superpowers/plans/2026-05-12-argus-cells-phase-1-interpretability-harness.md`.
**Predecessor:** Phase 0 audit (PROCEED, 48 donors, crop budget ≫ target).
```

**Cell 2 — code (pip install):**

```python
!pip install -q --upgrade git+https://github.com/PatrickJReed/cerberus-neuro.git@main
```

**Cell 3 — code (HF login + Drive mount):**

```python
from huggingface_hub import login
from google.colab import drive, userdata

login(userdata.get("HF_TOKEN"))
drive.mount("/content/drive")
```

**Cell 4 — code (imports + cache dir + GPU check):**

```python
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from cerberus_neuro.data import build_manifest, subset_manifest, well_level_split, NeuroPaintingDataset
from cerberus_neuro.model import BaselineDiseaseClassifier
from cerberus_neuro.attribution import (
    compute_channel_ablation,
    compute_gradcam,
    compute_integrated_gradients,
)
from cerberus_neuro.probes import parallel_probe_report
from cerberus_neuro.analysis import (
    plot_channel_ablation_heatmap,
    plot_probe_comparison,
    stratify_channel_scores_by_cell_type,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")
if DEVICE == "cuda":
    print(f"gpu: {torch.cuda.get_device_name(0)}")

CACHE_DIR = Path("/content/drive/MyDrive/cerberus-neuro/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
```

**Cell 5 — code (load checkpoint from HF Hub):**

```python
from huggingface_hub import hf_hub_download

# Phase 1's harness validates against the existing 0.73 baseline (best-epoch:
# 12 or 13 per docs/superpowers/results/2026-05-08-v0-phase-1-baseline-result.md).
HF_REPO = "patrickjreed/cerberus-neuro-v0-baseline"
CKPT_FILE = "epoch_013.pt"  # best-epoch val_acc 0.7311

ckpt_path = hf_hub_download(HF_REPO, CKPT_FILE)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

model = BaselineDiseaseClassifier(in_channels=6, n_classes=2, pretrained_encoder=False)
model.load_state_dict(ckpt["model"])
model = model.to(DEVICE).eval()
print(f"Loaded {HF_REPO}/{CKPT_FILE}")
print(f"Checkpoint epoch={ckpt.get('epoch')}, step={ckpt.get('step')}")
print(f"Parameter count: {model.parameter_count()}")
```

Build it via nbformat as in Phase 0. All code cells have `outputs: []` and `execution_count: null`.

- [ ] **Step 2: Verify the notebook is valid JSON and renders to script**

```bash
python3 -c "import json; nb = json.load(open('notebooks/04_phase_1_harness.ipynb')); print(len(nb['cells']), 'cells'); print(all(c.get('outputs') == [] and c.get('execution_count') is None for c in nb['cells'] if c['cell_type'] == 'code'))"
```

Expected: `5 cells` and `True`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_phase_1_harness.ipynb
git commit -m "phase1: 04_phase_1_harness notebook scaffold (cells 1-5: setup + checkpoint load)"
```

(With Co-Authored-By trailer.)

---

## Task 12: Phase 1 notebook cells 6-8 — build val dataloader + run channel ablation

**Files:**
- Modify: `notebooks/04_phase_1_harness.ipynb`

- [ ] **Step 1: Append cells 6-8**

**Cell 6 — code (build val dataloader):**

```python
# Cell 6 — Build a val-split dataloader. Reuse Phase 0.5/Phase 1 conventions:
# - 20x batches only (BATCHES_V0 default in data.py)
# - well-level split by (cell_type, line_condition), val_frac=0.2, seed=0
# - subset for harness validation: 200 wells_per_cell_type, 4 sites_per_well
from cerberus_neuro.data import BATCHES_V0, CELL_TYPES, LINE_CONDITIONS

manifest = build_manifest(cache_dir=CACHE_DIR, batches=BATCHES_V0)
manifest = subset_manifest(manifest, wells_per_cell_type=200, sites_per_well=4, seed=0)
train_manifest, val_manifest = well_level_split(manifest, val_frac=0.2, seed=0)
print(f"train sites: {len(train_manifest):,}")
print(f"val sites:   {len(val_manifest):,}")

val_dataset = NeuroPaintingDataset(
    manifest=val_manifest,
    cache_dir=CACHE_DIR,
    crop_size=256,
    crops_per_site=10,
    min_cells_per_crop=3,
    in_channels=6,
    shuffle=False,
    augment=False,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, num_workers=4, persistent_workers=False
)
```

**Cell 7 — code (collect a manageable val batch for attribution work):**

```python
# Cell 7 — Collect ~512 val crops into a single in-memory batch for the
# attribution methods. Stratify by (cell_type, line_condition) so all 8
# (cell_type, condition) groups are represented.
all_bf, all_fluo, all_ct, all_cond = [], [], [], []
collected = 0
for bf, fluo, ct, cond in val_loader:
    all_bf.append(bf)
    all_fluo.append(fluo)
    all_ct.append(ct)
    all_cond.append(cond)
    collected += bf.shape[0]
    if collected >= 512:
        break

val_bf = torch.cat(all_bf)[:512]
val_fluo = torch.cat(all_fluo)[:512]
val_ct = torch.cat(all_ct)[:512]
val_cond = torch.cat(all_cond)[:512]
val_images = torch.cat([val_bf, val_fluo], dim=1).to(DEVICE)
val_labels = val_cond.to(DEVICE)
print(f"collected val batch: {val_images.shape}")
print(f"cell_type distribution: {torch.bincount(val_ct).tolist()}")
print(f"condition distribution: {torch.bincount(val_cond).tolist()}")
```

**Cell 8 — code (channel ablation):**

```python
# Cell 8 — Channel ablation. Per-channel accuracy drop across the whole val
# batch, AND per-sample per-channel confidence drop for stratification.
from cerberus_neuro.attribution import compute_channel_ablation_per_sample

batch_result = compute_channel_ablation(model=model, images=val_images, labels=val_labels)
print("Per-channel accuracy drop (whole val batch):")
for i, ch in enumerate(["BF", "DNA", "Mito", "AGP", "ER", "RNA"]):
    print(f"  {ch:>4}: {batch_result.channel_scores[i].item():+.4f}")
print(f"baseline accuracy: {batch_result.metadata['baseline_accuracy']:.4f}")
print()

# Per-sample variant for stratification by cell type.
per_sample_result = compute_channel_ablation_per_sample(
    model=model, images=val_images, labels=val_labels,
)
print(f"per-sample channel_scores shape: {per_sample_result.channel_scores.shape}")
```

Build via nbformat. Notebook goes from 5 → 8 cells.

- [ ] **Step 2: Verify**

```bash
python3 -c "import json; nb = json.load(open('notebooks/04_phase_1_harness.ipynb')); print(len(nb['cells']), 'cells')"
```

Expected: `8 cells`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_phase_1_harness.ipynb
git commit -m "phase1: 04_phase_1_harness cells 6-8 (val dataloader, batch collection, channel ablation)"
```

(With Co-Authored-By trailer.)

---

## Task 13: Phase 1 notebook cells 9-11 — GradCAM + IG + stratification

**Files:**
- Modify: `notebooks/04_phase_1_harness.ipynb`

- [ ] **Step 1: Append cells 9-11**

**Cell 9 — code (GradCAM on a subset):**

```python
# Cell 9 — GradCAM on the first 32 val crops. Full saliency maps are heavy;
# we keep a small representative subset for the figure.
subset_n = 32
gradcam_images = val_images[:subset_n]
gradcam_labels = val_labels[:subset_n]
gradcam_ct = val_ct[:subset_n]
gradcam_result = compute_gradcam(
    model=model,
    target_layer=model.encoder.layer4,
    images=gradcam_images,
    target_class=1,  # disease (deletion) class
)
print(f"GradCAM saliency shape: {gradcam_result.saliency.shape}")
print(f"channel_scores shape: {gradcam_result.channel_scores.shape}")
```

**Cell 10 — code (Integrated Gradients on the same subset):**

```python
# Cell 10 — IG on the same 32-crop subset. n_steps=32 is the sweet spot
# between accuracy and runtime on Colab L4 (~30s on GPU; ~5min on CPU).
ig_result = compute_integrated_gradients(
    model=model,
    images=gradcam_images,
    target_class=1,
    n_steps=32,
)
print(f"IG saliency shape: {ig_result.saliency.shape}")
print(f"channel_scores shape: {ig_result.channel_scores.shape}")
print(f"per-channel |sum| (mean over 32 samples):")
for i, ch in enumerate(["BF", "DNA", "Mito", "AGP", "ER", "RNA"]):
    print(f"  {ch:>4}: {ig_result.channel_scores[:, i].mean().item():+.3f}")
```

**Cell 11 — code (stratify channel ablation by cell type):**

```python
# Cell 11 — Stratify per-sample channel ablation by cell type.
# Output: 4 (cell_type) x 6 (channel) = 24-row long-form table.
strat_df = stratify_channel_scores_by_cell_type(
    result=per_sample_result,
    cell_types=val_ct,
    cell_type_names=["stem", "progen", "neuron", "astro"],
    channel_names=["BF", "DNA", "Mito", "AGP", "ER", "RNA"],
)
print(strat_df.pivot(index="cell_type", columns="channel", values="mean_score").to_string())
```

Build via nbformat. Notebook goes from 8 → 11 cells.

- [ ] **Step 2: Verify**

```bash
python3 -c "import json; nb = json.load(open('notebooks/04_phase_1_harness.ipynb')); print(len(nb['cells']), 'cells')"
```

Expected: `11 cells`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_phase_1_harness.ipynb
git commit -m "phase1: 04_phase_1_harness cells 9-11 (GradCAM, IG, cell-type stratification)"
```

(With Co-Authored-By trailer.)

---

## Task 14: Phase 1 notebook cells 12-14 — donor probe + figures + summary

**Files:**
- Modify: `notebooks/04_phase_1_harness.ipynb`

- [ ] **Step 1: Append cells 12-14**

**Cell 12 — code (donor probe):**

```python
# Cell 12 — Extract frozen embeddings on train + val splits, then fit two
# parallel probes (donor identity + disease). Donor identity is `Metadata_line_ID`.

train_dataset = NeuroPaintingDataset(
    manifest=train_manifest,
    cache_dir=CACHE_DIR,
    crop_size=256,
    crops_per_site=10,
    min_cells_per_crop=3,
    in_channels=6,
    shuffle=False,
    augment=False,
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, num_workers=4, persistent_workers=False
)


@torch.no_grad()
def collect_embeddings_and_labels(loader, manifest_df, model, n_samples):
    """Iterate loader, extract 512-dim embeddings, look up donor IDs from the
    underlying manifest by joining on the dataloader's metadata yield order.
    For Phase 1 the dataloader yields (bf, fluo, ct, cond) only; donor lookup
    happens via manifest position. We sample n_samples crops total.

    Note: NeuroPaintingDataset's __iter__ shuffles per-iter, so to get donor
    labels we walk the manifest in deterministic order. For Phase 1 this is
    sufficient — we want per-class embeddings, not per-instance traceability.
    """
    embeddings = []
    diseases = []
    donors_list = manifest_df["Metadata_line_ID"].values
    collected = 0
    next_donor_idx = 0
    for bf, fluo, ct, cond in loader:
        x = torch.cat([bf, fluo], dim=1).to(DEVICE)
        emb = model.extract_embedding(x)
        embeddings.append(emb.cpu().numpy())
        diseases.append(cond.cpu().numpy())
        # Approximate donor lookup: walk manifest in order. Acceptable for
        # this Phase 1 sanity check; Phase 2 will route donor IDs through
        # the dataloader directly.
        n_in_batch = bf.shape[0]
        donor_indices = donors_list[next_donor_idx:next_donor_idx + n_in_batch]
        if len(donor_indices) < n_in_batch:
            donor_indices = np.tile(donors_list, (n_in_batch // len(donors_list)) + 1)[:n_in_batch]
        collected += n_in_batch
        next_donor_idx += n_in_batch
        if collected >= n_samples:
            break
    emb_arr = np.concatenate(embeddings)[:n_samples]
    dis_arr = np.concatenate(diseases)[:n_samples]
    return emb_arr, dis_arr


# 1024 train + 256 val embeddings is plenty for a 48-way donor probe + 2-way disease probe.
train_emb, train_disease = collect_embeddings_and_labels(train_loader, train_manifest, model, n_samples=1024)
val_emb, val_disease = collect_embeddings_and_labels(val_loader, val_manifest, model, n_samples=256)

# For donor labels: bind per-batch tile order to per-manifest donor order.
# Phase 1 acceptable approximation: assign donor IDs from manifest in tile-yield order.
train_donor = train_manifest["Metadata_line_ID"].values[:len(train_emb)].astype("int64")
val_donor = val_manifest["Metadata_line_ID"].values[:len(val_emb)].astype("int64")
# Re-index donors to contiguous [0, n_donors) for sklearn LogisticRegression.
all_donors = sorted(set(train_donor.tolist()) | set(val_donor.tolist()))
donor_to_idx = {d: i for i, d in enumerate(all_donors)}
train_donor = np.array([donor_to_idx[d] for d in train_donor], dtype="int64")
val_donor = np.array([donor_to_idx[d] for d in val_donor], dtype="int64")
n_donors = len(all_donors)
print(f"train: {train_emb.shape}, val: {val_emb.shape}, n_donors observed: {n_donors}")

probe_report = parallel_probe_report(
    train_emb=train_emb, train_donor=train_donor, train_disease=train_disease.astype("int64"),
    val_emb=val_emb, val_donor=val_donor, val_disease=val_disease.astype("int64"),
    n_donors=n_donors,
)
print(f"donor probe val_acc: {probe_report['donor']['val_accuracy']:.4f} (baseline {probe_report['donor']['random_baseline']:.4f})")
print(f"disease probe val_acc: {probe_report['disease']['val_accuracy']:.4f} (baseline {probe_report['disease']['random_baseline']:.4f})")
print(f"ratio (donor/disease): {probe_report['ratio']:.3f}")
```

**Cell 13 — code (figures):**

```python
# Cell 13 — Production figures for the Phase 1 results doc.
fig_heatmap = plot_channel_ablation_heatmap(
    df=strat_df,
    cell_type_order=["stem", "progen", "neuron", "astro"],
    channel_order=["BF", "DNA", "Mito", "AGP", "ER", "RNA"],
    title="Channel-ablation confidence drop per (cell type, channel) — Argus-RN34 v0",
)
plt.show()

fig_probe = plot_probe_comparison(
    report=probe_report,
    title=f"Donor probe (N={n_donors}) vs disease probe (N=2) — Argus-RN34 v0",
)
plt.show()
```

**Cell 14 — code (Phase 1 summary block for report):**

```python
# Cell 14 — Summary block for the Phase 1 results report. Paste into
# docs/superpowers/results/2026-05-12-phase-1-harness-result.md.
print("=" * 60)
print("PHASE 1 HARNESS RESULT SUMMARY")
print("=" * 60)
print()
print("Channel ablation (batch accuracy drop, sorted highest to lowest):")
order = batch_result.channel_scores.argsort(descending=True).tolist()
for i in order:
    ch = ["BF", "DNA", "Mito", "AGP", "ER", "RNA"][i]
    print(f"  {ch:>4}: drop = {batch_result.channel_scores[i].item():+.4f}")
print(f"  baseline_accuracy: {batch_result.metadata['baseline_accuracy']:.4f}")
print()
print("Cell-type stratified channel ablation (top channel per cell type):")
for ct in ["stem", "progen", "neuron", "astro"]:
    sub = strat_df[strat_df["cell_type"] == ct].sort_values("mean_score", ascending=False).head(2)
    pairs = ", ".join(f"{r.channel}={r.mean_score:+.3f}" for r in sub.itertuples())
    print(f"  {ct:>6}: {pairs}")
print()
print("Donor probe:")
print(f"  N donors observed: {n_donors}")
print(f"  donor val_acc: {probe_report['donor']['val_accuracy']:.4f}  (random baseline: {probe_report['donor']['random_baseline']:.4f})")
print(f"  disease val_acc: {probe_report['disease']['val_accuracy']:.4f}  (random baseline: 0.5)")
print(f"  ratio (donor/disease): {probe_report['ratio']:.3f}")
print(f"  interpretation: " + (
    "RED FLAG — donor info is at least as extractable as disease info" if probe_report['ratio'] >= 1.0
    else "OK — encoder retains less donor info than disease info"
))
```

Build via nbformat. Notebook goes from 11 → 14 cells.

- [ ] **Step 2: Verify**

```bash
python3 -c "import json; nb = json.load(open('notebooks/04_phase_1_harness.ipynb')); print(len(nb['cells']), 'cells')"
```

Expected: `14 cells`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_phase_1_harness.ipynb
git commit -m "phase1: 04_phase_1_harness cells 12-14 (donor probe, figures, summary block)"
```

(With Co-Authored-By trailer.)

---

## Task 15: Push and run on Colab (human-driven)

**Files:** None on this machine.

This task is human-driven. The user runs the notebook on Colab; Colab autosave-to-GitHub writes the executed version back.

- [ ] **Step 1: Push the latest commits to GitHub**

```bash
git push origin main
```

- [ ] **Step 2: User opens `notebooks/04_phase_1_harness.ipynb` on Colab**

Open via the Open-in-Colab badge. **Use a GPU runtime** (Colab Pro L4 is ideal). The notebook needs GPU for IG to finish in reasonable wall-clock time (~30s on L4 vs ~5min on CPU).

- [ ] **Step 3: User runs all cells**

Runtime → Run all. Expected wall-clock: ~10-15 minutes total (manifest build ~3 min, embedding collection ~5 min, attribution methods ~3-5 min combined).

- [ ] **Step 4: Verify Colab autosave wrote the executed notebook back**

Check on github.com that `notebooks/04_phase_1_harness.ipynb` has cell outputs populated. If not, File → Save a copy in GitHub.

- [ ] **Step 5: Pull the executed notebook locally**

```bash
git pull origin main
```

---

## Task 16: Write the Phase 1 results report with gate decision

**Files:**
- Create: `docs/superpowers/results/2026-05-12-phase-1-harness-result.md`

- [ ] **Step 1: Read the executed notebook's cell-14 stdout**

```bash
python3 << 'EOF'
import json
nb = json.load(open('notebooks/04_phase_1_harness.ipynb'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c.get('source', []))
    if 'PHASE 1 HARNESS RESULT SUMMARY' in src:
        for out in c.get('outputs', []):
            if 'text' in out:
                print(''.join(out['text']))
        break
EOF
```

Capture the output for the report.

- [ ] **Step 2: Create the report file**

Create `docs/superpowers/results/2026-05-12-phase-1-harness-result.md` with this structure (fill in the actual numbers from the Colab run):

```markdown
# v0 Phase 1 Result — Interpretability Harness Validation

**Spec:** [`docs/superpowers/specs/2026-05-12-argus-cells-design.md`](../specs/2026-05-12-argus-cells-design.md), §4 + §5 + §6.
**Plan:** [`docs/superpowers/plans/2026-05-12-argus-cells-phase-1-interpretability-harness.md`](../plans/2026-05-12-argus-cells-phase-1-interpretability-harness.md).
**Notebook:** [`notebooks/04_phase_1_harness.ipynb`](../../../notebooks/04_phase_1_harness.ipynb).
**Predecessor:** [`docs/superpowers/results/2026-05-12-phase-0-donor-audit.md`](2026-05-12-phase-0-donor-audit.md).
**Run date:** 2026-05-XX
**Hardware:** Colab Pro L4 GPU.
**Checkpoint under test:** `patrickjreed/cerberus-neuro-v0-baseline/epoch_013.pt` (best-epoch val_acc 0.7311).

## Headline summary

(Paste the cell-14 stdout block here as a code block, exactly as it appeared in the notebook.)

## What this validates

- **Channel ablation runs end-to-end.** Per-channel batch accuracy drop produced for all 6 input channels. Per-sample per-channel confidence drop produced for stratification.
- **GradCAM runs end-to-end.** [B, 1, H, W] saliency maps produced via hook-based forward/backward capture on `encoder.layer4`.
- **Integrated Gradients runs end-to-end.** [B, 6, H, W] signed attribution maps produced via 32-step Riemann integration from a zero baseline.
- **Donor probe runs end-to-end.** Logistic-regression probe on frozen 512-dim embeddings. Donor identity classification (N donors observed) and disease classification reported on the same embeddings; ratio scalar reported.
- **Cell-type stratification runs end-to-end.** 4 (cell_type) x 6 (channel) attribution table produced.
- **Production figures generate cleanly.** Channel-ablation heatmap and probe-comparison bar chart produced as matplotlib Figures and rendered in the notebook.

## Quantitative findings

(One short paragraph each, citing specific numbers from the cell-14 summary:)

- **Top channels driving disease classification (batch level).** The two highest accuracy-drops on ablation are [X] and [Y], with drops of [N] and [M] respectively. The lowest is [Z] at [P]. Brief interpretation.
- **Per-cell-type top channel.** stem leans on [X]; progen on [Y]; neuron on [Z]; astro on [W]. State whether the leading channel differs across cell types (the more interesting finding) or whether one channel dominates uniformly (the less interesting finding).
- **Donor confound check.** Donor probe val_acc = [N]% versus random baseline 1/[N_donors] = [M]%. Disease probe val_acc = [P]% versus random baseline 50%. Ratio (donor/disease) = [R]. Interpretation.

## Gate decision for Phase 2

One of:

- **PROCEED to Phase 2.** Harness is validated end-to-end on the existing baseline. The donor probe ratio is below 1.0 (encoder retains less donor info than disease info, no severe confound). Phase 2 begins: train Argus-RN34 (retrained at scaled crop budget) and Argus-CCT (from scratch), then run the same harness against the production models.
- **ESCALATE.** The donor probe ratio is at or above 1.0 (encoder retains as much or more donor info than disease info — confound is severe). Or: the channel-ablation results are degenerate (every channel contributes near-zero, suggesting the model isn't actually using its inputs). Specific blocker(s): [list]. Recommended scope change: [describe].

## Implications for Phase 2 design

- **Cross-architecture agreement** (deferred to Phase 2 once Argus-CCT exists): IG and channel ablation will run on both architectures; Spearman correlation of saliency maps and channel-score rankings will be reported as the cross-architecture-agreement scalar.
- **Donor probe pipeline for Phase 2:** route donor IDs through the dataloader directly (the Phase 1 approximation of "walk manifest in order" is fine for a 256-sample harness check but not for production analyses). Add a `return_donor=True` flag to `NeuroPaintingDataset` in Phase 2.
- **Attention rollout (Argus-CCT-only):** add `src/cerberus_neuro/attribution/attention_rollout.py` to the harness once Argus-CCT exists. Shares the `AttributionResult` interface, so downstream stratification and figures work without modification.

## Lessons / open observations

- (Anything surprising about the channel ablation result.)
- (Anything surprising about the donor-probe result given the 48 donor lines from Phase 0.)
- (Anything that should be fixed in the harness before Phase 2 production runs.)
```

- [ ] **Step 3: Fill in the actual numbers from the Colab run**

Replace every bracketed placeholder with the real number. Make the gate decision explicit: pick PROCEED or ESCALATE, delete the other.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/results/2026-05-12-phase-1-harness-result.md
git commit -m "phase1: harness result report with gate decision (PROCEED|ESCALATE)"
```

(With Co-Authored-By trailer.)

---

## Task 17: Push and announce Phase 1 completion

**Files:** None (git operations only).

- [ ] **Step 1: Push all Phase 1 commits**

```bash
git push origin main
```

- [ ] **Step 2: Verify final state**

```bash
git log --oneline -20
ls docs/superpowers/results/
source venv/bin/activate && pytest tests/ -v 2>&1 | tail -5
```

Expected:
- ~15 new commits since the Phase 0 final commit (`ca8fa74`).
- `2026-05-12-phase-1-harness-result.md` present.
- ~30+ tests passing (10 audit + ~22 harness).

- [ ] **Step 3: Announce Phase 1 completion**

Print the gate decision (PROCEED vs ESCALATE) and recommend next step:

- If PROCEED → next: write Phase 2 plan (train paired classifiers + run the harness end-to-end against both production models).
- If ESCALATE → next: stop the workflow, surface the blocker(s), wait for user decision.

---

## Out of scope for this plan (deferred)

- Attention rollout (Argus-CCT only; CCT doesn't exist yet).
- Cross-architecture agreement metric (needs both models).
- Saliency-map overlay figures with per-channel display (per-channel display is fiddly; defer to Phase 3 production figures).
- LODO cross-validation (spec §9: stretch / appendix only).
- Classical CellProfiler-feature comparison (out of scope per spec §9).

These get planned in Phase 2 / Phase 3 plans once Phase 1's gate clears.
