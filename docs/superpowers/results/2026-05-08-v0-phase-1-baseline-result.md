# v0 Phase 1 Result — Baseline Disease Classifier

**Spec:** [`docs/superpowers/specs/2026-05-08-v0-baseline-first-paired-experiment-design.md`](../specs/2026-05-08-v0-baseline-first-paired-experiment-design.md)
**Plan:** [`docs/superpowers/plans/2026-05-08-v0-phase-1-baseline-disease.md`](../plans/2026-05-08-v0-phase-1-baseline-disease.md)
**Run date:** 2026-05-08
**Wall-clock:** ~2 hours (15 epochs full run)
**Hardware:** Colab Pro L4 GPU, batch_size=64, num_workers=8
**HF artifact:** `patrickjreed/cerberus-neuro-v0-baseline` (epoch_000.pt through epoch_014.pt pushed)

## Headline numbers

- **Best-epoch val_acc_line_condition: 0.7311** (epochs 12 and 13, identical to 4 decimal places)
- **Final-epoch val_acc_line_condition: 0.7203** (epoch 14)
- **Best val_loss: 0.5207** (epoch 14, well below `ln(2)=0.693` random baseline)
- **Stable convergence** across epochs 8–14 (val_acc range 0.643–0.731)

## Per-epoch trajectory

| epoch | val_acc | val_loss | comment |
|---|---|---|---|
| 0 | 0.680 | 0.599 | post-warmup signal: model finds a useful head from pretrained features |
| 1 | 0.577 | 0.683 | LR-transition turbulence (peak LR just reached at end of warmup ≈ end of ep 0) |
| 2 | 0.496 | 0.857 | LR-transition continues; head temporarily over-confident in wrong direction |
| 3 | 0.677 | 0.591 | recovery — model re-finds the signal at peak LR |
| 4 | 0.670 | 0.649 | stable around 0.67 |
| 5 | 0.710 | 0.531 | first crossing of 0.70 |
| 6 | 0.654 | 0.618 | minor dip |
| 7 | 0.643 | 0.648 | minor dip |
| 8 | 0.712 | 0.525 | back to 0.71 |
| 9 | 0.715 | 0.572 | |
| 10 | 0.721 | 0.539 | |
| 11 | 0.713 | 0.537 | |
| 12 | **0.731** | 0.528 | best |
| 13 | **0.731** | 0.523 | best (matches ep 12) |
| 14 | 0.720 | **0.521** | best val_loss; cosine LR ≈ 0 |

## Decision-gate applied

Per the spec's three-regime table:

| Threshold | Action |
|---|---|
| ≥ 0.65 | proceed to Phase 2 |
| 0.55–0.65 | marginal; decide between scope-up or accept narrower gap |
| ≤ 0.55 | scope-up mandatory |

**Best-epoch val_acc = 0.7311 → Phase 2 is unblocked, with an upper bound that supports the full A3 paired-experiment narrative.**

## What this tells us

**Disease signal is robust at the 16k-crop scale.** The model converged to 0.73 val_acc with stable training across 15 epochs and a clean cosine schedule. The val_loss trajectory (0.60 → 0.52, never plateauing prematurely) confirms the model continued learning meaningful features through the end of the run.

**Training stability is good once past the warmup transition.** Epochs 1–2 showed an LR-transition transient (val_acc dipped to 0.50, val_loss to 0.86) that was indistinguishable from genuine catastrophic forgetting on those two data points alone. Epochs 3–14 showed clean convergence with the kind of mild val_acc fluctuation typical of healthy training (0.64 to 0.73 range, slowly drifting up).

**The cell-type-only training recipe transfers to disease classification at this scale**, just over more epochs. Cell-type converged in 1 epoch to 0.96 because the signal is gross (visually distinct cell morphologies). Disease needs the full 15 epochs because the signal is subtle (within-cell-type morphological differences from 22q11.2 deletion).

## Implications for Phase 2 design

We now have a meaningful upper bound for Cerberus brightfield-only disease accuracy. The project-level success criterion (A3) requires `baseline > cerberus > 0.50, gap ≥ 5pp`:

| Cerberus result | Interpretation |
|---|---|
| 0.55–0.65 | meaningful brightfield recovery; ~10–20pp gap to baseline; defensible v0 paired-experiment claim |
| 0.50–0.55 | brightfield carries minimal disease signal at 20×; gap is large but Cerberus is at chance — honest negative result |
| > 0.65 | unexpected; would suggest brightfield carries most of the signal that the fluorescence channels add |

**Phase 2 design considerations** (now informed by Phase 1):

1. **Same training recipe should work.** 15 epochs at LR `3e-4` (head) / `3e-5` (encoder) with the existing AMP / grad-clip / cosine setup converged the baseline cleanly. Use the same recipe for the multi-task Cerberus model.
2. **Best-epoch reporting convention.** Phase 1 confirms best-epoch is the right number to report — it captures real convergence rather than random end-of-run fluctuation. Phase 2 will use the same convention.
3. **Allow the LR transition.** Phase 2 should also run at least 5 epochs before any "is it working?" assessment, given the Phase 1 transient lasted 2 epochs.
4. **Multi-task gradient interference is the remaining unknown.** Phase 1 showed the encoder + classification recipe works for disease in isolation; whether the segmentation head's gradient disrupts this in the multi-task setup is the open Phase 2 question.

## Lessons documented for the project

**The user pushed back on a premature trend call at epoch 2.** The original draft of this document concluded "stop training, signal is real but unstable" based on three monotonic-down data points (0.68 → 0.58 → 0.50). That call was wrong. After resuming, the model recovered cleanly at epoch 3 and exceeded its epoch-0 number by epoch 5, ultimately reaching 0.73.

**The principled rule: 3 data points is not enough to call a trend in noisy training metrics**, especially when one of those points is at an LR-schedule transition (end of warmup → start of peak LR). The baseline run's epochs 1–2 were textbook LR-transition turbulence, not catastrophic forgetting.

This is a worthwhile failure mode to internalize: monotonic-down across 3 data points feels conclusive but isn't, especially when:
- The schedule has a known regime change in that window (warmup → cosine).
- Per-batch training loss is bouncy across the same window.
- A directly comparable prior run (the multi-task Cerberus runs) showed the same transient followed by recovery.

The right move when uncertain at the warmup-transition boundary is **let it run**, not stop early.

## Decision and next action

**Phase 2 unblocked with strong upper bound.** Next step: brainstorm + write Phase 2 design spec for the 2-head Cerberus + paired-experiment evaluation.

The Phase 2 success criterion is already implicitly defined by the project-level criterion (A3) and Phase 1's 0.73 baseline number: Cerberus brightfield-only val_acc in [0.55, 0.65] would constitute a defensible "brightfield recovers ~70-90% of the all-channel signal" claim with a meaningful gap.

Phase 0.5 (cell-type single-task deployment to HF Hub) can run in parallel with Phase 2 design work.
