# v0 Phase 1 Result — Baseline Disease Classifier

**Spec:** [`docs/superpowers/specs/2026-05-08-v0-baseline-first-paired-experiment-design.md`](../specs/2026-05-08-v0-baseline-first-paired-experiment-design.md)
**Plan:** [`docs/superpowers/plans/2026-05-08-v0-phase-1-baseline-disease.md`](../plans/2026-05-08-v0-phase-1-baseline-disease.md)
**Run date:** 2026-05-08
**Wall-clock:** ~30 min (run terminated early at epoch 2 of 15; trajectory was monotonically degrading)
**Hardware:** Colab Pro L4 GPU, batch_size=64, num_workers=8
**HF artifact:** `patrickjreed/cerberus-neuro-v0-baseline` (epoch_000.pt and epoch_001.pt pushed)

## Headline number

- **Best-epoch val_acc_line_condition: 0.6796 (epoch 0)**
- Final-observed-epoch val_acc_line_condition: 0.4964 (epoch 2)
- Run was terminated early; epochs 3–14 not executed.

## Per-epoch trajectory

| epoch | acc_line_condition | L_line_condition | reading |
|---|---|---|---|
| 0 | **0.6796** | 0.599 | real signal: above chance (95% CI for n=3068 excludes 0.55), below `ln(2)=0.693` floor |
| 1 | 0.5766 | 0.683 | back at random-baseline loss; head losing the signal |
| 2 | 0.4964 | 0.857 | below chance accuracy with above-random loss → confidently wrong on val |

## Decision-gate applied

Per the spec's three-regime table:

| Threshold | Action |
|---|---|
| ≥ 0.65 | proceed to Phase 2 |
| 0.55–0.65 | marginal; decide between scope-up or accept narrower gap |
| ≤ 0.55 | scope-up mandatory |

**Best-epoch val_acc = 0.6796 → Phase 2 is unblocked.**

The criterion was deliberately keyed to best-epoch (not final-epoch) val_acc to handle the case where training is unstable but the data carries signal. This run is exactly that case.

## What this tells us

**Disease signal exists in the 16k-crop subset.** The encoder found enough signal in epoch 0 to push val_acc from chance (0.50) to 0.68 in fewer than 200 gradient steps. That's not noise — it's the model genuinely learning something disease-relevant from the 6-channel Cell Painting input.

**Training is unstable at the cell-type-only recipe applied to this task.** The same recipe (encoder LR `3e-5`, head LR `3e-4`, AMP, grad clip 1.0, 5% warmup, cosine schedule, 15-epoch budget) that converged the cell-type classifier to 0.96 val acc in 1 epoch *destroys* the disease classifier's val performance over 2 epochs. The disease signal is weaker than the cell-type signal and the model overfits to spurious training-set patterns once the head has fit the easy signal.

**Implications for Phase 2 design:**

1. The 2-head Cerberus model at v0 scope can plausibly produce a meaningful disease number — but only if Phase 2 uses early stopping or a much lower LR / fewer epochs to avoid the same overfitting trajectory.
2. The disease task at this data scale is fundamentally harder than the cell-type task, so multi-task balancing (Kendall weighting) needs to be tuned to not treat them as comparable difficulty.
3. The "meaningful gap ≥ 5pp" criterion in the project-level spec is now contextualized: we have a baseline upper bound around 0.68 and the gap measurement needs to capture it via best-epoch comparison, not final-epoch.

## Open questions and Phase 2 considerations

- **Best vs final epoch reporting.** For the eventual writeup, decide whether to report best-epoch or final-epoch numbers. Best-epoch is more optimistic but reflects what an HTS deployment would actually use (you'd ship the best checkpoint, not the last one). Final-epoch is conservative and acknowledges training instability.
- **Why the regression?** Three hypotheses, in order of likelihood:
  1. Genuine signal-to-noise issue at our 16k-crop scale: real disease signal is small, head overfits to spurious training-set features once it's done with the easy signal. **Implies: Phase 2 needs early stopping or aggressive regularization.**
  2. Train/val split confound: well_level_split keeps lines in both train and val (each line has 8 wells per plate, split at the well level not the line level), so the model could learn line-specific morphology that pseudo-generalizes within line. As training progresses, the head over-relies on these pseudo-features and degrades on the val wells of held-out lines. **Implies: Phase 2 should use a line-level holdout split for the rigorous experiment, with well-level as the optimistic upper bound.**
  3. The pretrained ImageNet features happen to be coincidentally informative for the disease question at init, and aggressive fine-tuning destroys those features. **Implies: Phase 2 should use a much smaller encoder LR ratio (e.g., `0.01` instead of `0.1`) for the disease task.**

  Without an ablation, we can't distinguish these three. For Phase 2 v0 we accept the optimistic best-epoch interpretation; v1 ablations would test the three hypotheses separately.

## Decision and next action

**Phase 2 unblocked.** Next step: brainstorm + write Phase 2 design spec (2-head Cerberus + paired-experiment evaluation) with explicit handling of the training-instability finding from this run.

Concretely, Phase 2's design should specify:
- Best-epoch checkpoint selection (early stopping based on val acc)
- Whether to lower encoder LR ratio further (e.g., `0.01`)
- Whether to add line-level holdout for the rigorous evaluation
- How to report the disease-accuracy gap (best-epoch baseline vs best-epoch Cerberus)

Phase 0.5 (cell-type single-task deployment to HF Hub) can run in parallel with Phase 2.
