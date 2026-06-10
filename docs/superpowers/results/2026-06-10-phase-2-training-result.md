# argus-cells Phase 2 Result — Paired Production Training

**Spec:** [`docs/superpowers/specs/2026-05-12-argus-cells-design.md`](../specs/2026-05-12-argus-cells-design.md) §2, §5, §6.
**Plan:** [`docs/superpowers/plans/2026-06-09-argus-cells-phase-2.md`](../plans/2026-06-09-argus-cells-phase-2.md) (Group C).
**Notebook:** [`notebooks/05_phase_2_train.ipynb`](../../../notebooks/05_phase_2_train.ipynb).
**Run date:** 2026-06-10.
**Hardware:** Argus-RN34 on Colab Pro (L4); Argus-CCT from scratch.
**HF artifacts:** `patrickjreed/argus-rn34-v0`, `patrickjreed/argus-cct-v0` (per-epoch checkpoints pushed).
**Data scope:** ~1728 sites (48 wells/cell type × 9 sites), `crops_per_site=30`, ~40-50k crops, well-level split (val_frac=0.2, seed=0).

## Headline

Both production models cleared the Phase 2 success gate (spec §6: at least one of the two ≥ 0.73 best-epoch val_acc). Both passed, not just one:

| Model | Best-epoch val_acc | Best epoch | Best val_loss | vs 0.73 gate | vs predecessor 0.7311 |
|---|---|---|---|---|---|
| **Argus-RN34** (ResNet34, pretrained) | **0.8181** | 8 | 0.4098 | +0.088 | +0.087 |
| **Argus-CCT** (CCT, from scratch) | **0.7751** | 22 | 0.5000 | +0.045 | +0.044 |

## Gate decision

**PROCEED to Phase 3.** Both models learned the disease task above the gate, so the paired-model interpretability comparison has two credible models to interpret. No scope-review pause.

## Per-model results

**Argus-RN34** reached **0.8181**, beating the predecessor 16k-crop baseline (0.7311) by ~9 points. The gain is consistent with the scaled crop budget (`crops_per_site=30`, ~3× the predecessor data). The trajectory shows the expected epoch-2 LR-transition dip (val_acc 0.67/0.65) followed by recovery to 0.79 by epoch 3, then a stable back half holding 0.79-0.82 (epochs 8-14 mostly ≥ 0.79). This matches the Phase 1 lesson that the warmup→peak-LR transient is not catastrophic forgetting.

**Argus-CCT** reached **0.7751** from scratch (no pretraining), +4 points over the predecessor. The trajectory is high-variance, the signature of from-scratch transformer training: it starts at chance (0.51, epoch 0), swings between ~0.49 and ~0.77 through the middle epochs, climbs past 0.73 from epoch 15 on, peaks at epoch 22, and plateaus around 0.77 in the tail (epochs 28-29 at 0.770/0.774). It did not collapse; it converged to a useful model, just noisily.

## Observation: the RN34 log holds two training runs

`train_log.jsonl` for Argus-RN34 contains **two sets of epochs 0-14** (the training cell was run twice; `train()` appends to the log). Both runs peaked near 0.82 (run A epoch 8 = 0.8181; run B epoch 11 = 0.8158), so the headline number is solid either way. But because each run's per-epoch `epoch_NNN.pt` overwrites the previous on HF, `patrickjreed/argus-rn34-v0` holds whichever run finished last.

**Carry-forward for Phase 3:** notebook 06 must re-verify the loaded RN34 checkpoint's val_acc on the val split before interpreting it, and pick the actual best-epoch checkpoint present on HF (one eval pass). Argus-CCT is a single clean 30-epoch run with no such ambiguity.

## Implications for Phase 3

- **Both models are worth interpreting.** A pretrained ResNet (0.82) and a from-scratch transformer (0.78) both learned the disease, so the cross-architecture agreement analysis compares two genuinely-trained models rather than one model and one near-chance baseline.
- **The accuracy gap (0.82 vs 0.78) is modest**, so disagreements in what they attend to are more likely to reflect architecture/initialization than one model simply being much weaker. That makes the agreement metric more informative.
- **The donor probe is the key remaining question.** Both models hit 0.78-0.82, but Phase 3's donor-confound probe (now correctly labeled via `yield_donor`) must rule out that the gain rides donor identity rather than disease biology before any channel/cell-type claim stands.

## Decision and next action

Phase 2 complete; gate cleared by both models. Next: cut over the `argus-cells` migration (training no longer runs against `main`), then write `notebooks/06_phase_3_analysis.ipynb` to run the full harness on both production checkpoints and produce the four Phase 3 deliverables.
