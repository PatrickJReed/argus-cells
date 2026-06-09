# argus-cells Phase 1 Result — Interpretability Harness Validation

**Spec:** [`docs/superpowers/specs/2026-05-12-argus-cells-design.md`](../specs/2026-05-12-argus-cells-design.md), §4 + §5 + §6.
**Plan:** [`docs/superpowers/plans/2026-05-12-argus-cells-phase-1-interpretability-harness.md`](../plans/2026-05-12-argus-cells-phase-1-interpretability-harness.md).
**Notebook:** [`notebooks/04_phase_1_harness.ipynb`](../../../notebooks/04_phase_1_harness.ipynb).
**Run date:** 2026-06-09.
**Hardware:** Google Colab.
**Harness target:** `patrickjreed/cerberus-neuro-v0-baseline`, `epoch_013.pt` (best-epoch val_acc 0.7311). This is the 6-channel `BaselineDiseaseClassifier`, the predecessor of the Phase 2 production model Argus-RN34. Phase 1 deliberately validates the harness against this existing checkpoint, not a new model.

## Headline

The Phase 1 interpretability harness runs end-to-end against the 0.73 baseline checkpoint. All four attribution / probe methods (channel ablation, GradCAM, Integrated Gradients, donor probe) plus the cell-type stratification produce correctly-shaped outputs on a balanced 512-crop validation batch, with no manual intervention after the data load. Per the spec, Phase 1 makes no accuracy claims; the goal is harness validation, and it is met.

Phase 1's success criterion (spec §6): "every attribution method + donor probe + cell-type stratification runs end-to-end on the existing 0.73 baseline without manual intervention; no new accuracy claims." That criterion is satisfied, with two carry-forward fixes for Phase 2 (below).

## Quantitative findings (preliminary; harness-validation batch, not a production result)

All numbers come from a single balanced validation batch (target 64 crops per (cell_type, condition), 512 total) drawn from the harness-validation subset, not the checkpoint's original training/val split. Treat them as evidence the harness produces sensible, representative output, not as biological findings.

**Channel ablation, whole batch (accuracy drop when each channel is zeroed):**

| Rank | Channel | Accuracy drop |
|---|---|---|
| 1 | BF (brightfield) | +0.1738 |
| 2 | Mito | +0.1230 |
| 3 | DNA | +0.1211 |
| 4 | RNA | +0.1016 |
| 5 | ER | +0.0879 |
| 6 | AGP | +0.0566 |

Batch baseline accuracy: **0.6934**, consistent with the checkpoint's reported best-epoch val_acc 0.7311 (the small gap is expected: a 512-crop balanced batch from a different subset, not the original val split). Every channel contributes a non-trivial drop, with brightfield the single most-used channel and AGP the least.

**Cell-type stratified channel ablation (per-sample target-confidence drop, top two channels per cell type):**

| Cell type | Top channel | 2nd channel |
|---|---|---|
| stem | BF (+0.248) | RNA (+0.179) |
| progen | DNA (+0.110) | Mito (+0.048) |
| neuron | BF (+0.062) | Mito (+0.046) |
| astro | BF (+0.163) | Mito (+0.155) |

All four cell types now have non-zero, differentiated channel profiles (the full 4×6 table is in the notebook's cell-11 output). The per-cell-type structure is the harness's core finding-shaped output, and it is now representative.

**Disease linear probe (valid):** val_acc **0.5859** vs 0.5 chance. The encoder's frozen 512-dim embeddings carry linearly-decodable disease signal above chance, weaker than the model's own head (0.69 on this batch), as expected for a linear probe on frozen features.

**Donor linear probe (NOT a valid confound measurement this phase):** val_acc 0.0391 vs 1/48 = 0.0208 chance; reported donor/disease ratio 0.067. The notebook printed "OK, no donor confound," but this verdict is not supported. The donor labels were assigned by manifest position rather than tracked per embedding, so they are effectively scrambled relative to the embeddings (the dataloader yields 10 crops/site across 4 workers; embedding *i* is not manifest row *i*). A scrambled-label probe sits at chance whether or not the encoder encodes donor identity, so 0.039 ≈ chance tells us nothing about a real confound. This is a plumbing check (the probe code runs and returns sane shapes), not a result. See carry-forward fix 1.

## What this validates, and what it does not

**Validates:**
- All four methods execute end-to-end and return correctly-shaped `AttributionResult` / probe outputs on real 6-channel Cell Painting crops.
- The common attribution interface and the analysis layer (stratification, figures) work against a real checkpoint.
- The disease signal is linearly present in the encoder embeddings (probe 0.586 > 0.5).
- On a balanced batch the model's behavior is consistent with its training (0.69 batch accuracy vs 0.73 reported).

**Does not establish (out of scope for Phase 1, by design):**
- Any biological claim. The channel rankings and per-cell-type profiles are preliminary, on the old baseline checkpoint and a 512-crop batch.
- Any donor-confound conclusion (the probe is mislabeled this phase).
- Any cross-architecture result (Argus-CCT does not exist yet; attention rollout and saliency-agreement are Phase 2).

## Caveats and honest framing

- **Old checkpoint, not the production model.** The target is `cerberus-neuro-v0-baseline` (~16k-crop training, old recipe). The Phase 2 Argus-RN34 / Argus-CCT will be retrained on a larger crop budget; harness outputs should be re-read on those models.
- **Small batch.** 512 crops, 64 per (cell_type, condition). Enough to validate the pipeline, not to support effect-size claims.
- **One plausible-looking observation, explicitly not a claim.** astro's second-strongest channel is Mito (+0.155), which is directionally consistent with mitochondrial structure being informative in deletion astrocytes. This is the kind of finding-shaped output Phase 3 is built to test rigorously on the production models; it is recorded here only as evidence the harness yields interpretable per-cell-type structure, not as a result. The spec is explicit that biology output is "the model says X," never "we proved X," and that no pre-committed published claim (including Tegtmeyer 2025) is being chased.

## Gate decision

**PROCEED to Phase 2.** The harness is validated end-to-end on the existing baseline. Phase 2 (train Argus-RN34 + Argus-CCT on the scaled crop budget, initialize the `argus-cells` repo/package) is unblocked.

## Carry-forward fixes for Phase 2

1. **Donor-ID routing through the dataloader (required before the donor probe is meaningful).** `NeuroPaintingDataset.__iter__` must yield `Metadata_line_ID` per crop so each embedding keeps its true donor label. Until then the donor probe and the donor/disease ratio cannot be interpreted as a confound measurement. This is the single most important Phase 2 plumbing item, since ruling out the donor confound is a stated project goal.
2. **Promote the balanced-collection logic into a reusable helper.** The balanced (cell_type, condition) bucketing now lives inline in notebook cell 7. Phase 3's production analysis should call a shared helper (e.g. in `analysis/`) so every attribution run is balanced by construction rather than by notebook discipline.

## Lessons / open observations

- **A sampling bug produced a falsely-clean-looking first result.** The first run reported a global channel ranking, a baseline accuracy of 0.57, and a stratification table in which stem/neuron/astro all showed exactly +0.000. The zeros were not "no effect": `stratify_channel_scores_by_cell_type` emits `mean_score=0` for cell types with `n_samples=0`, and the collection cell had grabbed the first 512 crops in loader order from a `shuffle=False`, cell-type-grouped manifest, so the batch was a single cell type (progen). The "global" ranking and the 0.57 baseline were therefore one-cell-type artifacts. Fixing the collection to bucket across (cell_type, condition) raised the baseline to 0.69 and populated all four cell-type rows.
- **The general rule:** an attribution / stratification result is only as trustworthy as the batch it ran on. A degenerate batch can produce numbers that look like findings (a sorted channel ranking, a confident donor verdict) while measuring almost nothing. Check the per-group sample counts before reading any stratified table. The balanced-collection cell now prints per-(cell_type, condition) counts for exactly this reason.
- **Two probe results from this run had different validity.** The disease probe was correctly paired (embeddings and disease labels from the same batch iteration) and is valid; the donor probe was positionally mislabeled and is not. Same code path, different label provenance, opposite trustworthiness. Phase 2's donor-ID routing closes the gap.

## Decision and next action

Phase 1 complete. Next: Phase 2 design + the donor-ID dataloader change, then train Argus-RN34 and Argus-CCT and re-run this harness on the production checkpoints.
