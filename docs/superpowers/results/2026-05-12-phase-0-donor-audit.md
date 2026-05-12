# v0 Phase 0 Result — Donor Structure Audit

**Spec:** [`docs/superpowers/specs/2026-05-12-argus-cells-design.md`](../specs/2026-05-12-argus-cells-design.md), §3 + §5.
**Plan:** [`docs/superpowers/plans/2026-05-12-argus-cells-phase-0-donor-audit.md`](../plans/2026-05-12-argus-cells-phase-0-donor-audit.md).
**Notebook:** [`notebooks/03_donor_audit.ipynb`](../../../notebooks/03_donor_audit.ipynb).
**Run date:** 2026-05-12
**Hardware:** Colab Free CPU.

## Quantitative findings

```
============================================================
PHASE 0 AUDIT SUMMARY
============================================================

Donor counts per condition:
  control: 24 donor lines  [OK]
  deletion: 24 donor lines  [OK]

Per-(cell_type, condition) imbalance (donor-balance CV):
  ( astro,  control): CV=0.163, N_donors=24
  ( astro, deletion): CV=0.000, N_donors=24
  (neuron,  control): CV=0.000, N_donors=24
  (neuron, deletion): CV=0.000, N_donors=24
  (progen,  control): CV=0.000, N_donors=24
  (progen, deletion): CV=0.000, N_donors=24
  (  stem,  control): CV=0.000, N_donors=24
  (  stem, deletion): CV=0.000, N_donors=24

Crop budget at crops_per_site=30:
  upper-bound crops: 315,960
  realistic (60% yield): 189,576
  realistic (90% yield): 284,364
```

## Interpretation

- **Donor coverage adequacy.** 24 donor lines per condition (48 total) sits an order of magnitude above the N ≥ 3 gate. With this many donors the Phase 1 donor probe has a meaningful denominator: random-chance accuracy on a 48-way donor classification is 1/48 ≈ 2.08%, so any non-trivial probe accuracy is detectable above noise. The probe will also have enough per-donor wells (4 wells per donor per cell type for the vast majority of combinations) to give the val split real coverage.

- **Imbalance severity.** 7 of 8 (cell_type × condition) groups show CV = 0.000, meaning every donor in those groups has identical well counts. The single non-zero group is astro/control at CV = 0.163, driven by two donors out of 24: donor 8 has 1 well and donor 20 has 3 wells, while the remaining 22 astro/control donors each have the standard 4 wells. Severity is negligible: weighted analyses can ignore this safely at Phase 1, and Phase 2 will absorb the imbalance via per-well crop sampling.

- **Crop budget.** At crops_per_site=10 the realistic yield is 63k-95k crops (60%-90% yield band), which already hits the Phase 2 target of 50-100k crops. At crops_per_site=30 the realistic yield is 190k-285k crops, leaving CCT-from-scratch substantial headroom if it needs more data to converge. Recommendation: start Phase 2 with crops_per_site=10 since it matches the Phase 0.5 and Phase 1 recipes and requires no recipe change, then escalate to 30 (190k-285k) or 50 (315k-475k) only if the from-scratch CCT underperforms at 10.

## Gate decision

**PROCEED to Phase 1.**

- N ≥ 3 gate: met. 24 donor lines per condition is 8x the minimum.
- Imbalance gate: tolerable. Max CV across the eight (cell_type, condition) cells is 0.163 (astro/control). The other seven cells are at 0.000.
- Crop budget gate: sufficient at crops_per_site=10. Realistic yield 63k-95k crops covers the 50-100k Phase 2 target without recipe escalation.

Phase 1 begins against the predecessor's 0.73 baseline checkpoint as the harness target.

## Implications for Phase 1 + Phase 2 design

- **Phase 1 donor probe baseline.** 48 donors gives random-chance accuracy ≈ 2.08% (1/48). A probe at > 10% accuracy on val embeddings signals the encoder retains donor structure beyond chance; a probe at > 50% signals a clear donor confound that needs intervention before disease classification claims can stand.
- **Phase 1 donor split for probe.** The probe trains on train-split embeddings and evaluates on val-split embeddings using the same well-level split as the disease classifier. Reusing the split keeps the probe's findings directly comparable to the disease head's findings.
- **Phase 2 crop budget recommendation.** crops_per_site=10 for the initial production run of both Argus-RN34 and Argus-CCT. Expected yield ~63-95k crops, matching the empirical Phase 0.5 and Phase 1 yield ratio. If CCT-from-scratch fails to converge at this scale, escalate to crops_per_site=30 (realistic 190-285k crops) as the second attempt before considering 50.
- **Open question for Phase 2.** Whether the astro/control imbalance (CV = 0.163, two donors with reduced wells) is worth correcting via weighted crop sampling. At this magnitude the answer is probably no, but flag it explicitly in the Phase 2 design doc so the choice is made on purpose rather than by omission.

## Open observations (not blockers)

- Donor IDs are integers 1-48 in two disjoint ranges: control = 1-24, deletion = 25-48. No donor appears in both conditions. Each donor line corresponds to a unique iPSC line and a unique (line, condition) pair, so the donor confound check is well-formed: a model that learns donor identity cannot win on disease classification by accident, because donor identity and disease label are independent given the cell-type covariate.
- Two astro/control donors have reduced well counts: donor 8 (1 well) and donor 20 (3 wells). Likely reflects failed wells in the original assay. Negligible aggregate impact.
- Total dataset footprint: 10,532 sites across 1,724 wells across 48 donor lines.
