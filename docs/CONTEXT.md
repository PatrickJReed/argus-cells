# argus-cells — project context

The strategic background behind this project: why it exists, how it evolved from a BMS-lineage reproduction into a focused interpretability study, the dataset and architecture rationale, and the short- and long-term goals. Companion to `README.md` (public-facing technical framing) and `CLAUDE.md` (working context for Claude Code sessions).

The repo and Python package are now named `argus-cells` / `argus_cells` (migrated in Phase 2 from `cerberus-neuro` / `cerberus_neuro`). See [The reframe](#the-reframe-cerberus-neuro--argus-cells) below and `docs/superpowers/specs/2026-05-12-argus-cells-design.md` for the full design.

## Why this project exists

This project is part of a deliberate portfolio strategy by **Patrick J. Reed, Ph.D.** (computational biologist, 15+ years; most recently Principal Scientist at Bristol Myers Squibb). It closes a specific portfolio gap: **prior work that lives only inside a former employer cannot be cited or shown to recruiters.**

At BMS in 2025, Patrick built a multi-task foundation-model proof-of-concept for the Neurobot high-throughput screening platform, a Cerberus-inspired ResNet34 with a shared encoder and three task heads (cell-type classification, virtual staining, disease-state classification) trained on the public Broad NeuroPainting dataset. The internal version was completed as a POC but not transferred to BMS internal data or productionized into Neurobot workflows before the April 2026 layoff. Patrick can describe the work in interviews but cannot point recruiters to a citable artifact.

This repo is the public, citable answer. No BMS internal data, no internal hyperparameters, no internal infrastructure: the public NeuroPainting data, an architecture in the same family, and public training runs that anyone can run, cite, fork, or extend. The decision to rebuild publicly was made during the 2026-05-05 portfolio refresh, alongside the sibling `cellduet` project (multimodal perturbation concordance).

## The reframe: cerberus-neuro → argus-cells

The project started as `cerberus-neuro`, a faithful three-headed reproduction of the BMS POC. During the build, two task-specific models shipped and validated the data pipeline and training infrastructure end-to-end on real biology: a cell-type classifier (val acc 0.9905) and a 6-channel disease classifier (val acc 0.7311). With the infrastructure proven, the framing was reconsidered on 2026-05-12.

The conclusion: a single honest disease classifier plus a rigorous interpretability harness is a stronger, more defensible artifact than three heads competing for encoder capacity. The interesting scientific question on this dataset is not "can a model separate 22q11.2 deletion from control" (it can) but "what is the model using to do it": which Cell Painting channels carry the signal, whether that differs by cell type, and whether the model is quietly riding donor identity rather than disease biology. That question doubles as a methods spine (a careful comparison of attribution methods across two architectures) and a biology payoff, serving both ML-methods and biology readers from one artifact.

So the multi-task reproduction framing was retired. The project is now `argus-cells`: the many-eyed watcher, a 6-channel model attending to morphology and then made to show where it looked.

## Why these design choices

### Why disease classification plus interpretability (not a faithful reproduction)

- **The reproduction was a means, not the end.** The portfolio gap is "demonstrate Patrick can build and ship vision models on bio imaging, and reason rigorously about what they learn." A polished interpretability study demonstrates more of that than a literal re-implementation of an internal model.
- **Honesty is easier and stronger.** A reproduction invites the question "how close to the internal numbers." An open-ended interpretability study makes no parity claim; it reports whatever the data says, framed as "the model says X" rather than "we proved X."
- **Interpretability is the differentiated skill.** Many candidates can train a classifier. Fewer can build a reusable attribution harness, stratify it by cell type, and explicitly test a donor confound before making a biological claim.

### Why this dataset (Broad NeuroPainting)

- **Public.** On the Cell Painting Gallery (AWS Open Data); permissive licensing for non-commercial use, no credentials required.
- **Disease-relevant.** 22q11.2 deletion is a well-characterized neuropsychiatric risk factor; the iPSC plus Cell Painting framing generalizes to neurodevelopmental and neurodegenerative work.
- **Multi-channel in the right way.** Five fluorescence channels plus brightfield give the per-channel ablation its meaning: a per-channel importance question only exists because the channels are biologically distinct (DNA, mitochondria, AGP, ER, RNA).
- **Well-structured for a confound check.** 48 cell lines, 24 per condition, in disjoint donor ranges (control 1-24, deletion 25-48). Donor identity and disease label are independent given cell type, so the donor probe is well-formed.
- **Manageable scale.** 48 lines across 4 cell types, large enough to be non-trivial, small enough to fit Colab Free with subset scoping.
- **Cited and traceable.** Tegtmeyer et al., *Nat Commun* 2025 ([10.1038/s41467-025-61547-x](https://doi.org/10.1038/s41467-025-61547-x)) is the canonical reference.

### Why two paired classifiers (Argus-RN34 + Argus-CCT)

- **Two architectures, chosen for complementary attribution surfaces, not for an accuracy tournament.** ResNet34 exposes GradCAM on its last conv stage; the Compact Convolutional Transformer exposes attention rollout on its encoder. Where two credible methods on two architectures agree, a biological claim is stronger; where they disagree, that disagreement is itself a finding worth reporting.
- **Argus-CCT echoes Patrick's actual BMS work.** The TDP43/STMN2 screening models he built at BMS (2024-2025) used a Compact Convolutional Transformer in the same Cell Painting / iPSC-neuron domain. Choosing CCT as the second backbone is an authentic extension of shipped prior work, not an arbitrary architecture pick.
- **ResNet34 and CCT are both small enough for cheap compute.** ResNet34 fine-tunes on an L4; CCT-from-scratch converges in 1-2 A100 hours. Neither is the "200M-parameter model on a tiny dataset" anti-pattern.
- **Pretrained vs from-scratch is a deliberate contrast.** Argus-RN34 starts from ImageNet1K_V1; Argus-CCT trains from scratch. The pair also probes whether initialization changes what the model attends to.

### Why Colab + Docker (not pure cloud or pure local)

Documented in `docs/SETUP.md`. Short version:
- Colab handles iteration speed and Free-tier compute for the harness and the pretrained-model training.
- Docker captures the environment so the same code runs locally, on Lambda/RunPod for the CCT-from-scratch A100 run, or in a production-style deployment.
- Multi-platform Docker portability is itself a credentialing signal; baked in from day one is cheap, retrofitted later is expensive.
- No paid-cloud commitment until Phase 2: one A100 session for CCT-from-scratch ($2-15), not recurring spend.

## Strategic positioning

The artifact supports applications to:

- **Recursion.** Image-based perturbation and Cell Painting are core to their platform; argus-cells is on-domain, and the interpretability framing matches how a screening company reasons about phenotypic hits.
- **Insitro.** Cellular phenotyping plus ML on iPSC imaging maps directly.
- **Iambic Therapeutics.** Multi-task and attribution-aware vision in drug discovery.
- **Anthropic Applied AI Engineer, Life Sciences.** Pairs with `cellduet` to show breadth: a multimodal embedding analysis and a trained-from-data interpretability study. The methods-spine framing (rigorous attribution comparison, explicit confound test) is the part that reads as applied-AI rigor rather than just model training.
- **Pharma comp-bio roles** with Cell Painting / phenotypic screening (BMS, Lilly, Pfizer, AbbVie). Patrick's BMS imaging-AI work gets a citable public companion.
- **Generally**: any role asking "show me your prior work on vision in biology, and how you reason about what the model learned."

## Connection to other portfolio work

- **`~/Sandbox/cellduet/`** — sibling project. cellduet reasons analytically across pre-computed multimodal embeddings (transcriptomic + morphological); argus-cells trains vision models on the morphological side and dissects them. Together they show Patrick can both train domain-specific models and reason across pre-trained embedding spaces.
- **TDP43/STMN2 CCT screening model (BMS, 2024-2025)** — the direct lineage for Argus-CCT. Same Cell Painting / iPSC-neuron domain, same Compact Convolutional Transformer family.
- **`docs/NeuroPainting_MultiTask_Model_Dossier.md`** in NewRoleEfforts — reference on the BMS internal multi-task version, dataset details, and connections to Tegtmeyer et al.
- **`PatrickReed_Accomplishments_Bank.md`** "Neurobot Multi-Task Foundation Model: Proof-of-Concept (BMS, 2025)" entry — the internal version's record. Patrick should reference argus-cells from that entry once Phase 4 ships.

## Honestly acknowledged gaps

Surfaced intentionally, not buried:

- **This is applied work on public data, not novel architecture research.** The value is "Patrick can build a classifier, then rigorously interrogate it," not "Patrick invented a new model." Both architectures are off-the-shelf families.
- **The disease classifier may carry a donor confound.** With 24 donor lines per condition, the model could in principle separate conditions partly by donor identity. The project does not assume the confound away; the donor probe explicitly measures it, and a high donor/disease accuracy ratio would be reported as a limitation rather than hidden.
- **Attribution methods can disagree, and none is ground truth.** GradCAM, attention rollout, Integrated Gradients, and channel ablation can attend to different features. Single-method claims are treated with caution; cross-architecture and cross-method agreement is the evidence bar.
- **CCT-from-scratch convergence is untested at this data scale.** It may need more crops or epochs than budgeted. Mitigation: Argus-RN34 trains in parallel as the fallback; if CCT fails, the harness still runs on one architecture and the project still ships.
- **22q11.2 deletion is one disease state.** The model does not generalize to other conditions without retraining.
- **Patrick is not an author on Tegtmeyer et al.** This applies an architecture and an interpretability harness to their public data; it is not a re-implementation of their methodology, and it makes no claim on their findings.

## Short-term goals (argus-cells v0, ~7 weeks evening work)

Per the design spec's phased plan:

1. **Phase 0.** Donor + dataset audit; hard gate before training. (Done: PROCEED, 24 donors/condition.)
2. **Phase 1.** Build and validate the interpretability harness against the existing 0.73 baseline. (Code complete and unit-tested; Colab run and results doc pending.)
3. **Phase 2.** Train Argus-RN34 and Argus-CCT on a scaled crop budget; initialize the `argus-cells` repo and package.
4. **Phase 3.** Full analysis: per-cell-type by per-channel ablation tables, cross-architecture saliency agreement, donor-probe confound result, honest biology writeup.
5. **Phase 4.** Polish: `argus-cells` on GitHub and PyPI, README narrative, HF model cards.

## Long-term goals (v1+, opportunistic)

- **Full-scale training run** on rented A100 (full resolution, all plates, full convergence). Push to HF as `patrickjreed/argus-*-v1`.
- **Leave-one-donor-out cross-validation** if the Phase 1/3 donor probe surfaces a real confound that needs full quantification.
- **Architecture or attribution extensions** beyond the two specified models, only if a phase surfaces a specific reason.
- **HF Spaces demo**: upload a Cell Painting crop, get the disease call plus its per-channel attribution live.
- **Possible preprint or workshop submission** if v1 quality justifies it.

## Source documents

- `docs/superpowers/specs/2026-05-12-argus-cells-design.md` — the approved argus-cells design spec (the authoritative "what and why")
- `~/Sandbox/cellduet/docs/CONTEXT.md` — sibling project's strategic background; shares the broader portfolio rationale
- `~/NewRoleEfforts/docs/portfolio_project_brainstorm.md` — overall portfolio strategy and project rankings
- `~/NewRoleEfforts/docs/NeuroPainting_MultiTask_Model_Dossier.md` — reference on the BMS internal POC
- `~/NewRoleEfforts/PatrickReed_Master_Resume.md` — gold-standard claims source for any prose about Patrick's prior work
- `~/NewRoleEfforts/PatrickReed_Accomplishments_Bank.md` — extended detail on BMS, Ionis, DNAtrix, Salk projects
- `~/NewRoleEfforts/PatrickReed_Claims_Audit.md` — claims flagged for revision; consult before writing about prior work
- `~/NewRoleEfforts/PatrickReed_Writing_Style.md` — voice rules
