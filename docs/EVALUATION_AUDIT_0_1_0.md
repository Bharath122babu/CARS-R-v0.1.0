# CARS-R 0.1.0 — Evaluation Audit Trail

## Why this document exists

The first evaluation pass produced attractive-looking numbers, but a subsequent source-level audit found several evaluator mistakes. Those results were not silently retained. This document records the corrections so future work does not repeat them.

Research credibility depends as much on correcting evaluation code as on correcting model code.

## Correction 1 — CARS-R auxiliary loss was mixed into BPB

The CARS-R forward result exposes a pure language-model loss and a compression/routing auxiliary loss. The training objective may combine them, but cross-model BPB must use pure language-model negative log-likelihood only.

Early evaluation used the combined `out.loss` for CARS-R while baselines used only LM loss. That penalized CARS-R with a term the baselines did not have.

### Correct rule

For all architectures, compute evaluation NLL from the next-position logits or the explicitly exposed pure LM loss. Report:

\[
\mathrm{BPB}=\frac{\sum_i \mathrm{NLL}^{\mathrm{bits}}_i}{\sum_i B_i},
\]

where `B_i` is the number of original UTF-8 bytes represented by record `i`.

Auxiliary compression/router losses are logged separately.

## Correction 2 — pseudo-paired bootstrap

The early evaluator copied a batch-average loss once for every example in the batch and called those values per-record losses. It then truncated arrays from different input representations to a common length. This did not constitute a paired bootstrap over the same original records.

### Correct rule

Store, for each original record and each model:

- record ID;
- domain;
- raw UTF-8 byte count;
- total LM NLL bits;
- BPB;
- native model-position count;
- accuracy numerator and denominator where relevant.

Bootstrap the same record IDs across models.

The corrected pilot bootstrap values recorded in `PILOT_RESULTS_0_1_0.md` supersede the invalid early intervals.

## Correction 3 — patch analysis used the wrong router tensor

The router exposes information mass as well as actual hard boundary decisions. Early analysis thresholded the information-mass tensor as though it were a boundary probability, resulting in no real measured patches, then included hard-coded illustrative segmentations.

### Correct rule

Patch analysis must use the actual model outputs/state:

- `hard_boundaries`;
- `patch_lengths`;
- `patch_count`;
- patch spans/starts/ends.

Representative examples must be rendered from those measured boundaries, never manually authored.

The corrected pilot measured a mean patch length of 5.1026 bytes and median of 5 bytes.

## Correction 4 — recurrence interpretation

The early report's prose claimed `K=2` was optimal despite its own diagnostic values not supporting that conclusion. It also omitted the requested `K=0` case.

### Correct rule

Do not hard-code qualitative classifications. Derive conclusions from logged numbers, and distinguish inference-time depth probing from a fully controlled retraining ablation.

Current status: deeper-than-one recurrence showed no clear pilot benefit; `K=0` remains required before a later architecture revision.

## Correction 5 — decode benchmark recomputed prefixes

An early "decode" benchmark repeatedly ran the model over a growing prefix rather than exercising the real incremental cache/session API. Those timings were not cached autoregressive decode timings.

### Correct rule

Benchmark the actual generation-session/incremental-cache path and report separately:

- prefill;
- incremental decode;
- cache occupancy;
- allocated cache capacity;
- source-byte throughput;
- native step throughput.

## Correction 6 — cache units and topology

A previous estimate treated a count of latent floating-point values as if it were already bytes and multiplied it by an incorrect number of identical caches. CARS-R has multiple cache types with different lifetimes: local byte ring caches, patch-layer CPLA caches, recurrent memory, and open-patch state.

### Correct rule

Prefer direct cache-object byte accounting. When using a theoretical law, state dtype size, number of layers, patch ratio, latent dimensions, and whether the value represents occupied memory or allocated capacity.

## Correction 7 — target alignment must be executable, not asserted

Because an earlier dense baseline had a label-shift bug, future evaluators must test the exact input-to-target mapping rather than store a hard-coded `aligned=true` field.

Required check:

`logits[t]` must score the intended next symbol under the model's serialization convention, with padding/special-token positions treated consistently across models.

## Consequence for the old pilot test

The 400-record pilot test has already been inspected during development. It remains useful as a **pilot held-out evaluation**, but it is no longer a pristine future final benchmark for a modified architecture.

Any later CARS-R revision should use a new frozen blind test set or external benchmarks that were not used to guide that revision.

## Publication rule

Only the corrected results documented in `PILOT_RESULTS_0_1_0.md` should be cited from the 0.1.0 pilot.

Never recover a more favorable early metric merely because it looks better.
