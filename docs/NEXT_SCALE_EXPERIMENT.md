# CARS-R — Next-Scale Experiment

## Objective

The pilot trained on only about 1.42 MB of raw source text and showed clear overfitting. The next experiment increases data scale by roughly two orders of magnitude while keeping CARS-R 0.1.0 frozen.

The goal is not to maximize a single score. It is to test whether the pilot ranking survives substantially more training exposure.

## Dataset

Versioned corpus:

`cars_r_sdg_16_domains_combined_v0_4_96mb`

Build summary:

- total records: **59,200**;
- training records: **58,240**;
- validation records: **480**;
- held-out test records: **480**;
- domains: **16**;
- training JSONL bytes: **95,874,037** (~95.874 MB decimal);
- validation JSONL bytes: **599,572**;
- held-out test JSONL bytes: **604,177**;
- duplicate IDs: 0;
- exact duplicate prompts: 0;
- duplicate prompt-response pairs: 0;
- reported normalized-skeleton overlap between train/validation/test: 0.

Archive checksums used for the next-scale preparation:

- complete dataset ZIP: `132c68300ae3419e12d590fb9a198fa0bfd1c589b3f26e48f147c3f598edb2f4`;
- train+validation-only ZIP: `ae31e69cbce7c44e7af6a588ad5a24b90d3c74b5f8df43a962a079cc56a22d04`.

Difficulty distribution across the full corpus is approximately 40% easy, 40% medium, 20% hard.

### Important limitation

This remains a synthetic/programmatic corpus. Ninety-six megabytes of synthetic text is **not** equivalent to ninety-six megabytes of independently authored natural pretraining text. Compression ratio of the ZIP file is not a measure of semantic diversity.

## Three finalists

Only three models should consume the expensive next-scale budget:

1. **CARS-R 0.1.0** — research architecture;
2. **Dense byte MLA** — strongest same-input-unit control from the pilot;
3. **Dense token MLA** — strongest tokenized control from the pilot.

The GQA controls already served their role in the pilot and need not be retrained in every scale study.

## Fairness

All models should see the same raw source records.

Compare progress using:

- raw UTF-8 training bytes seen;
- raw bytes per optimizer step;
- wall-clock time;
- estimated/observed compute;
- native model positions as a diagnostic, not as the common exposure unit.

For token models, train the tokenizer on the training split only. Validation and test text must not influence tokenizer construction.

## Context matching

A byte context and a token context should represent approximately the same amount of original text. If byte context is 1024 and the measured tokenizer compression is `R_tok` bytes/token, use approximately:

\[
T_{token}\approx\frac{1024}{R_{tok}}.
\]

Round only for practical batching/alignment.

## Training objective and evaluation

CARS-R may train with:

\[
\mathcal L_{train}=\mathcal L_{LM}+\lambda\mathcal L_{compression},
\]

but reported cross-model BPB must use **only** the LM NLL.

Evaluation denominator is original UTF-8 bytes represented by the evaluated records.

## Initial training budget

Run **one full epoch first**.

Do not automatically continue to multiple epochs. At the end of epoch one, classify the validation trajectory as:

- still improving;
- plateaued;
- overfitting;
- unstable.

A second epoch requires an explicit decision based on the predeclared validation rule, not on held-out test performance.

## Seed policy

Start with seed 42 for the scale sanity run. If the three-model result remains scientifically interesting, replicate finalists with at least seeds 42, 43, and 44.

One seed is enough to find catastrophic failure; it is not enough for an architecture-level statistical claim.

## Held-out test policy

The new 480-record test split should remain sealed during training, checkpoint selection, learning-rate decisions, and architecture changes.

The old 400-record pilot test has already been inspected and must not be reused as a pristine test for a modified architecture.

## Required logging

All models:

- pure LM loss;
- total training loss where different;
- validation BPB;
- raw bytes seen;
- native positions seen;
- gradient norm;
- learning rate;
- wall-clock time;
- peak/reserved VRAM;
- raw source bytes/second.

CARS-R additionally:

- mean/median/std patch length;
- hard patch count;
- bytes per global patch;
- router gradient norm;
- compression auxiliary loss;
- recurrent update norm;
- CPLA cache dimensions.

## Hardware target

The current experiment target is an 8 GB VRAM GPU with 32 GB system RAM. Use microbatch calibration plus gradient accumulation rather than changing model architecture to fit memory.

## Decision after the run

The next research decision should be based on the shape of the scaling evidence:

- If CARS-R remains close to MLA while using materially fewer global positions/cache, prioritize efficiency and long-context profiling.
- If CARS-R quality falls substantially behind byte MLA, isolate patching/CPLA before adding features.
- If CARS-R improves relative to token MLA as data grows, replicate across seeds before interpreting that as a tokenizer-free advantage.
- If every model continues improving at one epoch, consider a second epoch before increasing model size.

Do not respond to a disappointing run by adding multiple new mechanisms at once.
