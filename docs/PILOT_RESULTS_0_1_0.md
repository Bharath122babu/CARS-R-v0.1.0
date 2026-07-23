# CARS-R 0.1.0 — Controlled Pilot Results

## Purpose

This document records the first controlled training evidence for the frozen CARS-R 0.1.0 architecture. It is deliberately separated from the mathematical notebook: the notebook records hypotheses and derivations; this file records what was actually observed in the pilot.

These results are **not** a frontier-model comparison and they are **not** a large-scale pretraining result. They are a controlled, approximately two-million-parameter, single-seed pilot intended to answer whether the architecture is functional and competitive enough to justify a larger replication.

### Evidence provenance

The corrected summary values recorded here were taken from the post-audit evaluation artifact containing `corrected_validation_summary.json`, `corrected_test_summary.json`, `corrected_bootstrap_results.json`, and `corrected_patch_analysis.json`. The artifact SHA-256 used while preparing this documentation was:

`fe2bafe3a4127f28c9608e57089e63f9e8733601d6013f06c91f5d5ea3bce628`

The checkpoints and raw per-record evaluator outputs are intentionally not bundled in this source-only architecture repository. A publication package should archive those separately.

## Experimental ladder

Five models were trained from scratch on the same raw source corpus:

| ID | Model | Input representation | Global attention | Parameters |
|---|---|---|---|---:|
| A | CARS-R 0.1.0 | UTF-8 bytes | CPLA over learned patches | 2,015,167 |
| B | Dense byte Transformer | UTF-8 bytes | GQA | 2,021,376 |
| C | Dense byte Transformer | UTF-8 bytes | MLA | 2,017,344 |
| D | Dense token Transformer | BPE-2048 | GQA | 2,031,624 |
| E | Dense token Transformer | BPE-2048 | MLA | 2,033,640 |

Parameter discrepancy relative to CARS-R remained below 1% for every baseline.

The pilot training exposure was approximately **1.42 million raw UTF-8 bytes**. This is intentionally recognized as a small-data regime.

## Corrected evaluation protocol

Cross-family language quality is measured in **bits per original UTF-8 byte (BPB)**. For every model the numerator is pure language-model negative log-likelihood; the denominator is the number of original source bytes represented by the evaluated records.

CARS-R's compression regularizer is a training auxiliary objective and is **not** part of reported BPB.

Byte next-position accuracy and token next-position accuracy are model-specific diagnostics and are not compared directly with one another.

## Corrected validation snapshot

| Model | Validation BPB | Next-position accuracy |
|---|---:|---:|
| CARS-R 0.1.0 | **1.89223** | 65.61% byte |
| Dense byte GQA | 2.00147 | 63.82% byte |
| Dense byte MLA | **1.89770** | 64.46% byte |
| Dense token GQA | 1.96660 | 36.20% token |
| Dense token MLA | **1.95841** | 37.58% token |

The corrected validation calculation differs from early experiment summaries because the early evaluator had normalization/alignment problems. Only the corrected values should be cited.

## Corrected held-out pilot test

The pilot test contained **400 records across 16 domains** and was not used for training or checkpoint selection before the first held-out evaluation.

| Rank | Model | Test BPB | Next-position accuracy |
|---:|---|---:|---:|
| 1 | Dense token MLA | **1.93224** | 38.78% token |
| 2 | Dense byte MLA | **1.93592** | 63.87% byte |
| 3 | **CARS-R 0.1.0** | **1.94020** | 64.66% byte |
| 4 | Dense token GQA | 1.96774 | 35.80% token |
| 5 | Dense byte GQA | 2.06745 | 62.66% byte |

### Differences relative to CARS-R

- CARS-R vs dense byte GQA: **-0.12725 BPB** — CARS-R lower/better.
- CARS-R vs dense token GQA: **-0.02754 BPB** — CARS-R lower/better.
- CARS-R vs dense byte MLA: **+0.00428 BPB** — byte MLA slightly lower, effectively unresolved at this pilot scale.
- CARS-R vs dense token MLA: **+0.00796 BPB** — token MLA slightly lower, effectively unresolved at this pilot scale.

## Paired evaluation bootstrap

A corrected 10,000-resample paired evaluation bootstrap reported:

| Comparison | Mean CARS-R − baseline BPB | 95% interval | P(CARS-R lower BPB) |
|---|---:|---:|---:|
| CARS-R vs byte GQA | -0.12721 | [-0.13738, -0.11748] | 1.0000 |
| CARS-R vs byte MLA | +0.00426 | [-0.00836, +0.01734] | 0.2564 |
| CARS-R vs token GQA | -0.02761 | [-0.04374, -0.01108] | 0.9995 |
| CARS-R vs token MLA | +0.00797 | [-0.00964, +0.02604] | 0.1931 |

These intervals quantify uncertainty over the **held-out examples for one set of trained checkpoints**. They do not measure training-seed variance. Architecture-level significance therefore still requires multiple independent training seeds.

## Learned patch behavior

The corrected patch analysis used the model's actual hard boundary outputs and reported:

- mean patch length: **5.1026 bytes**;
- median patch length: **5 bytes**;
- configured minimum: 2 bytes;
- configured maximum: 8 bytes.

Representative measured segmentations included variable groupings in English, programming, mathematics, and structured JSON. These examples show that the router is active and not simply equivalent to one-byte processing. They do **not** prove that patches are linguistic tokens. A stronger claim requires quantitative boundary analysis across domains and seeds.

At a mean length of 5.10 bytes, the expensive global patch sequence contains roughly one position for every five raw input bytes. CARS-R still performs local byte computation, so this ratio must not be confused with end-to-end compute reduction.

## Recurrence diagnostic

A preliminary zero-retraining diagnostic found no quality improvement when increasing recurrence beyond one update. The comparison tested `K=1`, `K=2`, and `K=4`; `K=0` was not measured in that diagnostic.

Consequently:

- the canonical 0.1.0 architecture remains frozen at its documented default for reproducibility;
- no claim is made that deeper recurrence improves language modeling;
- the next architecture decision should include a proper `K=0` control before recurrence is retained in a later revision.

## Efficiency status

The pilot demonstrates architectural functionality and quality competitiveness. It does **not** yet establish end-to-end efficiency superiority.

A corrected CARS-R-only cached-decoding diagnostic exists, but it is not a matched CARS-R-versus-MLA context-scaling study. The required next measurements are:

- actual occupied cache bytes, not only allocated buffer capacity;
- prefill latency;
- cached decode latency;
- UTF-8 bytes/second;
- peak VRAM;
- active global positions;
- the same measurements for CARS-R, byte MLA, and token MLA over increasing source-byte contexts.

## What the pilot supports

The pilot supports the following statement:

> At approximately two million parameters in a small synthetic-data regime, CARS-R 0.1.0 achieved lower corrected held-out BPB than matched byte-GQA and token-GQA controls and remained close to both byte-MLA and token-MLA controls, while operating directly on UTF-8 bytes and learning variable-length global patches.

## What the pilot does not support

It does not establish:

- superiority to MLA;
- superiority to tokenization in general;
- superiority at large model scale;
- long-context efficiency superiority;
- multi-seed statistical significance;
- production throughput superiority;
- comparison with Meta, DeepSeek, or any frontier-scale pretrained model.

## Why the result matters

The pilot clears the threshold for continued research. The architecture is no longer only a diagram or smoke-test implementation: it survived a controlled comparison against strong controls without collapsing, and its learned patch mechanism is measurably active.

The appropriate next move is therefore replication at substantially larger data scale, not adding more architectural mechanisms.
