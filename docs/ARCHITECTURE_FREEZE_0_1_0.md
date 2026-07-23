# CARS-R 0.1.0 — Architecture Freeze

## Status

CARS-R 0.1.0 is now an architecture-frozen research milestone.

The purpose of the freeze is to make the next larger-data experiment interpretable. Once an architecture has been selected because of pilot results, changing its mechanisms while scaling the data confounds architecture improvement with data-scale improvement.

## Frozen core

The 0.1.0 identity is defined by the following core path:

```text
raw UTF-8 bytes
→ local sliding-window byte GQA
→ learned bounded causal patching
→ order-aware patch compression
→ span-aware patch geometry
→ CPLA global patch attention
→ shared recurrent patch refinement
→ completed-patch / local-byte fusion
→ local byte decoder
→ next-byte logits
```

For a result to be called **CARS-R 0.1.0**, do not silently change:

- the byte vocabulary/serialization contract;
- learned patch-router semantics;
- minimum/maximum patch constraints;
- order-aware patch representation;
- CPLA attention definition;
- span-position definition;
- canonical recurrence configuration;
- local exact-byte residual/fusion semantics;
- causal patch-commit rule;
- training objective definition.

Bug fixes are allowed only when they restore the documented semantics. They must be recorded and affected comparisons rerun when necessary.

## What may change without creating a new architecture

Engineering changes are allowed when they preserve the mathematical computation within numerical tolerance, for example:

- fewer temporary allocations;
- fused tensor operations;
- better buffer reuse;
- equivalent SDPA implementations;
- more efficient data loading;
- mixed-precision safety fixes;
- serialization/checkpoint tooling;
- documentation and evaluation corrections.

For a speed optimization to be considered semantics-preserving, verify on a fixed corpus:

1. identical hard patch boundaries where exact equality is expected;
2. full-forward logits within a stated tolerance;
3. incremental-generation logits within tolerance;
4. no change in causal behavior;
5. no change in parameter count except non-trainable buffers.

## 0.1.1-speed branch

A future `0.1.1-speed` branch may optimize execution without changing the research hypothesis. Candidate targets include:

- router scan fusion;
- packed patch compression;
- fewer assignment intermediates;
- CPLA kernel fusion;
- preallocated patch/cache buffers;
- compilation of stable submodules;
- better ragged batching.

Changing recurrence depth from the canonical 0.1.0 value, replacing the router, altering CPLA mathematics, or adding a new memory/attention mechanism is **not** a pure speed optimization and should be treated as an ablation or new architecture revision.

## Why no new mechanisms are being added now

The pilot already contains enough novelty to ask meaningful questions. More mechanisms would make it harder to identify why the model succeeds or fails.

The immediate research question is now empirical:

> Does the frozen CARS-R architecture remain competitive as training data and context scale increase, and does patch/CPLA compression produce a favorable quality–memory–compute curve?

That question is more valuable than another unvalidated module.
