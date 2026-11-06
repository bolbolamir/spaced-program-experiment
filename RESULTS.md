# MiniLM direct weight editing: measured results

This is a small, synthetic shortcut-learning experiment inspired by weight-space programming, not a reproduction of the paper.

Device: mps; seed 42; training 2.6s; edit stage 0.5s.
Selected candidate: `semantic_head_0.5`; changed 770/22,713,986 scalar parameters.

Accuracy (correct / total):

| Method | Train (24) | Test (8) | Hard repair (8) | Hard test (10) |
|---|---:|---:|---:|---:|
| pretrained_semantic_head | 24/24 (100.0%) | 8/8 (100.0%) | 2/8 (25.0%) | 2/10 (20.0%) |
| trained_baseline | 24/24 (100.0%) | 8/8 (100.0%) | 1/8 (12.5%) | 0/10 (0.0%) |
| backbone_edit | 24/24 (100.0%) | 8/8 (100.0%) | 2/8 (25.0%) | 2/10 (20.0%) |
| selected_direct_edit | 24/24 (100.0%) | 8/8 (100.0%) | 2/8 (25.0%) | 2/10 (20.0%) |
| gradient_head_repair | 24/24 (100.0%) | 8/8 (100.0%) | 8/8 (100.0%) | 7/10 (70.0%) |
| analytic_head_repair | 24/24 (100.0%) | 7/8 (87.5%) | 8/8 (100.0%) | 10/10 (100.0%) |

Hard-test accuracy changed from 0% to 20%.
Backbone-only candidate selected: `backbone_projection_1.0`. A no-op means neither tested backbone intervention met the selection rule with a better outcome.

The main edit is chosen on repair examples, never on hard-test examples. A semantic head rewrite uses extra authored class probes; it is a head replacement, and does not establish that MiniLM learned negation. The ordinary test shares the training shortcut. Both hard splits reverse it. Check per-example predictions for unresolved failures.

The analytic head control is supervised ridge fitting from repair labels, despite requiring no gradients. It is not equivalent to the paper’s reasoning-only intervention. Its target margin is 3 and ridge parameter is 0.01, fixed before final evaluation.

All classification and probability metrics, Wilson accuracy intervals, paired exact McNemar comparisons, candidate scores and tensor-change counts are in results/. With only ten hard-test examples, a single example changes accuracy by ten percentage points. Single seed, synthetic data, and shared hard-case construction patterns limit generalization claims.

References:
- https://rikiyat.github.io/blog/weight-space-programming/
- https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
