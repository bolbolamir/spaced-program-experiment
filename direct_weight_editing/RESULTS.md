# Direct encoder editing: measured results

Selected patch: `header_position_neutralization`. Classifier weight and bias are bitwise unchanged.

| Split | Baseline | LLM encoder edit |
|---|---:|---:|
| train | 24/24 (100.0%) | 23/24 (95.8%) |
| hard_repair | 1/8 (12.5%) | 3/8 (37.5%) |
| ordinary_test | 8/8 (100.0%) | 8/8 (100.0%) |
| old_hard_test | 0/10 (0.0%) | 7/10 (70.0%) |
| fresh_hard_test | 1/24 (4.2%) | 15/24 (62.5%) |
| format_stress_test | 4/12 (33.3%) | 8/12 (66.7%) |

Changed 6,912 / 22,713,986 scalar parameters. Four nonbaseline candidates; MPS device recorded as `mps`.

The selected positional intervention replaces rows 1..18 with eight times the pretrained SEP embedding. It attempts to weaken archived-header content through the embedding LayerNorm. It is format-specific weight programming; it does not establish general negation understanding. The varied-format stress split measures some consequences of this restriction. Tokenization, attention masks, pooling and the classifier are unchanged.

The editor was Codex acting interactively in the chat: it inspected diagnostic reports, then authored each next patch. The notebook replays this recorded sequence; it does not make independent editor-model calls or rediscover the edits. No optimizer, gradients, supervised residual fitting or automated numeric search is used during repair.

The new 24-case hard test and 12-case format stress set were frozen before diagnostics and scored only after selection. They were authored by the same agent and share construction patterns, so the holdout is procedural, not independently authored or blinded. The original ordinary and hard tests are previously seen. All final results are retained and no post-evaluation edits are selected.

This is a general weight-space programming adaptation, not a reproduction of the paper’s Experiment 8: it permits positional embedding edits, whereas that experiment restricted embedding edits. The original baseline was already gradient-trained; the no-gradient claim applies only to this repair stage.

Metrics include classification and probability metrics, Wilson intervals, confusion matrices and exploratory unadjusted exact paired McNemar tests. Single baseline seed and small synthetic tests limit conclusions. See `results/` for all predictions, candidates, diagnostics and hashes.

On format stress, the edit fixes seven baseline errors and breaks three previously correct answers (net +4/12). The training set also loses one correct answer. These regressions remain in the prediction records.
