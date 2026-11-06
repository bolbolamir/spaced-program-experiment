from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
nb=nbf.v4.new_notebook()
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md('''# MiniLM: can direct weight edits repair difficult ticket routing?

A 50-example, two-class experiment on Apple Silicon. Classify the **current actionable request** as billing or account access. Archived routing metadata becomes misleading on hard examples; some requests explicitly deny or resolve the competing intent.

This notebook contains the complete experiment code and displays the locally measured results. The default opens saved results without training again. Set `RUN_EXPERIMENT = True` and **Run All** to repeat the fixed-budget MPS experiment.

**Research boundary:** this is a paper-inspired intervention study. The backbone projection follows its direction-removal formula. Semantic head replacement is a separate adaptation; the paper's Experiment 8 disallowed output-layer edits. No external editor LLM or API is called here. Extra semantic probes and repair feedback are supervision. A direct weight edit is not evidence of autonomous raw-weight understanding.''')
code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Markdown, Image
ROOT = Path.cwd()
assert (ROOT / 'experiment.py').exists(), 'Open this notebook from the project folder.'
RUN_EXPERIMENT = False
''')
md('''## 1. Fixed protocol and dataset

Exactly 50 total examples: **24 train + 8 ordinary test + 8 hard repair + 10 hard test**. Hard repair is development data, so success there is not generalization. The final hard test never participates in candidate construction or selection. All examples are synthetic and authored in advance; hard-test text is visible to the project author, so the holdout is procedural rather than blinded.

The training/ordinary-test metadata is always correct; the hard metadata is always reversed. This is an intentionally severe, constructed shortcut shift. It does not estimate production performance.''')
code("display(Markdown((ROOT / 'PROTOCOL.md').read_text()))\ndata = pd.read_json(ROOT / 'data/tickets.json')\ndisplay(pd.crosstab(data['split'], data['label_name']))\ndisplay(data[['id', 'split', 'text', 'label_name']])")
md('''## 2. Reproduce the experiment

The cells below contain the actual source. They execute only when `RUN_EXPERIMENT=True`. Frozen settings: one seed, eight epochs, batch eight, 96 tokens, first five blocks frozen, final block and classifier head trained. Apple MPS is selected automatically; CPU fallback is explicitly recorded.

The final evaluation occurs after candidate selection. Do not change the method in response to the hard-test scores and keep calling that set held out.''')
source=(ROOT/'experiment.py').read_text()
for block in source.split('# %% '):
 if not block.strip():continue
 title,body=block.split('\n',1)
 md('### '+title)
 code('if RUN_EXPERIMENT:\n'+''.join('    '+line+'\n' for line in body.splitlines()))
md('''## 3. Measured environment, runtime, and training curve

Runtime excludes dependency installation and may include initial model download. The saved upstream checkpoint is pinned by revision and dataset bytes are hashed. A single seed is used to keep computation low; repeated-seed uncertainty is not measured.''')
code("metadata = json.loads((ROOT / 'results/run_metadata.json').read_text())\ndisplay(pd.Series(metadata, name='recorded value').to_frame())\ndisplay(pd.read_csv(ROOT / 'results/training_log.csv'))")
md('''## 4. Failure diagnosis and fixed edit candidates

Repair predictions reveal whether the trained classifier follows archived metadata. A nuisance direction `u` is the mean embedding contrast from four neutral requests, holding the request fixed while swapping the archived header.

Backbone candidates edit the **last block's attention and FFN output matrices and biases** using `W_new = W - alpha * u (u.T @ W)` for alpha 0.5 or 1.0. This uses pooled activations as an approximation to a residual-stream direction; it may fail or remove useful semantics. The LayerNorms remain in place, so projection is not guaranteed to remove the final activation component.

Semantic-head candidates construct nearest-centroid head weights from eight authored current-request probes, with optional nuisance projection (0, 0.5 or 1.0). This supplies extra class semantics and changes only the head; it does not show that the encoder acquired negation reasoning.

Select the best repair accuracy among candidates losing no more than 1/24 training accuracy; tie-break by training accuracy, then smallest changed parameter count. The no-edit candidate permits a negative result. All scores are retained. An independent best backbone-only selection is reported, including no-op when appropriate.''')
code("display(pd.read_csv(ROOT / 'results/repair_diagnosis.csv'))\ndisplay(pd.read_csv(ROOT / 'results/edit_candidates.csv'))\ndisplay(pd.read_csv(ROOT / 'results/weight_changes.csv'))\ndisplay(Markdown('**Authored probes (additional supervision):**'))\ndisplay(json.loads((ROOT / 'results/diagnostic_probes.json').read_text()))")
md('''## 5. Accuracy, uncertainty, and classification metrics

Controls: (1) semantic head on the original pretrained encoder, and (2) ordinary gradient repair of only the classifier using cached train+repair embeddings, 100 very small steps. Initial baseline fine-tunes the last block; gradient repair freezes it. A third control performs an analytic ridge head update (target margin 3, ridge 0.01) preserving train logits while fitting repair residuals. Although gradient-free, this is supervised fitting and falls outside the paper’s no-training rules. These are controls with different budgets, not an equal-compute superiority test.

Probability metrics are uncalibrated. ROC-AUC and average precision use account-access as positive. Brier score uses its probability; ECE uses ten fixed confidence bins. Wilson intervals quantify finite example count only. Exact paired McNemar tests are exploratory and unadjusted for multiple comparisons. A ten-example hard test cannot support strong claims.''')
code("metrics = pd.read_csv(ROOT / 'results/metrics.csv')\ndisplay(metrics.pivot(index='method', columns='split', values='accuracy').style.format('{:.0%}'))\ndisplay(Image(filename=str(ROOT / 'results/accuracy_comparison.png')))\ndisplay(metrics[metrics['split']=='hard_test'].T)\ndisplay(metrics)\ndisplay(pd.read_csv(ROOT / 'results/paired_hard_test.csv'))\ndisplay(Image(filename=str(ROOT / 'results/hard_test_confusions.png')))")
md('''## 6. Every held-out prediction and remaining failures

All examples are retained, including regressions. No post-evaluation candidate changes are made.''')
code("predictions = pd.read_csv(ROOT / 'results/predictions.csv')\nhard = predictions[predictions['split']=='hard_test']\ndisplay(hard[['method','id','text','label_name','predicted','p_billing','p_account_access','correct']])\ndisplay(Markdown('**Remaining selected-edit errors:**'))\ndisplay(hard[(hard['method']=='selected_direct_edit') & (hard['correct']==0)])\ndisplay(Markdown((ROOT / 'RESULTS.md').read_text()))")
md('''## 7. Saved artifacts and exact tensor replay

`checkpoints/baseline.pt` stores all baseline weights. `edited_tensors.pt` stores exact replacement tensors; `weight_patch.pt` stores additive diffs and the selected candidate. Prefer exact replacements when reproducing predictions because floating-point subtraction/addition need not round-trip bitwise. `predict_ticket.py` loads these locally without retraining. `results/` contains every metric, prediction, candidate and measured runtime.

Sources: [paper](https://rikiyat.github.io/blog/weight-space-programming/), [released Experiment 8 protocol](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math), [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).''')
code("display(pd.DataFrame([{'file':str(p.relative_to(ROOT)), 'bytes':p.stat().st_size} for folder in ['data','results','checkpoints'] for p in sorted((ROOT/folder).rglob('*')) if p.is_file()]))")
nb.cells=cells
nb.metadata={'kernelspec':{'display_name':'MiniLM research (.venv)','language':'python','name':'minilm-weight-space'},'language_info':{'name':'python','version':'3.12'}}
nbf.write(nb,ROOT/'minilm_weight_space_research.ipynb')
print('Created notebook with',len(cells),'cells')
