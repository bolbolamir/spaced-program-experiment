from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parent.parent
HERE=ROOT/'direct_weight_editing'
nb=nbf.v4.new_notebook();cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md('''# Direct encoder weight editing with a frozen classifier

An additional experiment in tool-assisted weight-space programming. Codex acted as the editor in the chat: it inspected failures, rationales and internal diagnostics, then chose and wrote each of four patches after seeing preceding feedback. The classifier head stayed bitwise unchanged. No optimizer, gradients, regression solve or automatic parameter search was used during repair.

**This notebook replays the recorded decisions.** It does not call an editor LLM or claim autonomous rediscovery. The original model was already fine-tuned with gradients; the no-gradient restriction applies to this repair stage.

Measured result: **fresh hard-test accuracy 1/24 → 15/24**, ordinary test **8/8 → 8/8**, train **24/24 → 23/24**. The selected edit changes **6,912 positional-embedding values** and depends on the original header layout. This is not evidence of general negation reasoning, and positional-embedding edits differ from the paper's Experiment 8 restrictions.''')
code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Markdown, Image, Code
ROOT = Path.cwd()
assert (ROOT / 'direct_weight_editing/session.py').exists(), 'Launch from the repository root.'
HERE = ROOT / 'direct_weight_editing'
REPLAY_RECORDED_EDITS = False
REPORT = HERE / 'results'
''')
md('''## 1. Protocol and feedback boundary

Only the 24 original train examples and eight hard-repair examples (plus their request-only variants) supplied model feedback during editing. The new 24-case hard set and 12-case varied-format stress set were frozen before diagnostics and first evaluated after selection. They were authored by the same agent, so this is a procedural holdout rather than independent or blinded evaluation. The original tests were previously seen.

A maximum of four nonbaseline patches was allowed. Accept candidates retaining at least 23/24 train examples. Select by repair accuracy, train accuracy, then fewer changed values and earlier candidate. All candidates, including failed interventions, are retained.''')
code("display(Markdown((HERE / 'PROTOCOL.md').read_text()))\noriginal = pd.read_json(ROOT / 'data/tickets.json')\ndisplay(original[original['split']=='hard_repair'][['id','request','label_name']])")
md('''## 2. Replay without altering the published result files

The default is to view the recorded run. Set `REPLAY_RECORDED_EDITS=True` and Run All to replay the saved sequence using your existing baseline. This makes no new LLM calls. Each replay writes an isolated directory under `direct_weight_editing/runs/`, leaving the original notebook/results intact.

If `checkpoints/baseline.pt` is missing, first run `.venv/bin/python experiment.py` at the repository root. This initial baseline generation uses training; the subsequent encoder-repair replay does not. Different baseline weights or kernels can change measured results.''')
code('''if REPLAY_RECORDED_EDITS:
    from direct_weight_editing.replay import run_replay
    REPORT = run_replay()
print('Displaying reports from:', REPORT.relative_to(ROOT))
''')
md('''## 3. What the LLM inspected

The initial probe showed only **1/8** correct full-context repair cases and **3/8** request-only variants. Metadata is one problem; competing resolved and active intents are another.

Attention mass is a diagnostic clue rather than proof of causal importance. Inspecting high-attention heads led to a causal ablation test. Token positions also exposed a stable archived-header region, supporting a separate format-specific weight program.''')
code("inspection = json.loads((REPORT / 'inspection.json').read_text())\ndisplay(pd.DataFrame(inspection['examples']))\ndisplay(pd.DataFrame(inspection['attention_summary']))\ndisplay(pd.DataFrame([{'id':r['id'],'header_positions':r['header_positions'],'active_request_starts':r['active_start_position']} for r in inspection['token_layout']]))\ndisplay(pd.DataFrame(inspection['last_ffn_top_head_aligned_columns']))")
code("display(json.loads((REPORT / 'embedding_weight_inspection.json').read_text()))\nffn = pd.read_csv(REPORT / 'ffn_diagnostic.csv')\ndisplay(ffn.groupby(['column','label'])[['signed_class0_write','header_activation','active_evidence_activation']].mean())")
md('''## 4. The four recorded edits and their feedback

1. Disable the contribution of last-block attention head 7. Repair did not improve.
2. Replace positional rows 1..18 with `8 × pretrained SEP embedding`. This intends to attenuate header-token identity through embedding LayerNorm. Repair improves, but the edit relies on the header's location.
3. Disable last-FFN output column 77 after inspecting its activation/write statistics. Repair did not improve. Statistics alone do not prove a semantic feature.
4. Combine the positional edit with disabling early head 0.6. It tied the positional candidate on development counts while changing more values, so the smaller edit wins.

Every patch is a direct assignment to an encoder tensor. No classifier parameter is assigned. The fixed model architecture, tokenization, attention mask and mean pooling are unchanged.''')
code("history = json.loads((REPORT / 'candidate_log.json').read_text())\ncolumns = ['candidate','train_correct','train_n','hard_repair_correct','hard_repair_n','repair_request_only_correct','changed_scalars','eligible','head_unchanged']\ndisplay(pd.DataFrame(history)[columns])\nfor record in history:\n    display(Markdown('**' + record['candidate'] + '**: ' + record['explanation']))")
md('''### Exact tensor-edit code

These functions were authored sequentially during the interactive editing session. The saved replay runs them in the same order. Constants are recorded choices, not a parameter sweep.''')
code("display(Code((HERE / 'editor_steps.py').read_text(), language='python'))")
md('''### Constrained diagnostic and patch harness

The harness disables gradient recording and freezes every parameter. The exact classifier weight and bias hashes are checked on every assessment. It only evaluates development data during editing and rejects any saved patch containing a non-encoder tensor.''')
code("display(Code((HERE / 'session.py').read_text(), language='python'))")
md('''## 5. Selection fixed before final evaluation

The saved selection binds the development log, baseline and holdout hashes. No final-test prediction is used in the selection rule. All final results below are retained without further edits.''')
code("selection = json.loads((REPORT / 'final_selection.json').read_text())\ndisplay(selection)\ndisplay(pd.read_csv(REPORT / 'weight_changes.csv'))")
md('''## 6. Final accuracy and all classification metrics

The new hard set follows the original header schema. A separate format-stress set varies header length/order and includes requests without headers. Both are small synthetic sets, written by the same agent, with shared construction patterns.

Wilson intervals describe finite example count only. They do not account for shared templates, seed variability or design choices. Probability metrics are uncalibrated. Exact paired McNemar results are exploratory and unadjusted for multiple comparisons.''')
code("metrics = pd.read_csv(REPORT / 'metrics.csv')\norder = ['train','hard_repair','ordinary_test','old_hard_test','fresh_hard_test','format_stress_test']\ndisplay(metrics.pivot(index='method',columns='split',values='accuracy')[order].style.format('{:.1%}'))\ndisplay(Image(filename=str(REPORT / 'accuracy_comparison.png')))\ndisplay(metrics[metrics['split'].isin(['fresh_hard_test','format_stress_test'])].T)\ndisplay(metrics)\ndisplay(pd.read_csv(REPORT / 'paired_comparisons.csv'))")
md('''## 7. Per-example outcomes, including regressions

A newly correct answer and a newly broken answer both matter. Inspect both final test groups rather than treating 15/24 as a general capability claim.''')
code("predictions = pd.read_csv(REPORT / 'predictions.csv')\ndisplay(predictions[predictions['split'].isin(['fresh_hard_test','format_stress_test'])][['method','id','split','text','label','predicted','p_billing','p_account_access','correct']])\ndisplay(Markdown('**Remaining selected-edit errors on the fresh hard set:**'))\ndisplay(predictions[(predictions['method']=='llm_encoder_edit') & (predictions['split']=='fresh_hard_test') & (~predictions['correct'])])")
md('''## 8. Frozen-head and exact patch verification

The original run's verifier reconstructs the model from the compact non-pickle patch and checks all 86 selected-edit probabilities against the saved predictions. It also verifies tensor hashes, changed-scalar count, all accuracy counts, and absence of optimizer/backward/regression-solve calls in repair code. Full checkpoints remain local and ignored; the compact patch is versioned.''')
code("display(json.loads((REPORT / 'verification.json').read_text()))\nif (REPORT / 'artifact_checks.json').exists():\n    display(json.loads((REPORT / 'artifact_checks.json').read_text()))\nelse:\n    print('See verification.json for this replay; full original-run checks are in the published results directory.')")
md('''## 9. Interpretation and measured report

This provides a concrete example of an LLM choosing a useful, direct encoder tensor edit with a frozen classifier. The mechanism is a positional intervention that weakens the archived header, not a demonstrated general reasoning circuit. Several intuitively motivated head/neuron edits failed. Training accuracy regressed by one example.

The original head-fitting control and this frozen-head experiment answer different questions and should not be conflated. A larger, independently authored benchmark and repeated baselines are needed to establish broader gains.''')
code("report_path = REPORT / 'RESULTS.md' if (REPORT / 'RESULTS.md').exists() else HERE / 'RESULTS.md'\ndisplay(Markdown(report_path.read_text()))")
md('''Reference: [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) and [released Experiment 8 protocol](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math). This follow-up uses the general direct-edit idea, while permitting positional-embedding edits that differ from that experiment's tensor restrictions.''')
nb.cells=cells;nb.metadata={'kernelspec':{'display_name':'MiniLM research (.venv)','language':'python','name':'minilm-weight-space'},'language_info':{'name':'python','version':'3.12'}}
nbf.write(nb,ROOT/'direct_encoder_weight_editing.ipynb');print('Built extra notebook:',len(cells),'cells')
if __name__=='__main__':pass
