"""Replay the already-recorded LLM decisions. Never claims autonomous rediscovery."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import contextlib,importlib,json

def run_replay():
 from . import session
 if not (session.ROOT/'checkpoints/baseline.pt').exists():
  raise FileNotFoundError('First run .venv/bin/python experiment.py from the repository root to generate the baseline checkpoint.')
 run=session.HERE/'runs'/datetime.now(ZoneInfo('Asia/Tehran')).strftime('replay-%Y%m%d-%H%M%S-%f')
 run.mkdir(parents=True)
 old_results,old_patches=session.RESULTS,session.PATCHES
 try:
  session.RESULTS=run;session.PATCHES=run/'patches'
  from . import editor_steps,inspect_embeddings,inspect_neurons,evaluate
  # Their imported path aliases must reflect this isolated replay directory.
  for module in [editor_steps,inspect_embeddings,inspect_neurons,evaluate]:importlib.reload(module)
  with (run/'replay.log').open('w') as log,contextlib.redirect_stdout(log):
   s=session.Session();s.inspect();s.assess('no_edit','Baseline: inspect errors before making any tensor change.')
   editor_steps.step_1(session.Session())
   inspect_embeddings.inspect_embeddings()
   editor_steps.step_2(session.Session())
   inspect_neurons.inspect_neurons()
   editor_steps.step_3(session.Session())
   editor_steps.step_4(session.Session())
   evaluate.finalize_and_evaluate(report_path=run/'RESULTS.md')
 finally:
  session.RESULTS=old_results;session.PATCHES=old_patches
 (run/'replay_notice.json').write_text(json.dumps({'recorded_decision_replay':True,'new_llm_editor_calls':0,'original_results_overwritten':False},indent=2))
 print('Recorded edit sequence replayed in:',run)
 return run
if __name__=='__main__':run_replay()
