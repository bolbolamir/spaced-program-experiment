"""Verify compact patch, head identity, no repair optimizers and all recorded accuracies."""
from pathlib import Path
import json,ast
import numpy as np
import pandas as pd
import torch
from .session import ROOT,HERE,RESULTS,Session,digest,file_digest

def verify():
 metadata=json.loads((RESULTS/'verification.json').read_text())
 selection=json.loads((RESULTS/'final_selection.json').read_text())
 assert file_digest(HERE/'data/final_holdout.json')==selection['holdout_sha256']
 assert file_digest(RESULTS/'candidate_log.json')==selection['candidate_log_sha256']
 assert file_digest(ROOT/'checkpoints/baseline.pt')==selection['baseline_sha256']
 for name in ['session.py','editor_steps.py','inspect_neurons.py','inspect_embeddings.py','evaluate.py','replay.py']:
  tree=ast.parse((HERE/name).read_text())
  forbidden={'backward','grad','lstsq','solve','pinv','Adam','AdamW','SGD','LBFGS'}
  for node in ast.walk(tree):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
    assert node.func.attr not in forbidden,(name,node.func.attr)
 records=pd.read_csv(RESULTS/'predictions.csv');metrics=pd.read_csv(RESULTS/'metrics.csv')
 for r in metrics.itertuples():
  group=records[(records.method==r.method)&(records.split==r.split)]
  assert len(group)==r.n and int(group.correct.sum())==r.correct
  assert np.isclose(group.correct.mean(),r.accuracy)
 s=Session();state={k:v.clone() for k,v in s.baseline.items()}
 with np.load(RESULTS/'selected_patch.npz',allow_pickle=False) as compact:
  idx=0
  while f'name_{idx}' in compact:
   name=str(compact[f'name_{idx}'].item());indices=compact[f'indices_{idx}'];values=compact[f'values_{idx}']
   assert name.startswith('encoder.')
   assert tuple(compact[f'shape_{idx}'])==tuple(state[name].shape)
   state[name].view(-1)[torch.from_numpy(indices)]=torch.from_numpy(values)
   idx+=1
 s.model.load_state_dict(state);s.assert_frozen_head()
 changed={k:state[k] for k in state if not torch.equal(state[k],s.baseline[k])}
 details=pd.read_csv(RESULTS/'weight_changes.csv')
 assert set(details.tensor)==set(changed)
 for r in details.itertuples():
  assert digest(changed[r.tensor])==r.after_sha256
  assert int(torch.count_nonzero(changed[r.tensor]-s.baseline[r.tensor]))==r.changed_scalars
 assert sum(int(torch.count_nonzero(state[k]-s.baseline[k])) for k in changed)==metadata['changed_scalars']
 cases=records[records.method=='llm_encoder_edit']
 actual=s.predict(cases.text.tolist());expected=cases[['p_billing','p_account_access']].to_numpy()
 assert np.allclose(actual,expected,atol=3e-5)
 report={'passed':True,'head_bitwise_unchanged':True,'compact_patch_replays_predictions':True,
  'replayed_predictions':len(cases),'max_probability_error':float(np.max(np.abs(actual-expected))),
  'no_optimizer_backward_or_regression_solve_calls_in_repair_code':True,'all_accuracy_counts_verified':True,
  'dataset_baseline_and_selection_hashes_verified':True,'changed_scalar_count':metadata['changed_scalars']}
 (RESULTS/'artifact_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return report
if __name__=='__main__':verify()
