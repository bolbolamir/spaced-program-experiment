"""Verify saved scientific artifacts without rerunning training or changing any selection."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import torch
import nbformat
from transformers import AutoConfig, AutoModel, AutoTokenizer
ROOT=Path(__file__).resolve().parent
meta=json.loads((ROOT/'results/run_metadata.json').read_text())
data=pd.read_json(ROOT/'data/tickets.json'); pred=pd.read_csv(ROOT/'results/predictions.csv'); metrics=pd.read_csv(ROOT/'results/metrics.csv')
assert len(data)==50 and not data.id.duplicated().any()
assert meta['dataset_sha256']==hashlib.sha256((ROOT/'data/tickets.json').read_bytes()).hexdigest()
for row in metrics.itertuples():
 group=pred[(pred.method==row.method)&(pred.split==row.split)]
 assert len(group)==row.n and int(group.correct.sum())==row.correct
 assert np.isclose(group.correct.mean(),row.accuracy)
 assert np.allclose(group[['p_billing','p_account_access']].sum(axis=1),1,atol=1e-6)
 assert set(group.id)==set(data.loc[data.split==row.split,'id'])
base=torch.load(ROOT/'checkpoints/baseline.pt',weights_only=True,map_location='cpu')
exact=torch.load(ROOT/'checkpoints/edited_tensors.pt',weights_only=True,map_location='cpu')
patch=torch.load(ROOT/'checkpoints/weight_patch.pt',weights_only=True,map_location='cpu')
assert set(exact)==set(patch['delta'])
assert sum(int(torch.count_nonzero(exact[k]-base[k])) for k in exact)==meta['changed_scalars']
for k in exact: assert torch.equal(exact[k]-base[k],patch['delta'][k])
# Offline reconstruction must reproduce selected-edit probabilities for all ten hard-test cases.
state=dict(base); state.update(exact)
encoder=AutoModel.from_config(AutoConfig.from_pretrained(ROOT/'checkpoints'),attn_implementation='eager')
encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')})
device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); encoder.to(device).eval()
head=torch.nn.Linear(384,2).to(device)
head.load_state_dict({k.removeprefix('head.'):v for k,v in state.items() if k.startswith('head.')})
tokenizer=AutoTokenizer.from_pretrained(ROOT/'checkpoints/tokenizer')
hard=data[data.split=='hard_test']
with torch.no_grad():
 outs=[]
 for start in range(0,len(hard),8):
  tokens={k:v.to(device) for k,v in tokenizer(hard.text.tolist()[start:start+8],padding=True,truncation=True,max_length=96,return_tensors='pt').items()}
  h=encoder(**tokens).last_hidden_state; mask=tokens['attention_mask'].unsqueeze(-1)
  features=torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),dim=-1)
  outs.append(head(features).softmax(-1).cpu().numpy())
actual=np.concatenate(outs)
expected=pred[(pred.method=='selected_direct_edit')&(pred.split=='hard_test')][['p_billing','p_account_access']].to_numpy()
assert np.allclose(actual,expected,atol=3e-5),np.max(np.abs(actual-expected))
nb=nbformat.read(ROOT/'minilm_weight_space_research.ipynb',as_version=4); nbformat.validate(nb)
assert not [out for cell in nb.cells for out in cell.get('outputs',[]) if out.output_type=='error']
report={'passed':True,'checks':['dataset size and unique IDs','dataset SHA256','all metric accuracies recomputed from predictions','all method/split memberships','probability normalization','exact saved tensor deltas and parameter count','offline checkpoint replay matches all 10 selected-edit hard-test probabilities','valid executed notebook with no cell errors'],'device':str(device),'max_replay_probability_error':float(np.max(np.abs(actual-expected)))}
(ROOT/'results/verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
