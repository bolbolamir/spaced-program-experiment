"""Constrained, inference-only harness for an interactive Codex weight editor."""
from pathlib import Path
import json, hashlib, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT=Path(__file__).resolve().parent.parent
HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
PATCHES=ROOT/'checkpoints/direct_encoder'
DEVICE=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
BATCH=8; MAX_LENGTH=96

def digest(tensor):return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def file_digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def state_digest(state):
 h=hashlib.sha256()
 for key,tensor in sorted(state.items()):
  h.update(key.encode());h.update(tensor.contiguous().numpy().tobytes())
 return h.hexdigest()

def wilson(correct,n):
 z=1.959963984540054;p=correct/n;den=1+z*z/n
 a=(p+z*z/(2*n))/den;b=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return a-b,a+b

class Model(nn.Module):
 def __init__(self):
  super().__init__()
  self.encoder=AutoModel.from_config(AutoConfig.from_pretrained(ROOT/'checkpoints'),attn_implementation='eager')
  self.head=nn.Linear(384,2)
 def forward(self,tokens):
  h=self.encoder(**tokens).last_hidden_state
  mask=tokens['attention_mask'].unsqueeze(-1)
  feature=nn.functional.normalize((h*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1)
  return self.head(feature)

class Session:
 def __init__(self):
  self.started=time.perf_counter()
  torch.set_grad_enabled(False);torch.set_num_threads(4)
  RESULTS.mkdir(exist_ok=True);PATCHES.mkdir(parents=True,exist_ok=True)
  self.baseline=torch.load(ROOT/'checkpoints/baseline.pt',weights_only=True,map_location='cpu')
  self.model=Model().to(DEVICE);self.model.load_state_dict(self.baseline);self.model.eval()
  for p in self.model.parameters():p.requires_grad_(False)
  self.tokenizer=AutoTokenizer.from_pretrained(ROOT/'checkpoints/tokenizer')
  self.data=pd.read_json(ROOT/'data/tickets.json')
  self.train=self.data[self.data.split=='train'].copy()
  self.repair=self.data[self.data.split=='hard_repair'].copy()
  # Current-request-only probes are variants of repair examples, never of final holdout examples.
  self.request_only=self.repair.copy();self.request_only['text']=self.request_only['request'];self.request_only['split']='repair_request_only'
  self.dev=pd.concat([self.train,self.repair,self.request_only],ignore_index=True)
  self.head_hash={k:digest(v) for k,v in self.baseline.items() if k.startswith('head.')}
 def tokenize(self,texts):
  return {k:v.to(DEVICE) for k,v in self.tokenizer(list(texts),padding=True,truncation=True,max_length=MAX_LENGTH,return_tensors='pt').items()}
 def predict(self,texts):
  self.model.eval()
  with torch.inference_mode():
   return np.concatenate([self.model(self.tokenize(texts[i:i+BATCH])).softmax(-1).cpu().numpy() for i in range(0,len(texts),BATCH)])
 def load_patch(self,name):
  self.model.load_state_dict(self.baseline)
  if name!='no_edit':
   patch=torch.load(PATCHES/f'{name}.pt',weights_only=True,map_location='cpu')
   assert all(k.startswith('encoder.') for k in patch)
   self.model.load_state_dict(patch,strict=False)
  self.assert_frozen_head()
 def assert_frozen_head(self):
  state=self.model.state_dict()
  assert all(digest(state[k])==v for k,v in self.head_hash.items())
  assert not any(p.requires_grad for p in self.model.parameters())
  assert not torch.is_grad_enabled()
 def assess(self,name,explanation):
  self.assert_frozen_head()
  probs=self.predict(self.dev.text.tolist())
  prediction=self.dev[['id','split','label','text']].copy()
  prediction['predicted']=probs.argmax(1);prediction['p_billing']=probs[:,0];prediction['p_account_access']=probs[:,1]
  prediction['correct']=prediction.predicted==prediction.label
  prediction.to_csv(RESULTS/f'dev_{name}.csv',index=False)
  patch={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items() if not torch.equal(v.detach().cpu(),self.baseline[k])}
  assert all(k.startswith('encoder.') for k in patch)
  if name!='no_edit':torch.save(patch,PATCHES/f'{name}.pt')
  row={'candidate':name,'explanation':explanation,'device':str(DEVICE),'head_unchanged':True,
   'changed_scalars':sum(int(torch.count_nonzero(v-self.baseline[k])) for k,v in patch.items()),
   'changed_tensors':list(patch),'tensor_hashes':{k:digest(v) for k,v in patch.items()}}
  for split,part in prediction.groupby('split'):
   row[split+'_correct']=int(part.correct.sum());row[split+'_n']=len(part);row[split+'_accuracy']=float(part.correct.mean())
  row['eligible']=row['train_correct']>=23
  row['seconds_including_load']=time.perf_counter()-self.started
  path=RESULTS/'candidate_log.json';history=json.loads(path.read_text()) if path.exists() else []
  assert name not in [r['candidate'] for r in history], 'Use a new candidate name; never overwrite feedback.'
  assert len(history)<5,'Budget exhausted: at most baseline + four patches.'
  history.append(row);path.write_text(json.dumps(history,indent=2))
  print(json.dumps({k:v for k,v in row.items() if k not in ['tensor_hashes']},indent=2),flush=True)
  return row
 def inspect(self):
  probs=self.predict(self.repair.text.tolist());only=self.predict(self.request_only.text.tolist())
  report={'device':str(DEVICE),'baseline_sha256':file_digest(ROOT/'checkpoints/baseline.pt'),
   'head_hashes':self.head_hash,'baseline_repair_correct':int((probs.argmax(1)==self.repair.label.values).sum()),
   'request_only_repair_correct':int((only.argmax(1)==self.repair.label.values).sum()),'repair_n':8,'examples':[]}
  for idx,row in enumerate(self.repair.itertuples()):
   report['examples'].append({'id':row.id,'text':row.text,'label':int(row.label),'baseline_prediction':int(probs[idx].argmax()),
   'request_only_prediction':int(only[idx].argmax()),'rationale':'Route the unresolved current '+('billing request; ignore resolved/denied access issues and the archived access department.' if row.label==0 else 'account-access request; ignore resolved/denied billing issues and the archived billing department.')})
  # Inspect token geometry for one header of each class, not final-test examples.
  head_rows=[]
  for row in self.repair.iloc[:2].itertuples():
   text=row.text;marker=text.index(' Customer current request:');active_start=text.index(row.request)
   enc=self.tokenizer(text,return_offsets_mapping=True,return_tensors='pt');offsets=enc.pop('offset_mapping')[0].tolist()
   header=[i for i,(a,b) in enumerate(offsets) if b>0 and b<=marker]
   active=[i for i,(a,b) in enumerate(offsets) if b>active_start]
   tokens=self.tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
   with torch.inference_mode():out=self.model.encoder(**{k:v.to(DEVICE) for k,v in enc.items()},output_attentions=True)
   report.setdefault('token_layout',[]).append({'id':row.id,'header_positions':header,'active_start_position':min(active),'tokens':list(enumerate(tokens))})
   for layer,attention in enumerate(out.attentions):
    for head in range(attention.shape[1]):
     a=attention[0,head].cpu()
     head_rows.append({'example':row.id,'layer':layer,'head':head,'request_attention_to_header':float(a[active][:,header].sum(1).mean()),'request_attention_to_request':float(a[active][:,active].sum(1).mean())})
  heads=pd.DataFrame(head_rows);heads.to_csv(RESULTS/'attention_diagnostic.csv',index=False)
  report['attention_summary']=heads.groupby(['layer','head'])[['request_attention_to_header','request_attention_to_request']].mean().sort_values('request_attention_to_header',ascending=False).head(10).reset_index().to_dict('records')
  pos=self.baseline['encoder.embeddings.position_embeddings.weight']
  report['position_weight_stats']={'shape':list(pos.shape),'median_row_norm':float(pos.norm(dim=1).median()),'first_24_row_norms':pos[:24].norm(dim=1).tolist()}
  ff=self.baseline['encoder.encoder.layer.5.output.dense.weight'];direction=self.baseline['head.weight'][0]-self.baseline['head.weight'][1]
  values,indices=torch.topk(torch.abs(direction@ff),8)
  report['last_ffn_top_head_aligned_columns']=[{'column':int(i),'abs_alignment':float(v)} for i,v in zip(indices,values)]
  (RESULTS/'inspection.json').write_text(json.dumps(report,indent=2))
  print(json.dumps(report,indent=2))
  return report
