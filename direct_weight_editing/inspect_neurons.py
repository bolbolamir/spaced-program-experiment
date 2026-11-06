"""Diagnostic only: last-FFN activations on train/repair, no tests and no edits."""
import json
import pandas as pd
import torch
from .session import Session,RESULTS

def inspect_neurons():
 s=Session();cache={}
 def hook(module,args,out):cache['activations']=out.detach().cpu()
 handle=s.model.encoder.encoder.layer[5].intermediate.register_forward_hook(hook)
 direction=s.baseline['head.weight'][0]-s.baseline['head.weight'][1]
 writes=direction@s.baseline['encoder.encoder.layer.5.output.dense.weight']
 cols=[744,143,77,265,507,475,729,1478]
 evidence=[
 'I only want the extra payment refunded.','I only need my password reset.',
 'the unresolved issue is a duplicate debit that must be reversed.',
 'the unresolved issue is that I cannot sign in.',
 'Please cancel the paid plan and stop charging me.',
 'Please unlock my account so I can use it.',
 'The charge for a cancelled order still needs reimbursement.',
 'The account recovery code still does not arrive.']
 records=[]
 for row,span in zip(s.repair.itertuples(),evidence):
  assert span in row.text
  enc=s.tokenizer(row.text,return_offsets_mapping=True,return_tensors='pt');offset=enc.pop('offset_mapping')[0].tolist()
  split=row.text.index(' Customer current request:');a=row.text.index(span);b=a+len(span)
  header=[i for i,(x,y) in enumerate(offset) if y>0 and y<=split]
  active=[i for i,(x,y) in enumerate(offset) if y>a and x<b]
  _=s.model({k:v.to(next(s.model.parameters()).device) for k,v in enc.items()})
  h=cache['activations'][0]
  for col in cols:
   records.append({'example':row.id,'label':int(row.label),'column':col,'signed_class0_write':float(writes[col]),
    'header_activation':float(h[header,col].mean()),'active_evidence_activation':float(h[active,col].mean())})
 handle.remove()
 frame=pd.DataFrame(records);frame.to_csv(RESULTS/'ffn_diagnostic.csv',index=False)
 print(frame.groupby(['column','label'])[['signed_class0_write','header_activation','active_evidence_activation']].mean().round(4).to_string())
 return frame
if __name__=='__main__':inspect_neurons()
