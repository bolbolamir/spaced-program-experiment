"""Freeze the development-only choice, then evaluate baseline and selected encoder patch."""
import json,time,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_recall_fscore_support,matthews_corrcoef,roc_auc_score,average_precision_score,log_loss,confusion_matrix
from scipy.stats import binomtest
from .session import Session,ROOT,HERE,RESULTS,PATCHES,DEVICE,digest,file_digest,wilson

def compute_metrics(y,p):
 pred=p.argmax(1);n=len(y);correct=int((pred==y).sum());lo,hi=wilson(correct,n)
 result={'n':n,'correct':correct,'accuracy':correct/n,'accuracy_ci95_low':lo,'accuracy_ci95_high':hi,
  'balanced_accuracy':balanced_accuracy_score(y,pred),'mcc':matthews_corrcoef(y,pred),
  'roc_auc':roc_auc_score(y,p[:,1]),'average_precision':average_precision_score(y,p[:,1]),
  'log_loss':log_loss(y,p,labels=[0,1]),'brier_score':float(np.mean((p[:,1]-y)**2))}
 for avg in ['macro','weighted']:
  pr,re,f,_=precision_recall_fscore_support(y,pred,average=avg,zero_division=0)
  result.update({f'precision_{avg}':pr,f'recall_{avg}':re,f'f1_{avg}':f})
 pr,re,f,support=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
 for i,label in enumerate(['billing','account_access']):
  result.update({f'precision_{label}':pr[i],f'recall_{label}':re[i],f'f1_{label}':f[i],f'support_{label}':int(support[i])})
 conf=p.max(1);ok=pred==y;ece=0.
 for a,b in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
  mask=(conf>=a)&((conf<b) if b<1 else (conf<=b))
  if mask.any():ece+=mask.mean()*abs(ok[mask].mean()-conf[mask].mean())
 result['ece_10_bins']=ece
 return result

def finalize_and_evaluate(report_path=None):
 started=time.perf_counter();s=Session()
 history=json.loads((RESULTS/'candidate_log.json').read_text())
 assert len(history)==5,'Exactly baseline and four interactive candidate outcomes must be retained.'
 eligible=[(idx,r) for idx,r in enumerate(history) if r['eligible']]
 _,selected=min(eligible,key=lambda ir:(-ir[1]['hard_repair_accuracy'],-ir[1]['train_accuracy'],ir[1]['changed_scalars'],ir[0]))
 selection_path=RESULTS/'final_selection.json'
 selection={'selected_candidate':selected['candidate'],'selection_rule':'repair accuracy, train accuracy, fewest changed scalars, earliest; train >=23/24',
  'selected_record':selected,'candidate_log_sha256':file_digest(RESULTS/'candidate_log.json'),
  'baseline_sha256':file_digest(ROOT/'checkpoints/baseline.pt'),
  'holdout_sha256':file_digest(HERE/'data/final_holdout.json'),
  'selected_at_tehran':datetime.now(ZoneInfo('Asia/Tehran')).isoformat(),
  'editor':'Codex in this interactive chat; no separate editor API; saved replay is not an autonomous editor',
  'final_test_feedback_used_for_selection':False,'head_frozen':True,'gradients_used_during_repair':False}
 if selection_path.exists():
  previous=json.loads(selection_path.read_text())
  assert previous['selected_candidate']==selection['selected_candidate']
  assert previous['candidate_log_sha256']==selection['candidate_log_sha256']
 else:selection_path.write_text(json.dumps(selection,indent=2))
 # Selection is on disk BEFORE opening held-out data for evaluation.
 holdout=pd.read_json(HERE/'data/final_holdout.json')
 assert len(holdout)==36
 original=s.data[['id','split','label','text']].copy()
 original['split']=original['split'].replace({'test':'ordinary_test','hard_test':'old_hard_test'})
 final=pd.concat([original,holdout],ignore_index=True)
 assert not final.text.duplicated().any()
 records=[];predictions=[];matrices={};all_probs={}
 for method,patchname in [('baseline','no_edit'),('llm_encoder_edit',selected['candidate'])]:
  s.load_patch(patchname);p=s.predict(final.text.tolist());all_probs[method]=p
  for split,group in final.groupby('split',sort=False):
   ids=group.index.to_numpy();truth=group.label.to_numpy()
   records.append({'method':method,'split':split,**compute_metrics(truth,p[ids])})
   matrices[method+'/'+split]=confusion_matrix(truth,p[ids].argmax(1),labels=[0,1]).tolist()
  for idx,row in final.iterrows():
   predictions.append({'method':method,**row.to_dict(),'predicted':int(p[idx].argmax()),'p_billing':float(p[idx,0]),'p_account_access':float(p[idx,1]),'correct':bool(p[idx].argmax()==row.label)})
 metrics=pd.DataFrame(records);metrics.to_csv(RESULTS/'metrics.csv',index=False)
 pd.DataFrame(predictions).to_csv(RESULTS/'predictions.csv',index=False)
 (RESULTS/'confusion_matrices.json').write_text(json.dumps(matrices,indent=2))
 comparisons=[]
 for split in ['ordinary_test','old_hard_test','fresh_hard_test','format_stress_test']:
  ids=final.index[final.split==split].to_numpy();truth=final.loc[ids,'label'].values
  before=all_probs['baseline'][ids].argmax(1)==truth;after=all_probs['llm_encoder_edit'][ids].argmax(1)==truth
  fixed=int((~before&after).sum());broken=int((before&~after).sum())
  comparisons.append({'split':split,'fixed':fixed,'broken':broken,'both_correct':int((before&after).sum()),'both_wrong':int((~before&~after).sum()),'exact_mcnemar_p':binomtest(fixed,fixed+broken,.5).pvalue if fixed+broken else 1.})
 pd.DataFrame(comparisons).to_csv(RESULTS/'paired_comparisons.csv',index=False)
 s.load_patch(selected['candidate']);state=s.model.state_dict()
 tensor_records=[];patch={}
 for k,before in s.baseline.items():
  after=state[k].cpu()
  if not torch.equal(before,after):
   assert k.startswith('encoder.')
   patch[k]=after.clone();delta=after-before
   tensor_records.append({'tensor':k,'changed_scalars':int(torch.count_nonzero(delta)),'before_sha256':digest(before),'after_sha256':digest(after),'delta_frobenius':float(delta.norm())})
 torch.save(patch,PATCHES/'selected_exact_tensors.pt')
 pd.DataFrame(tensor_records).to_csv(RESULTS/'weight_changes.csv',index=False)
 # A compact, non-pickle patch can be tracked even though full checkpoints are ignored.
 compact={}
 for idx,(k,value) in enumerate(patch.items()):
  changed=(value!=s.baseline[k]).flatten().nonzero().flatten().numpy()
  compact[f'indices_{idx}']=changed;compact[f'values_{idx}']=value.flatten()[changed].numpy()
  compact[f'name_{idx}']=np.array(k);compact[f'shape_{idx}']=np.array(value.shape)
 np.savez_compressed(RESULTS/'selected_patch.npz',**compact)
 s.assert_frozen_head()
 verification={'head_unchanged':True,'head_hashes':s.head_hash,'all_parameters_requires_grad_false':not any(p.requires_grad for p in s.model.parameters()),
  'gradient_recording_disabled':not torch.is_grad_enabled(),'only_encoder_tensors_changed':all(k.startswith('encoder.') for k in patch),
  'changed_scalars':sum(r['changed_scalars'] for r in tensor_records),'total_parameters':sum(p.numel() for p in s.model.parameters()),
  'recorded_candidate_runs':len(history),'candidate_seconds_including_load_total':sum(r['seconds_including_load'] for r in history),
  'evaluation_seconds_including_load':time.perf_counter()-started,'device':str(DEVICE),'selection_precedes_final_evaluation':True,
  'baseline_sha256':selection['baseline_sha256'],'holdout_sha256':selection['holdout_sha256']}
 (RESULTS/'verification.json').write_text(json.dumps(verification,indent=2))
 # Ordinary metrics and all adverse results remain visible.
 splits=['train','hard_repair','ordinary_test','old_hard_test','fresh_hard_test','format_stress_test']
 plt.rcParams.update({'font.family':'DejaVu Sans'})
 fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(splits));width=.34
 for off,method,color in [(-width/2,'baseline','#C48B78'),(width/2,'llm_encoder_edit','#3D857A')]:
  part=metrics[metrics.method==method].set_index('split').loc[splits]
  bars=ax.bar(x+off,part.accuracy,width,label=method.replace('_',' '),color=color)
  ax.bar_label(bars,labels=[f'{int(r.correct)}/{int(r.n)}' for r in part.itertuples()],padding=3,fontsize=9)
 ax.set_xticks(x,[v.replace('_','\n') for v in splits],fontsize=9);ax.set_ylim(0,1.12);ax.set_ylabel('Accuracy');ax.legend(loc='upper right',fontsize=9)
 ax.set_title('Direct LLM-chosen encoder edit · classifier bitwise frozen');ax.spines[['top','right']].set_visible(False)
 fig.tight_layout();fig.savefig(RESULTS/'accuracy_comparison.png',dpi=180);plt.close(fig)
 lines=['# Direct encoder editing: measured results','',f"Selected patch: `{selected['candidate']}`. Classifier weight and bias are bitwise unchanged.",'',
 '| Split | Baseline | LLM encoder edit |','|---|---:|---:|']
 for split in splits:
  part=metrics[metrics.split==split].set_index('method')
  cells=[f"{int(part.loc[m,'correct'])}/{int(part.loc[m,'n'])} ({part.loc[m,'accuracy']:.1%})" for m in ['baseline','llm_encoder_edit']]
  lines.append('| '+split+' | '+' | '.join(cells)+' |')
 lines+=['',f"Changed {verification['changed_scalars']:,} / {verification['total_parameters']:,} scalar parameters. Four nonbaseline candidates; MPS device recorded as `{DEVICE}`.",'',
 ('The selected positional intervention replaces rows 1..18 with eight times the pretrained SEP embedding. It attempts to weaken archived-header content through the embedding LayerNorm. It is format-specific weight programming; it does not establish general negation understanding. The varied-format stress split measures some consequences of this restriction. Tokenization, attention masks, pooling and the classifier are unchanged.' if selected['candidate'] in ['header_position_neutralization','position_plus_early_head_off'] else 'This replay selected another recorded candidate. Consult candidate_log.json for its exact intervention and compare all regression metrics. Architecture, tokenization, pooling and classifier are unchanged.'),
 '', 'The editor was Codex acting interactively in the chat: it inspected diagnostic reports, then authored each next patch. The notebook replays this recorded sequence; it does not make independent editor-model calls or rediscover the edits. No optimizer, gradients, supervised residual fitting or automated numeric search is used during repair.',
 '', 'The new 24-case hard test and 12-case format stress set were frozen before diagnostics and scored only after selection. They were authored by the same agent and share construction patterns, so the holdout is procedural, not independently authored or blinded. The original ordinary and hard tests are previously seen. All final results are retained and no post-evaluation edits are selected.',
 '', 'This is a general weight-space programming adaptation, not a reproduction of the paper’s Experiment 8: it permits positional embedding edits, whereas that experiment restricted embedding edits. The original baseline was already gradient-trained; the no-gradient claim applies only to this repair stage.',
 '', 'Metrics include classification and probability metrics, Wilson intervals, confusion matrices and exploratory unadjusted exact paired McNemar tests. Single baseline seed and small synthetic tests limit conclusions. See `results/` for all predictions, candidates, diagnostics and hashes.']
 stress=next(r for r in comparisons if r['split']=='format_stress_test')
 lines+=['',f"On format stress, the edit fixes {stress['fixed']} baseline errors and breaks {stress['broken']} previously correct answers. All regressions are retained in the prediction records."]
 (report_path or HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 print(metrics.pivot(index='method',columns='split',values='accuracy').to_string())
 print(json.dumps(verification,indent=2))
 return metrics
if __name__=='__main__':finalize_and_evaluate()
