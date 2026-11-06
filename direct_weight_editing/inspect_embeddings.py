"""Inspect actual embedding coordinates/norms; do not construct or select edits."""
import json
from .session import Session,RESULTS

def inspect_embeddings():
 s=Session();w=s.baseline['encoder.embeddings.word_embeddings.weight']
 report={token:{'id':s.tokenizer.convert_tokens_to_ids(token),
  'norm':float(w[s.tokenizer.convert_tokens_to_ids(token)].norm()),
  'first_8_coordinates':w[s.tokenizer.convert_tokens_to_ids(token)][:8].tolist()}
  for token in ['[CLS]','[SEP]','billing','password','current','not']}
 (RESULTS/'embedding_weight_inspection.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2));return report
if __name__=='__main__':inspect_embeddings()
