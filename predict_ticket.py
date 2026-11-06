"""Offline inference: .venv/bin/python predict_ticket.py 'Your ticket' [--baseline]."""
from pathlib import Path
import argparse
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(); parser.add_argument('text'); parser.add_argument('--baseline',action='store_true'); parser.add_argument('--method', choices=['direct','baseline','gradient','analytic'], default='direct'); args=parser.parse_args()
method='baseline' if args.baseline else args.method
device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
state=torch.load(ROOT/'checkpoints/baseline.pt',map_location='cpu',weights_only=True)
if method!='baseline':
 filename={'direct':'edited_tensors.pt','gradient':'gradient_control_head.pt','analytic':'analytic_control_head.pt'}[method]
 state.update(torch.load(ROOT/'checkpoints'/filename,map_location='cpu',weights_only=True))
encoder=AutoModel.from_config(AutoConfig.from_pretrained(ROOT/'checkpoints'),attn_implementation='eager')
encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')})
encoder.to(device).eval(); tokenizer=AutoTokenizer.from_pretrained(ROOT/'checkpoints/tokenizer')
head=torch.nn.Linear(384,2).to(device)
head.load_state_dict({k.removeprefix('head.'):v for k,v in state.items() if k.startswith('head.')})
tokens={k:v.to(device) for k,v in tokenizer(args.text,return_tensors='pt',truncation=True,max_length=96).items()}
with torch.no_grad():
 h=encoder(**tokens).last_hidden_state; mask=tokens['attention_mask'].unsqueeze(-1)
 pooled=torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),dim=-1)
 p=head(pooled).softmax(-1)[0].cpu().tolist()
print({'device':str(device),'variant':method,'label':['billing','account_access'][int(p[1]>p[0])],'probabilities':dict(zip(['billing','account_access'],p))})
