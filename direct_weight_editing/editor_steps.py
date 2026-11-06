"""Explicit patches authored by Codex after reading each preceding diagnostic result.

This file replays decisions. It is not an automated editor or parameter search.
Every patch starts from the same saved baseline unless stated otherwise.
"""
import torch
from .session import Session

def step_1(s:Session):
 s.load_patch('no_edit')
 # Layer 5, head 7: highest archived-header attention among the last-block heads.
 # Head size 32; zero only this head's contribution through the output projection.
 key='encoder.encoder.layer.5.attention.output.dense.weight'
 weight=s.model.state_dict()[key]
 weight[:,7*32:8*32]=0
 return s.assess('attention_head_5_7_off',
  'Inspection showed last-block head 7 allocates 45.3% of request attention to the archived header on two repair probes. Test its causal contribution by zeroing only its 32 output-projection columns. Attention mass alone does not establish causality; measure repair and train retention.')

STEPS=[step_1]

def step_2(s:Session):
 s.load_patch('no_edit')
 # A format-specific experiment, not a general learned rationale mechanism.
 # The two observed archived headers occupy positions 1..17 or 1..18;
 # actual request text begins at 22 or 23. CLS remains untouched.
 key='encoder.embeddings.position_embeddings.weight'
 sep=s.baseline['encoder.embeddings.word_embeddings.weight'][s.tokenizer.sep_token_id]
 replacement=8.0*sep
 s.model.state_dict()[key][1:19]=replacement.to(s.model.state_dict()[key].device)
 return s.assess('header_position_neutralization',
  'Head 5.7 ablation did not improve repair. The rationales say archived metadata is irrelevant. Token inspection places that header at positions 1..18 in this schema. Replace those positional rows with 8 times the pretrained SEP embedding (norm about 8.46 versus ordinary token norms about 1.3), intending to attenuate header-token identity through embedding LayerNorm. Forward code and masks stay fixed. This is a format-specific weight program and may damage inputs whose requests occur earlier; assess that limitation on format stress.')

STEPS.append(step_2)

def step_3(s:Session):
 s.load_patch('no_edit')
 key='encoder.encoder.layer.5.output.dense.weight'
 s.model.state_dict()[key][:,77]=0
 return s.assess('ffn_column_77_off',
  'FFN diagnostic: column 77 writes toward account access (signed billing-minus-access alignment -0.3511). Its mean header activation is 0.1729 on true-billing repair cases versus 0.0543 on their active evidence, suggesting a possible archived-access contribution. Disable only that column. Do not disable column 1478, whose activation is also much higher on genuine active access evidence. These are attribution hypotheses tested by intervention, not proven feature identities.')

STEPS.append(step_3)

def step_4(s:Session):
 s.load_patch('header_position_neutralization')
 key='encoder.encoder.layer.0.attention.output.dense.weight'
 s.model.state_dict()[key][:,6*32:7*32]=0
 return s.assess('position_plus_early_head_off',
  'Position neutralization improved repair from 1/8 to 3/8 with 23/24 train retained; the isolated FFN edit did not help. For the final budgeted candidate, combine the positional patch with disabling early attention head 0.6, which had the highest request-to-header attention in the diagnostic (49.4%). This tests whether an early header-focused pathway continues to carry misleading information after input neutralization. No further candidates after final evaluation.')

STEPS.append(step_4)
