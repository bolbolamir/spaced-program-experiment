from pathlib import Path
import json, hashlib
ROOT=Path(__file__).resolve().parent
# Frozen before model diagnostics: new requests, no model-dependent filtering.
pairs=[
('Signing in is working. Please apply the advertised discount to my renewal invoice.', 'The advertised discount is already applied. Please restore my login after the lockout.'),
('My recovery email arrived and I can get in. Please credit the days I paid for after cancellation.', 'The cancellation credit arrived. Please fix the recovery email that never reaches me.'),
('The authentication issue is settled. Please explain the foreign currency surcharge.', 'The foreign currency surcharge is settled. Please help with a rejected authentication token.'),
('My profile opens normally. Please change the company name printed on my invoice.', 'The company name on the invoice is correct. Please help me regain entry to my profile.'),
('I have my verification codes. Please send proof of the tax charged on this purchase.', 'The tax document is complete. Please help me replace my missing verification codes.'),
('The password change succeeded. Please reimburse the extra seat that was removed last week.', 'The extra seat was reimbursed. Please help me change the password I no longer remember.'),
('Leave my sign-in details alone. I need to move the next renewal from monthly to annual billing.', 'Leave my renewal schedule alone. I need help with my blocked sign-in credentials.'),
('There is no account lockout. Please investigate a charge from a subscription I never ordered.', 'There is no unauthorized subscription charge. Please remove the lockout preventing account entry.'),
('Account recovery is finished. Please issue a partial refund for the service outage.', 'The service outage refund is finished. Please help with account recovery after losing my backup codes.'),
('The security challenge passes. Please fix a receipt that lists the wrong payment amount.', 'The receipt has the correct amount. Please help me pass a security challenge that rejects my answers.'),
('I can access the account again. Please stop collecting fees for the closed workspace.', 'Fees for the closed workspace have stopped. Please help me access the account again.'),
('My new authenticator is connected. Please transfer the remaining credit to the new plan.', 'The remaining credit was transferred. Please connect a replacement authenticator so I can log in.')]
headers={0:'Billing department. Invoice, payment, refund, subscription.',1:'Account access department. Login, password, authentication, security.'}
rows=[]
for i,pair in enumerate(pairs):
 for label,request in enumerate(pair):
  rows.append({'id':f'fresh_{i:02d}_{label}','split':'fresh_hard_test','label':label,'text':f'Archived routing metadata: {headers[1-label]} Customer current request: {request}'})
# Separate format stress cases: short/no prefix and variable-length headers, no training examples.
stress=[
(0,'Password recovery is complete. Please return the unused balance on my cancelled plan.'),
(1,'My subscription is paid. I need another way to verify my identity after losing my phone.'),
(0,'Old category: access. Current issue: I can log in, but the promised credit is missing from my statement.'),
(1,'Old category: billing. Current issue: the credit is on my statement, but my recovery link is invalid.'),
(0,'Current request: send an itemized invoice for my last purchase. Historical team: account security.'),
(1,'Current request: unlock the account following too many password attempts. Historical team: payments.'),
(0,'Routing history: login issues, account recovery, security verification, password assistance, device enrollment. None of those remain open. The remaining request is to refund an accidental upgrade.'),
(1,'Routing history: invoicing, card charges, refunds, payment collection, billing adjustments. None of those remain open. The remaining request is to recover access after a stolen phone.'),
(0,'I can enter my account, receive verification codes and change my password. The only thing I need is a corrected business receipt.'),
(1,'I have my receipt, the amount is correct and the refund is complete. The only thing I need is a working password reset email.'),
(0,'Customer current request: reverse the duplicate transfer. No issue with authentication. Archived routing metadata: account access.'),
(1,'Customer current request: remove the sign-in block. No issue with the transfer. Archived routing metadata: billing.')]
for i,(label,text) in enumerate(stress):rows.append({'id':f'format_{i:02d}','split':'format_stress_test','label':label,'text':text})
p=ROOT/'data/final_holdout.json';p.write_text(json.dumps(rows,indent=2)+'\n')
(ROOT/'data/holdout_sha256.txt').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  final_holdout.json\n')
print('Frozen final holdout:',len(rows),'cases; do not evaluate until selection is finalized.')
