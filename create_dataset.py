"""Freeze an authored, synthetic 50-ticket benchmark; no labels are inferred from a model."""
import json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
pairs={
 'train':[
 ('Please refund the duplicate charge on my card.','Please reset my forgotten password.'),
 ('I was billed twice for the same purchase.','My account is locked after failed sign-in attempts.'),
 ('Cancel my subscription and stop the recurring payment.','The password reset link has expired.'),
 ('The invoice total is incorrect; please adjust it.','I cannot log in with my username.'),
 ('Return the money taken for a cancelled order.','My authentication code never arrives.'),
 ('I need a receipt for last month’s payment.','I lost access to my two-factor authentication device.'),
 ('Update the credit card used for my subscription.','Please help me recover my account.'),
 ('A payment was declined even though my card is valid.','Someone changed my password and I cannot sign in.'),
 ('Please remove the unexpected service fee.','My login session keeps being rejected.'),
 ('How can I dispute this transaction?','I need to change the email used for account recovery.'),
 ('The annual renewal price is wrong.','I forgot the answer to my security question.'),
 ('Issue a credit for the unused subscription period.','The sign-in page rejects my verification code.')],
 'test':[
 ('Send me a corrected invoice for this order.','Unlock my profile so I can sign in again.'),
 ('Why did the monthly subscription price increase?','The recovery email contains a broken reset link.'),
 ('Please reimburse the extra amount you collected.','I replaced my phone and lost the authenticator app.'),
 ('I want to switch the card for future payments.','I cannot get past the account verification screen.')],
 'hard_repair':[
 ('My login works and no password reset is needed. I only want the extra payment refunded.','The invoice is correct and no refund is needed. I only need my password reset.'),
 ('Access has been restored; the unresolved issue is a duplicate debit that must be reversed.','The duplicate debit has been reversed; the unresolved issue is that I cannot sign in.'),
 ('Do not unlock anything. Please cancel the paid plan and stop charging me.','Do not cancel the paid plan. Please unlock my account so I can use it.'),
 ('I am already signed in. The charge for a cancelled order still needs reimbursement.','I received the reimbursement. The account recovery code still does not arrive.')],
 'hard_test':[
 ('There is no access problem anymore. I need a receipt for a recent transaction.','There is no payment problem anymore. I need a new account recovery link.'),
 ('Authentication succeeds. What remains is an incorrect renewal amount on the statement.','The renewal amount is correct. What remains is an authentication failure.'),
 ('Keep my credentials unchanged; please return the money for the unused month.','Keep my payment method unchanged; please replace my lost two-factor device.'),
 ('The verification screen is fine. Remove the unexplained fee from my bill.','The fee on my bill is fine. Help me get through the verification screen.'),
 ('I can enter my profile normally. The next payment must use a different credit card.','My credit card is fine. I forgot my security answer and cannot enter my profile.')]
}
headers={0:'Billing department. Invoice, payment, refund, subscription.',1:'Account access department. Login, password, authentication, security.'}
rows=[]
for split,items in pairs.items():
 for pair in items:
  for label,request in enumerate(pair):
   header_label=1-label if split.startswith('hard') else label
   rows.append(dict(id=f'{split}_{len([r for r in rows if r["split"]==split]):02d}',split=split,label=label,label_name=['billing','account_access'][label],legacy_route=header_label,request=request,text=f'Archived routing metadata: {headers[header_label]} Customer current request: {request}'))
assert len(rows)==50 and len({r['text'] for r in rows})==50
out=ROOT/'data/tickets.json'; out.write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
(ROOT/'data/SHA256.txt').write_text(hashlib.sha256(out.read_bytes()).hexdigest()+'  tickets.json\n')
print({s:sum(r['split']==s for r in rows) for s in pairs})
