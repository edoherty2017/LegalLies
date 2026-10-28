"""
Verify the evaluation results reported in README.md.

Recomputes, from data/evaluation_results.csv alone and with no third-party
dependencies:

  * claims / testable / testability / hallucination rate per variant
  * Wilson 95% confidence intervals on the hallucination rate
  * two-sided Fisher exact tests for all six variant pairs

Rationale: the README previously asserted that activation steering "produces the
largest reduction in factual contradictions." With 15 questions per variant, no
pairwise difference reaches significance (smallest p = 0.193, Baseline vs
Steering), so that claim was removed. This script exists so the corrected
numbers and the negative result can be re-checked.

Caveat: the Fisher test treats each claim as independent, but claims are nested
within only 15 questions per variant. Accounting for that clustering would widen
the intervals further, so these p-values are an optimistic bound.

Usage:
    python3 scripts/verify_reported_results.py
"""

import csv, math
from collections import defaultdict

import os
PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "evaluation_results.csv")
agg=defaultdict(lambda: dict(claims=0,sup=0,con=0,uns=0,n=0))
with open(PATH) as f:
    for r in csv.DictReader(f):
        v=r["variant"]; a=agg[v]; a["n"]+=1
        for k,c in (("claims","num_claims"),("sup","num_supported"),("con","num_contradicted"),("uns","num_unsupported")):
            try: a[k]+=int(float(r[c] or 0))
            except: pass

def wilson(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); m=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-m)/d,(c+m)/d)

def fisher(a,b,c,d):
    # two-sided Fisher exact
    n=a+b+c+d
    def p(a_):
        b_=a+b-a_; c_=a+c-a_; d_=n-a_-b_-c_
        if min(a_,b_,c_,d_)<0: return 0.0
        return (math.comb(a+b,a_)*math.comb(c+d,c_))/math.comb(n,a+c)
    obs=p(a); lo=max(0,a-d); hi=min(a+b,a+c)
    return min(1.0,sum(p(i) for i in range(lo,hi+1) if p(i)<=obs+1e-12))

print(f"{'variant':<16}{'resp':>5}{'claims':>8}{'testable':>9}{'testabil':>9}{'hall.rate':>10}  {'95% CI (Wilson)':>22}")
rows={}
for v,a in sorted(agg.items()):
    t=a["sup"]+a["con"]
    hr=a["con"]/t if t else float('nan')
    lo,hi=wilson(a["con"],t)
    rows[v]=(a["con"],t)
    print(f"{v:<16}{a['n']:>5}{a['claims']:>8}{t:>9}{(t/a['claims'] if a['claims'] else 0):>8.1%}{hr:>10.1%}  [{lo:>6.1%},{hi:>6.1%}]")

print("\nPairwise two-sided Fisher exact on CONTRADICTED vs SUPPORTED:")
keys=list(rows)
import itertools
for x,y in itertools.combinations(keys,2):
    cx,tx=rows[x]; cy,ty=rows[y]
    pv=fisher(cx,tx-cx,cy,ty-cy)
    star="***" if pv<0.001 else "**" if pv<0.01 else "*" if pv<0.05 else "ns"
    print(f"  {x:<16} vs {y:<16} p={pv:.4f}  {star}")
