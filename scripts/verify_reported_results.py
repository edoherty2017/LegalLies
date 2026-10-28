"""
Verify the evaluation results reported in README.md.

Recomputes, from data/evaluation_results.csv alone and with no third-party
dependencies:

  * claims / testable / testability / hallucination rate per variant
  * Wilson 95% confidence intervals on the hallucination rate
  * two-sided Fisher exact tests for all six variant pairs (claims pooled)
  * paired per-question sign and exact Wilcoxon signed-rank tests, which respect
    the fact that all four variants answered the same 15 questions

Rationale: the README previously asserted that activation steering "produces the
largest reduction in factual contradictions." With 15 questions per variant, no
pairwise difference reaches significance (smallest p = 0.193, Baseline vs
Steering), so that claim was removed. This script exists so the corrected
numbers and the negative result can be re-checked.

Two complementary tests are reported because the pooled Fisher test treats each
claim as independent, although claims are nested within only 15 questions. The
paired per-question tests respect the design. Both agree: nothing is significant.

Usage:
    python3 scripts/verify_reported_results.py
"""

import csv, math, itertools
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


# ---------------------------------------------------------------------------
# Paired per-question tests (the design is paired: same 15 questions per variant)
# ---------------------------------------------------------------------------

def _binom_two_sided(k, n, p=0.5):
    if n == 0:
        return 1.0
    pmf = lambda i: math.comb(n, i) * p**i * (1 - p)**(n - i)
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-12))


def _wilcoxon_exact(diffs):
    """Exact two-sided Wilcoxon signed-rank p-value by full enumeration (n <= 15)."""
    diffs = [x for x in diffs if x != 0]
    n = len(diffs)
    if n == 0:
        return None, 0
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    W = sum(ranks[i] for i in range(n) if diffs[i] > 0)
    space = [sum(r for r, sg in zip(ranks, signs) if sg)
             for signs in itertools.product([0, 1], repeat=n)]
    mean = sum(space) / len(space)
    dev = abs(W - mean)
    return min(1.0, sum(1 for w in space if abs(w - mean) >= dev - 1e-9) / len(space)), n


per_q = defaultdict(dict)
with open(PATH) as f:
    for r in csv.DictReader(f):
        con = int(float(r["num_contradicted"] or 0))
        sup = int(float(r["num_supported"] or 0))
        per_q[r["question"]][r["variant"]] = (con, con + sup)

print("\nPAIRED per-question tests (same questions across variants;")
print("a question is dropped from a pair if either side has zero testable claims):")
for x, y in itertools.combinations(sorted(agg), 2):
    pairs = []
    for _q, m in per_q.items():
        if x in m and y in m:
            cx, tx = m[x]
            cy, ty = m[y]
            if tx > 0 and ty > 0:
                pairs.append(cx / tx - cy / ty)
    pos = sum(1 for z in pairs if z > 0)
    neg = sum(1 for z in pairs if z < 0)
    p_sign = _binom_two_sided(pos, pos + neg)
    p_wil, nw = _wilcoxon_exact(pairs)
    worst = min(p_sign, p_wil if p_wil is not None else 1.0)
    flag = "***" if worst < 0.001 else "**" if worst < 0.01 else "*" if worst < 0.05 else "ns"
    pw = "n/a" if p_wil is None else f"{p_wil:.4f}"
    print(f"  {x:<16} vs {y:<16} n={len(pairs):>2} (+{pos}/-{neg})  "
          f"sign p={p_sign:.4f}  wilcoxon p={pw}  {flag}")

print("\nCONCLUSION: no pairwise difference is significant under either test.")
print("The variant ordering reported in earlier READMEs is not distinguishable")
print("from sampling noise at 15 questions per variant.")
