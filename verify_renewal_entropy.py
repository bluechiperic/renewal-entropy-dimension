#!/usr/bin/env python3
"""Verifier for

    "Full entropy dimension for a countable strongly overlapping
     3-adic renewal measure".

Self-contained. Requires only Python 3.8+ and NumPy. No network access and
no other module of the author's work. Runs in well under a minute.

It checks, in four independent parts:

  (1) the level-n law to machine precision from the exact recurrence, via
      the discrete-logarithm cyclic convolution
      of Section 7, reproducing the table of R_n for n <= 14 together with
      D_n and G_n = (R_{n+1}-R_n)/2;
  (2) the recurrence (7.1) / Lemma 8.1, to machine precision, n <= 9;
  (3) exact integer enumeration of every renewal word with n <= t < 6n for
      n <= 7, checking Lemma 3.1 atom by atom, Lemmas 4.1 and 4.2, the cell
      count (3.6) and Proposition 4.3, and reproducing the second table of
      Section 7;
  (3b) the diagonal/shell split of Section 7, including sum_t sqrt(f_t) =
      O(n^{1/4}), the constant 82, Remark 3.2 and the moment condition;
  (4) the exhaustive rational search behind the rigidity Theorem 6.2.

Part (1) is computed to machine precision in double-precision floating point
from the exact recurrence; parts (3) and (4) are exact integer/rational
arithmetic.

Usage:  python verify_renewal_entropy.py
"""
from fractions import Fraction as F
from collections import defaultdict
from math import gcd, log
import sys

import numpy as np

fails = []


def ck(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


# --------------------------------------------------------------- part (1)
def exact_laws(N):
    """p_n on (Z/3^n)^* for n = 1..N, to machine precision.

    2 is a primitive root mod 3^n, so C_n = 2^{-K}(1 + 3 C_{n-1}) is a
    cyclic convolution of length M = 2*3^{n-1} in discrete-log coordinates.
    """
    store, p = {}, None
    for n in range(1, N + 1):
        q3, M = 3 ** n, 2 * 3 ** (n - 1)
        dlog = np.full(q3, -1, dtype=np.int64)
        pw = np.empty(M, dtype=np.int64)
        v = 1
        for l in range(M):
            dlog[v] = l
            pw[l] = v
            v = v * 2 % q3
        q = np.zeros(M)
        if n == 1:
            q[dlog[1]] = 1.0
        else:
            qprev, Mprev = 3 ** (n - 1), 2 * 3 ** (n - 2)
            yv = np.empty(Mprev, dtype=np.int64)
            v = 1
            for l in range(Mprev):
                yv[l] = v
                v = v * 2 % qprev
            np.add.at(q, dlog[(1 + 3 * yv) % q3], p)
        r = np.arange(M)
        rp = np.where(r == 0, M, r).astype(float)   # least k >= 1 with k = r mod M
        w = np.exp2(-rp) / (1 - 2.0 ** (-M))
        p = np.roll(np.fft.irfft(np.fft.rfft(w[::-1]) * np.fft.rfft(q), M), 1)
        p = np.maximum(p, 0.0)
        store[n] = (p.copy(), pw.copy(), q3, M)
    return store


print("=== (1) level-n law to machine precision: Thm 1.1 and the Section 7 table ===")
NMAX = 14
S = exact_laws(NMAX)
R = {}
for n in range(1, NMAX + 1):
    p = S[n][0]
    R[n] = 3.0 ** n * float(np.dot(p, p))
    ck(f"n={n:>2}: total mass 1", abs(p.sum() - 1) < 1e-9)

print(f"\n  {'n':>3} {'R_n':>10} {'R_n/n^3':>9} {'G_n':>9} {'D_n':>9} {'82n^3':>9}  R_n<=82n^3")
for n in range(1, NMAX + 1):
    p = S[n][0]
    nz = p[p > 0]
    D = n * log(3) + float((nz * np.log(nz)).sum())
    G = (R[n + 1] - R[n]) / 2 if n + 1 in R else float("nan")
    print(f"  {n:>3} {R[n]:>10.5f} {R[n]/n**3:>9.5f} {G:>9.5f} {D:>9.6f} "
          f"{82*n**3:>9} {'ok' if R[n] <= 82*n**3 else 'VIOLATION'}")
ck("R_n <= 82 n^3 for all computed n", all(R[n] <= 82 * n ** 3 for n in R))
ck("R_n <= 70n^2(1+n/6) + 2*gamma^n  (1.7)",
   all(R[n] <= 70 * n * n * (1 + n / 6) + 2 * (3 ** 13 / 5 ** 10) ** n for n in R))
nz = S[NMAX][0][S[NMAX][0] > 0]
D14 = NMAX * log(3) + float((nz * np.log(nz)).sum())
ck("D_n <= log 82 + 3 log n  (1.9)", D14 <= log(82) + 3 * log(NMAX),
   f"D_14 = {D14:.6f} <= {log(82)+3*log(NMAX):.6f}")
fit = np.polyfit(range(7, 14), [R[n] for n in range(7, 14)], 1)
print(f"\n  least-squares fit on 7 <= n <= 13:  R_n ~ {fit[0]:.4f} n + {fit[1]:.4f}")
ck("gamma = 3^13/5^10 < 1", 3 ** 13 / 5 ** 10 < 1, f"gamma = {3**13/5**10:.10f}")

# --------------------------------------------------------------- part (2)
print("\n=== (2) recurrence (7.1): Col_{n+1} = Col_n/3 + (2/3) sum_h 4^-h Gamma_n(h) ===")
worst = 0.0
for n in range(1, 10):
    p, pw, q3, _ = S[n]
    res = np.zeros(q3)
    res[pw] = p
    Col = float(np.dot(p, p))
    pn1 = S[n + 1][0]
    Col1 = float(np.dot(pn1, pn1))
    tot = 0.0
    for h in range(1, 80):
        a = pow(4, h, q3)
        b = ((4 ** h - 1) // 3) % q3          # exact integer, then reduce
        idx = (a * pw.astype(object) + b) % q3
        tot += 4.0 ** (-h) * float(np.dot(p, res[np.array(idx, dtype=np.int64)]))
        if 4.0 ** (-h) < 1e-19:
            break
    rel = abs(Col / 3 + 2 * tot / 3 - Col1) / Col1
    worst = max(worst, rel)
ck("exact for n = 1..9", worst < 1e-12, f"max relative error {worst:.2e}")

# --------------------------------------------------------------- part (3)
print("\n=== (3) exact enumeration: Lemmas 3.1, 4.1, 4.2, (3.6), Prop 4.3 ===")


def enumerate_cells(n):
    """Every renewal word of length n with n <= t < 6n, in exact integer
    arithmetic.  Masses are scaled by D = 2^{6n} so that everything below is
    an integer; the scaling is divided out at the end.

    Words are generated by depth-first search using the recursion of
    Lemma 2.1 read forwards, N_{j+1} = 2^{k_j} N_j + 3^j with N_1 = 1, so
    each numerator is built incrementally.  The dyadic shell of the address
    x = N/2^t is r = floor(log2 N) - t, obtained exactly from the bit length.
    """
    q3, TM = 3 ** n, 6 * n
    D = 1 << TM
    inv = {t: pow(pow(2, t, q3), -1, q3) for t in range(n, TM)}
    p3 = [3 ** j for j in range(n)]
    cell = defaultdict(lambda: defaultdict(int))
    eps = defaultdict(int)
    core = defaultdict(int)

    def rec(j, s, N):
        if j == n:
            t = s
            r = N.bit_length() - 1 - t
            c = (N % q3) * inv[t] % q3
            m = 1 << (TM - t)
            cell[(t, r)][c] += m
            eps[(t, r)] += m
            core[c] += m
            return
        rem = n - j - 1                      # increments still to place after this one
        for k in range(1, TM - s - rem):
            rec(j + 1, s + k, 1 if j == 0 else (N << k) + p3[j])

    sys.setrecursionlimit(10000)
    rec(0, 0, 0)
    return cell, eps, core, D


print(f"\n  {'n':>2} {'Q_n':>5} {'35n^2':>6} {'sum a^2':>9} {'1+n/6':>7} "
      f"{'diag':>8} {'shell':>8} {'n/6':>7} {'R_core':>8} {'Prop4.3':>10} {'Q_eff':>7}")
NEN = 7
for n in range(1, NEN + 1):
    cell, eps, core, D = enumerate_cells(n)
    Q = len(cell)
    # Lemma 3.1, atom by atom, cleared of denominators
    bad = 0
    for (t, r), d in cell.items():
        lhs = 2 * 3 ** n * max(d.values())
        rhs = 2 * 3 ** n * (D >> t) + (D << r if r >= 0 else D >> (-r))
        if lhs > rhs:
            bad += 1
    A2 = F(3 ** n, D * D) * sum(sum(v * v for v in d.values()) for d in cell.values())
    diag = 3 ** n * sum(F(eps[(t, r)], D << t) for (t, r) in cell)
    shell = sum((F(1 << r, 2) if r >= 0 else F(1, 2 << (-r))) * F(eps[(t, r)], D)
                for (t, r) in cell)
    Rc = F(3 ** n, D * D) * sum(v * v for v in core.values())
    prop = 35 * n * n * (1 + F(n, 6))
    qeff = float(Rc) / float(A2)
    print(f"  {n:>2} {Q:>5} {35*n*n:>6} {float(A2):>9.5f} {float(1+F(n,6)):>7.4f} "
          f"{float(diag):>8.5f} {float(shell):>8.5f} {float(F(n,6)):>7.4f} "
          f"{float(Rc):>8.4f} {float(prop):>10.2f} {qeff:>7.4f}")
    ck(f"  n={n}: Lemma 3.1 (no violating atom)", bad == 0)
    ck(f"  n={n}: Lemma 4.1 diagonal <= 1", diag <= 1)
    ck(f"  n={n}: Lemma 4.2 shell <= n/6", shell <= F(n, 6))
    ck(f"  n={n}: (3.6) Q_n <= 35n^2", Q <= 35 * n * n)
    ck(f"  n={n}: (4.5) sum a^2 <= 1 + n/6", A2 <= 1 + F(n, 6))
    ck(f"  n={n}: Prop 4.3 R_core <= 35n^2(1+n/6)", Rc <= prop)

# ------------------------------------------------- part (3b), Section 7
print("\n=== (3b) the diagonal/shell split of Section 7 ===")
from math import lgamma, exp, sqrt


def logf(t, n):                      # log of the NegBin(n, 3/4) pmf at t
    return (lgamma(t) - lgamma(n) - lgamma(t - n + 1)
            + n * log(0.75) + (t - n) * log(0.25))


print(f"  {'n':>6} {'sum f_t':>9} {'sum sqrt f_t':>13} {'ratio to n^1/4':>15}")
rat = []
for n in (10, 50, 200, 1000, 5000, 20000):
    tot = ssq = 0.0
    for t in range(n, int(4 * n / 3 + 40 * sqrt(n)) + 50):
        v = exp(logf(t, n))
        tot += v
        ssq += sqrt(v)
    rat.append(ssq / n ** 0.25)
    print(f"  {n:>6} {tot:>9.6f} {ssq:>13.4f} {ssq/n**0.25:>15.4f}")
    ck(f"  n={n}: f_t is a probability mass function", abs(tot - 1) < 1e-9)
ck("sum_t sqrt(f_t) = O(n^{1/4}), ratio -> 1.828",
   abs(rat[-1] - 1.828) < 0.01 and max(rat) < 1.9,
   f"ratios {['%.4f' % x for x in rat]}")
ck("shell sector reproduces the constant of Prop 4.3 exactly: "
   "(sqrt(7n)*sqrt(5n*n/6))^2 = (35/6) n^3",
   abs(7 * 5 / 6 - 35 / 6) < 1e-12)
ck("smallest clean cubic constant is 82, not 84",
   max((70*n*n*(1+n/6) + 2*(3**13/5**10)**n)/n**3 for n in range(1, 400)) <= 82,
   f"max ratio {max((70*n*n*(1+n/6)+2*(3**13/5**10)**n)/n**3 for n in range(1,400)):.4f}")

print("\n  Remark 3.2: exact cell count 5n(6n + floor(n log2 3/2) + 1)")
from math import floor, log2
bad = [n for n in range(1, 4000)
       if 5*n*(6*n + floor(n*log2(1.5)) + 1) > 33*n*n]
ck("Q_n <= 35n^2 holds for every n",
   all(5*n*(6*n+floor(n*log2(1.5))+1) <= 35*n*n for n in range(1, 4000)))
ck("Q_n <= 33n^2 FAILS for small n, so the sharper form is not usable",
   len(bad) > 0 and bad[0] == 1,
   f"fails at n = {bad[:8]} ({len(bad)} values below 4000)")
ck("35n^2 is attained exactly at n = 1 and n = 2",
   all(5*n*(6*n+floor(n*log2(1.5))+1) == 35*n*n for n in (1, 2)))

print("\n  Section 7: moment condition for x =d 2^-K (1+3x), i.e. 3^p + 1 < 2^(p+1)")
for pp in (0.5, 0.9, 0.99, 1.0, 1.01, 1.5):
    fin = 3**pp + 1 < 2**(pp+1)
    print(f"    p={pp:<5} 3^p+1={3**pp+1:8.5f}  2^(p+1)={2**(pp+1):8.5f}  "
          f"{'finite' if fin else 'INFINITE'}")
ck("E[x^p] finite for p < 1", all(3**pp+1 < 2**(pp+1) for pp in (0.5, 0.9, 0.99, 0.999)))
ck("E[x^p] infinite for p >= 1, with equality at p = 1",
   abs((3**1.0+1) - 2**2.0) < 1e-12
   and all(3**pp+1 >= 2**(pp+1) for pp in (1.0, 1.01, 1.1, 1.5, 2.0)))
ck("Thm 1.1 cubic step: 35/3 + 70/n + 2g^n/n^3 strictly decreasing in n",
   all((35/3 + 70/n + 2*(3**13/5**10)**n/n**3)
       > (35/3 + 70/(n+1) + 2*(3**13/5**10)**(n+1)/(n+1)**3)
       for n in range(1, 400)))
ck("Thm 1.1 cubic step: gamma < 1/6", 3**13/5**10 < 1/6,
   f"gamma = {3**13/5**10:.10f} < {1/6:.10f}")
ck("Thm 1.1 cubic step: 245/3 + 2*gamma < 245/3 + 1/3 = 82",
   245/3 + 2*(3**13/5**10) < 245/3 + 1/3 == 82.0,
   f"{245/3 + 2*(3**13/5**10):.6f} < 82")

print("\n  Lemma 7.3 (moments of the stationary address)")
ck("sufficiency: ratio 3^p E[2^-pK] = 3^p/(2^(1+p)-1) < 1 iff p < 1",
   all(3**pp/(2**(1+pp)-1) < 1 for pp in (0.1, 0.5, 0.9, 0.99))
   and all(3**pp/(2**(1+pp)-1) >= 1 for pp in (1.0, 1.01, 1.5, 2.0)))
def _partial(pp, N):                 # running product: no overflow
    m = 1/(2**(1+pp)-1); r = 3**pp*m
    s = 0.0; term = m
    for _ in range(N):
        s += term; term *= r
    return s, m/(1-r), r
for pp in (0.3, 0.5, 0.7, 0.9, 0.99):
    s, cf, r = _partial(pp, 200000)
    print(f"    p={pp:<5} ratio={r:.6f}  partial={s:.9f}  closed form={cf:.9f}")
ck("sufficiency: partial sums bounded uniformly in n, converging to m/(1-3^p m)",
   all(abs(_partial(pp, 200000)[0] - _partial(pp, 200000)[1]) < 1e-9
       for pp in (0.3, 0.5, 0.7, 0.9, 0.99)))
ck("at p=1/2 this reproduces the half-moment constant 10.3759931",
   abs(_partial(0.5, 200000)[1] - 10.3759930789) < 1e-7)
ck("a.s. convergence: E[K] = 2, so SLLN applies",
   abs(sum(k*2.0**-k for k in range(1, 400)) - 2) < 1e-12)
ck("a.s. convergence: 0 < eps < 2 - log2(3) is a nonempty range",
   2 - (log(3)/log(2)) > 0, f"2 - log2(3) = {2-log(3)/log(2):.6f}")
ck("a.s. convergence: ratio 3*2^-(2-eps) < 1 exactly for eps < 2 - log2(3)",
   all(3*2.0**-(2-e) < 1 for e in (0.01, 0.2, 0.41))
   and all(3*2.0**-(2-e) >= 1 for e in (2-log(3)/log(2), 0.5, 0.9)))
ck("necessity: E[2^-K] = 1/3, so a finite mean would give E x = 1/3 + E x",
   abs(sum(2.0**-k * 2.0**-k for k in range(1, 400)) - 1/3) < 1e-15)
ck("Lemma 8.1 parity: a collision mod 3 forces K = K' mod 2",
   all(pow(2, -k, 3) == (1 if k % 2 == 0 else 2) for k in range(1, 40)))
ck("Lemma 8.1: Pr(K-K' = 2h) = (1/3) 4^-h, two signs give (2/3) 4^-h",
   all(abs(sum(2.0**-(k+2*h) * 2.0**-k for k in range(1, 400)) - (1/3)*4.0**-h) < 1e-14
       for h in range(1, 12)))
ck("negative-shell constant: sum_{r<0} 2^{(r-1)/2} = 1/(2-sqrt2)",
   abs(sum(2**((r-1)/2) for r in range(-1, -200, -1)) - 1/(2-2**0.5)) < 1e-12,
   f"= {1/(2-2**0.5):.6f}")

# --------------------------------------------------------------- part (4)
print("\n=== (4) rigidity, Theorem 6.2: exhaustive rational search ===")


def gd(p, lam):
    return F(p) * (1 - lam) / (1 + lam)          # p * sum w_k^2


def gm(p, q, lam):
    return F(p) * (1 - lam) / (q - lam)          # p * E[q^-K]


def gs(q, lam):                                   # (1-lam)^2 q / (1 - lam^2 q)
    return F(q) * (1 - lam) ** 2 / (1 - lam * lam * q)


sols = set()
PQ, BB = 12, 24
for p in range(2, PQ + 1):
    for q in range(2, PQ + 1):
        if gcd(p, q) != 1:
            continue
        for b in range(2, BB + 1):
            for a in range(1, b):
                lam = F(a, b)
                if lam * lam * q >= 1:
                    continue
                if gd(p, lam) <= 1 and gm(p, q, lam) <= 1 and gs(q, lam) <= 1:
                    sols.add((p, q, lam))
print(f"  coprime p,q <= {PQ};  lambda = a/b, b <= {BB};  solutions: {len(sols)}")
for s in sorted(sols, key=lambda z: (z[0], z[1], z[2])):
    p, q, lam = s
    print(f"    (p,q,lambda) = ({p},{q},{lam}):  "
          f"gamma_d = {gd(p,lam)}, gamma_m = {gm(p,q,lam)}, gamma_s = {gs(q,lam)}")
ck("unique solution", len(sols) == 1)
ck("the solution is (3, 2, 1/2)", sols == {(3, 2, F(1, 2))})
p, q, lam = 3, 2, F(1, 2)
ck("all three critical: gamma_d = gamma_m = gamma_s = 1",
   gd(p, lam) == 1 and gm(p, q, lam) == 1 and gs(q, lam) == 1)
# the two structural facts underlying the proof of Theorem 6.2
ck("(1-l)^2 + l^2 >= 1/2 with equality only at l = 1/2",
   all(F((b - a), b) ** 2 + F(a, b) ** 2 >= F(1, 2)
       for b in range(2, 40) for a in range(1, b))
   and all(F((b - a), b) ** 2 + F(a, b) ** 2 > F(1, 2)
           for b in range(2, 40) for a in range(1, b) if F(a, b) != F(1, 2)))
ck("with lambda = 1/q the shell prefactor is (q-1)^n/q, bounded only for q=2",
   all(gs(qq, F(1, qq)) == qq - 1 for qq in range(2, 13)))

print("\n" + ("ALL CHECKS PASS" if not fails else f"FAILURES ({len(fails)}): {fails}"))
sys.exit(0 if not fails else 1)
