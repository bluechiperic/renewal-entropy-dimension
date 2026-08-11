Ancillary file for

    "Full entropy dimension for a countable strongly overlapping
     3-adic renewal measure"

CONTENTS

    verify_renewal_entropy.py    complete verifier (Python 3.8+, NumPy)

USAGE

    python verify_renewal_entropy.py

Runtime is about 20 seconds. No network access is required and no other
module of the author's work is imported.

WHAT IS CHECKED

  (1) The level-n law of C_n, computed to machine precision from the exact
      recurrence by the discrete-logarithm cyclic convolution of Section 7
      (2 is a primitive root modulo 3^n).  This part is double-precision
      floating point; parts (3) and (4) below are exact.
      Reproduces the first table of Section 7 for n <= 14, verifies the
      bounds (1.7), (1.8) and (1.9) of Theorem 1.1 at every computed level,
      and reports the least-squares fit behind Conjecture 7.1.

  (2) The recurrence (7.1) / Lemma 8.1, to machine precision, for n <= 9.

  (3) Exact integer enumeration of every renewal word of length n with
      n <= t < 6n, for n <= 7.  Verifies Lemma 3.1 atom by atom, Lemma 4.1,
      Lemma 4.2, the cell count (3.6), inequality (4.5) and Proposition 4.3,
      and reproduces the second table of Section 7.

  (4) The exhaustive rational search behind the rigidity Theorem 6.2, over
      coprime p, q <= 12 and lambda = a/b with b <= 24, together with the
      two structural facts used in its proof.

SCOPE

No statement in Sections 2 through 6 depends on any of these computations.
Every theorem, lemma and proposition is proved in full in the text; the
verifier reproduces the constants and the two tables, and confirms the
finite search quoted in Remark 6.4.
