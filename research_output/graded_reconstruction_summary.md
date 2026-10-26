# Fixed-zero Hall rank and local truncated recovery

## Main result

Fix positive integers \(d,D,L\), and let \(\nu=\nu_{d,L}\) be the
number of Hall/log-signature coordinates through degree \(L\). For
componentwise shallow biased-tanh fields
\[
V_i(x)=\tanh(A_ix+b_i),\qquad i=1,\ldots,d,
\]
the degree-ordered Hall evaluation matrix satisfies
\[
\operatorname{rank}[\rho(H_1)(0)\ \cdots\ \rho(H_\nu)(0)]
=\min(D,\nu)
\]
outside a Lebesgue-null analytic exceptional set in the **full** parameter
space \(\mathbb R^{d(D^2+D)}\). Initialization is fixed at zero. The result
holds almost surely under jointly absolutely continuous parameter laws;
bias-free, tied, sparse or other singular laws are not covered automatically.

When \(D\geq\nu\), one zero-initialized vector of the **degree-\(L\)
truncated polynomial response** locally determines all Hall coordinates
near zero, at any fixed nonzero input scale. At \(D=\nu\), the whole vector
has a local analytic inverse; for \(D>\nu\), a selected \(\nu\)-coordinate
projection does. The inverse neighborhood may depend on parameters and scale.
Invariance of domain rules out continuous local injectivity from an open
\(\nu\)-dimensional Hall-coordinate domain into \(D<\nu\) outputs.
Thus \(D=\nu\) is sharp for this one-vector truncated observation model.

The specific contribution is the finite-parameter tanh realization and its
exact prescribed-degree Hall minors. Generic Lie spanning and the use of
analytic evaluation determinants have precedents; the scalar constant/tanh
pair also predates this work. The source comparison below separates these
antecedents from the present finite-degree statement.

## Why the rank statement is proved

Identify the free step-\(L\) Lie algebra with its Hall coordinates, write
\(e_i\) for its generators, and let \(\mathcal E\) multiply grade \(m\)
by \(m\). The finite witness at width \(\nu\) is
\[
\Lambda_a=\mathcal E^{-1}\operatorname{ad}_a\mathcal E,\qquad
V_i(x)=\tanh(qe_i-\Lambda_{e_i}x),\qquad q=\operatorname{atanh}(1/2).
\]
The classical affine action \(a/2-\Lambda_ax\) preserves the repository
bracket. A finite weighted-jet argument gives a block upper triangular
Hall matrix with diagonal blocks \(I/2\). Lower-grade off-diagonal entries
can be nonzero: the full matrix need not equal \(I/2\).

Graded ideal quotients cover \(D<\nu\), including \(D<d\); inactive
coordinates cover \(D>\nu\). In every case the selected square minor is
exactly \(2^{-\min(D,\nu)}\). This is an admissible finite tanh witness,
not a limiting field or a numerical rank sample. Analyticity of that fixed
minor in all weights and biases yields genericity.

The [authoritative theorem note](intrinsic_hall_rank_theorem.md) contains
the complete proof, the local inverse and dimension lower bound, and the
precise attribution of the affine construction to Burde. Finite exact checks
support the implementation; they do not replace the general proof.

## Three observation objects

- **A. Ideal scale jets:** normalized derivatives
  \(J_m=(1/m!)\partial_r^mY_T^{y,r}(X)|_{r=0}\).
- **B. Truncated responses:** the finite polynomial retaining tensor word
  length, equivalently Hall weighted degree, through \(L\).
- **C. Actual CDE responses:** the full reservoir flow, including higher
  levels beyond the truncation.

The new one-vector inverse concerns B. It does not establish exact recovery
from one actual full-CDE endpoint at fixed nonzero scale, a linear-readout
approximation, global inversion, or practical conditioning.

## Preserved scalar results

For \(d=2,D=1,L=4\), the separate witness
\(V_1(y)=\tanh y\), \(V_2(y)=1/2\) has exact nonzero homogeneous
certificates in every degree through four. Its eight Hall fields nevertheless
have cross-degree function rank seven, so homogeneous injectivity is not
joint injectivity. The scalar witness and certificates are unchanged.

Homogeneous injectivity supports recursive Hall recovery from A and,
separately, joint state-scale independence for B. The existing eight-output
design uses initial states \(-q,0,q\) and scales \(1,2,3\). Its local
truncated inverse has an exact rational stability certificate on the cube
of radius \(R_*=1/2048\), with inverse bound \(C_*\). The
[technical overview](technical_note.md) records the scalar determinants,
ordered design and constants, and links to the complete matrices and proofs.

The existing actual-response theorem additionally gives **approximate**
recovery from those same eight experiments. For a continuous BV path with
\(X_0=0\) and total component variation at most \(M\), let
\[
D_h=\operatorname{diag}(h,h,h^2,h^3,h^3,h^4,h^4,h^4),\qquad
(c_1,c_2,c_3,c_4)=(1,1,1,13/12).
\]
Require \(0<h\leq1\) and \(c_m(hM)^m\leq R_*\) for every \(m\leq4\).
Only \(D_hz(X)\), not the unscaled coordinates, must lie in the cube.
For the existing pairs \((y_j,r_j)\), put
\(b_h(X)_j=Y_T^{y_j,hr_j}(X)-y_j\). With response error
\(\|e\|_\infty\leq\delta\), and the existing clipped decoder \(G\),
\[
\|b_h(X)-\Phi(D_hz(X))\|_\infty\leq3402h^5M^5,
\qquad
\|D_h^{-1}G(b_h(X)+e)-z(X)\|_\infty
\leq C_*(3402M^5h+\delta h^{-4}).
\]
Finite iteration has a separate residual bound; response-fit error and
roundoff are also distinct. These scalar estimates do not transfer to the
new general Hall map. The [actual-response audit addendum](paper_mathematical_audit.md#2026-09-08-approximate-recovery-from-eight-actual-cde-responses)
records the proof assumptions and deterministic validation.

## Separate word theorem and reproduction

The [sufficient-width word theorem](general_shallow_tanh_width_theorem.md)
and the fixed-zero corollary in the
[canonical manuscript](../paper/randomized_signature_identifiability.tex)
(label \(\texttt{cor:zero}\)) prove generic independence of all
\(Q_{d,L}=\sum_{m=1}^L d^m\) nonempty word responses, at sufficient width
\(D\geq D_0(d,L)=\sum_{m=0}^L d^m\). The Hall theorem does not establish
independence of that larger family at width \(\nu\). For \(d=2,L=4\),
these counts are \(\nu=8\), \(D_0=31\) and \(Q=30\).

The authoritative note gives the narrow comparison with existing
reconstruction references. [Monzio Compagnoni et al.](https://arxiv.org/pdf/2201.00384v3)
are a same-architecture bracket-generation antecedent.
[Cuchiero–Larsson–Teichmann](https://arxiv.org/pdf/1908.07838v2) supply
analytic-minor genericity methodology for interpolation.
[Grong–Vega-Molino](https://arxiv.org/html/2403.08090v3), Remark 3.2,
already use the scalar constant/tanh pair for landmark controllability.
These results do not specify the exact finite-degree Hall matrix here.
Classical affine machinery is attributed, and projected tree terms remain
distinct from complete summed fields. No worldwide priority, new general
reconstruction principle, or financial-performance consequence is claimed.

Reproduce the 17 finite exact witness checks without writing artifacts:

```text
python -B research_output/reproduce_intrinsic_hall_rank.py
```
