# Exact local stability of the certified truncated observation map

## 1. Scope and authoritative objects

This note proves a quantitative local stability theorem only for observation
object B: the degree-four truncated response polynomial at the certified scalar
tanh witness

\[
q=\frac{\log 3}{2},\qquad V_1(x)=\tanh x,\qquad V_2(x)=\frac12.
\]

The separate [fixed-state Hall-rank theorem](intrinsic_hall_rank_theorem.md)
gives generic rank \(\min(D,\nu_{d,L})\) at zero in the full biased
shallow-tanh family, where \(\nu_{d,L}\) is the total Hall dimension through
degree \(L\), and, for \(D\geq\nu_{d,L}\), dimension-optimal local
recovery near zero Hall coordinates from one truncated terminal vector at
a fixed nonzero scale. That qualitative result has no certified radius or
conditioning constant. None of the scalar design's exact constants below
is asserted for the new vector map.

All word, tensor, Hall-basis, operator-order, bracket-sign, and factorial
conventions are those proved in
`research_output/graded_reconstruction_derivation.md`.  The fixed ordered
state-scale design and its exact zero-signature Jacobian are those certified in
`research_output/finite_observation_theorem.md` and
`research_output/finite_observation_certificate.json`:

\[
\mathcal D_*=((-q,1),(-q,2),(-q,3),(0,2),(0,3),
               (q,1),(q,2),(q,3)).
\]

Let \(z\in\mathbb R^8\) denote the Hall-coordinate vector in degree-then-Lyndon
order, with degree vector

\[
d=(1,1,2,3,3,4,4,4).
\]

Write \(\Phi=\Phi_{\mathcal D_*}^{(4)}\) for the literal truncated observation
map and

\[
\widehat\Phi(z)=\Phi(z)-\Phi(0)
\]

for its centered version.  The two maps have identical differences and
Jacobians.  In particular,

\[
\widehat\Phi(0)=0,\qquad
B:=D\widehat\Phi(0)=D\Phi(0).
\]

The Gate-3 exact certificate proves

\[
\det B=-\frac{258733035}{524288}\ne0
\]

and stores a two-sided exact inverse.  Throughout this note,

\[
\|x\|_\infty=\max_i|x_i|,
\qquad
\|A\|_\infty=\max_i\sum_j|A_{ij}|
\]

are the vector infinity norm and its induced matrix norm.  Exact row sums give

\[
K:=\|B^{-1}\|_\infty=\frac{23812}{2151}.
\]

No floating-point quantity enters the stability proof.

## 2. Exact coefficient-sum Jacobian majorant

The centered map is an exact polynomial of total Hall-coordinate degree at
most four.  Therefore

\[
D\widehat\Phi(z)-B
\]

has total degree at most three.  For its \((j,i)\) entry, write

\[
\bigl(D\widehat\Phi(z)-B\bigr)_{ji}
 =\sum_{1\le |\alpha|\le3}c_{ji,\alpha}z^\alpha,
\qquad c_{ji,\alpha}\in\mathbb Q.
\]

For output row \(j\) and polynomial degree \(k\), define the exact
coefficient sum

\[
a_{j,k}:=
 \sum_{i=1}^8\ \sum_{|\alpha|=k}|c_{ji,\alpha}|.
\]

The authoritative word/Hall expansion and exact response representation give
the following complete row data:

| row \(j\) | \(a_{j,1}\) | \(a_{j,2}\) | \(a_{j,3}\) |
|---:|---:|---:|---:|
| 1 | \(225/128\) | \(189/256\) | \(47/128\) |
| 2 | \(135/8\) | \(165/16\) | \(47/8\) |
| 3 | \(9369/128\) | \(12717/256\) | \(3807/128\) |
| 4 | \(18\) | \(8\) | \(2\) |
| 5 | \(315/4\) | \(297/8\) | \(81/8\) |
| 6 | \(225/128\) | \(189/256\) | \(47/128\) |
| 7 | \(135/8\) | \(165/16\) | \(47/8\) |
| 8 | \(9369/128\) | \(12717/256\) | \(3807/128\) |

For \(R>0\), put

\[
Q_R:=\{z\in\mathbb R^8:\|z\|_\infty\le R\}
\]

and define the exact coefficient-sum majorant

\[
\boxed{
\Delta(R):=
 \max_{1\le j\le8}
 \left(a_{j,1}R+a_{j,2}R^2+a_{j,3}R^3\right).}
\]

If \(z\in Q_R\), then \(|z^\alpha|\le R^{|\alpha|}\).  The triangle
inequality applied entry by entry and then row by row gives

\[
\boxed{
\|D\widehat\Phi(z)-B\|_\infty\le\Delta(R),
\qquad z\in Q_R.}
\]

Here \(\Delta(R)\) is an exact rational upper bound obtained from absolute
coefficient sums.  It is not claimed to equal the supremum of the Jacobian
variation.

## 3. Complete prescribed-radius certificate

The prescribed search evaluates all sixteen radii

\[
R_n=2^{-n},\qquad 1\le n\le16,
\]

in decreasing order.  A radius passes the fixed half-margin test precisely
when

\[
K\Delta(R_n)\le\frac12.
\]

The complete exact table is:

| \(n\) | \(R_n\) | \(\Delta(R_n)\) | \(K\Delta(R_n)\) | pass? |
|---:|---:|---:|---:|:---:|
| 1 | \(1/2\) | \(3375/64\) | \(2232375/3824\) | no |
| 2 | \(1/4\) | \(11349/512\) | \(7506733/30592\) | no |
| 3 | \(1/8\) | \(42777/4096\) | \(28294609/244736\) | no |
| 4 | \(1/16\) | \(166113/32768\) | \(109874521/1957888\) | no |
| 5 | \(1/32\) | \(654705/262144\) | \(433050985/15663104\) | no |
| 6 | \(1/64\) | \(2599569/2097152\) | \(1719470473/125304832\) | no |
| 7 | \(1/128\) | \(10360017/16777216\) | \(6852575689/1002438656\) | no |
| 8 | \(1/256\) | \(41363793/134217728\) | \(27359851081/8019509248\) | no |
| 9 | \(1/512\) | \(165302865/1073741824\) | \(109338661705/64156073984\) | no |
| 10 | \(1/1024\) | \(660907089/8589934592\) | \(437153322313/513248591872\) | no |
| 11 | \(1/2048\) | \(2643019857/68719476736\) | \(1748210800969/4105988734976\) | **yes** |
| 12 | \(1/4096\) | \(10570862673/549755813888\) | \(6992038388041/32847909879808\) | yes |
| 13 | \(1/8192\) | \(42281017425/4398046511104\) | \(27966544081225/262783279038464\) | yes |
| 14 | \(1/16384\) | \(169119203409/35184372088832\) | \(111862957543753/2102266232307712\) | yes |
| 15 | \(1/32768\) | \(676467081297/281474976710656\) | \(447445392773449/16818129858461696\) | yes |
| 16 | \(1/65536\) | \(2705848860753/2251799813685248\) | \(1789768696451401/134545038867693568\) | yes |

Thus the largest passing member of the prescribed dyadic list is

\[
\boxed{R_*=\frac1{2048}.}
\]

This is a largest-passing-list statement, not a claim that \(R_*\) is the
largest possible stability radius among all positive real numbers.  At this
radius,

\[
\Delta_*:=\Delta(R_*)
 =\frac{2643019857}{68719476736},
\]

\[
\eta_*:=K\Delta_*
 =\frac{1748210800969}{4105988734976}
 <\frac12,
\]

with exact half-margin

\[
\frac12-\eta_*
 =\frac{304783566519}{4105988734976}>0.
\]

## 4. Quantitative infinity-norm stability theorem

**Theorem 4.1.**  For every \(z,z'\in Q_{R_*}\),

\[
c_*\|z-z'\|_\infty
 \le \|\Phi(z)-\Phi(z')\|_\infty,
\]

where

\[
\boxed{
c_*:=\frac1K-\Delta_*
 =\frac{21220001406063}{409087045009408}>0,}
\]

Consequently \(\Phi\) is injective on the entire certified cube \(Q_{R_*}\),
and its inverse on \(\Phi(Q_{R_*})\) is infinity-norm Lipschitz with exact
constant

\[
\boxed{
C_*:=c_*^{-1}
 =\frac{409087045009408}{21220001406063}.}
\]

In particular, the half-margin also gives the simpler conservative bound

\[
C_*\le2K=\frac{47624}{2151}.
\]

**Proof.**  Put \(v=z-z'\) and

\[
\gamma(t)=z'+tv,\qquad 0\le t\le1.
\]

The cube \(Q_{R_*}\) is convex, so the entire segment \(\gamma([0,1])\) lies in
the region on which the exact Jacobian bound holds.  The fundamental theorem
of calculus gives

\[
B^{-1}\bigl(\widehat\Phi(z)-\widehat\Phi(z')\bigr)
 =v+\int_0^1
 B^{-1}\bigl(D\widehat\Phi(\gamma(t))-B\bigr)v\,dt.
\]

By submultiplicativity and the coefficient-sum certificate,

\[
\left\|
 \int_0^1 B^{-1}
 \bigl(D\widehat\Phi(\gamma(t))-B\bigr)v\,dt
\right\|_\infty
\le K\Delta_*\|v\|_\infty
=\eta_*\|v\|_\infty.
\]

The reverse triangle inequality, applied to the fixed vector \(v\) and this
integral error, therefore yields

\[
\left\|B^{-1}
 \bigl(\widehat\Phi(z)-\widehat\Phi(z')\bigr)
\right\|_\infty
\ge(1-\eta_*)\|v\|_\infty.
\]

Since

\[
\|B^{-1}w\|_\infty\le K\|w\|_\infty,
\]

we obtain

\[
\|\widehat\Phi(z)-\widehat\Phi(z')\|_\infty
\ge\left(\frac1K-\Delta_*\right)\|v\|_\infty
=c_*\|v\|_\infty.
\]

This is the required lower segment bound.  It does **not** use the false
inequality

\[
\left\|\int_0^1u(t)\,dt\right\|
 \ge\int_0^1\|u(t)\|\,dt;
\]

norms satisfy the opposite integral upper bound, and vector cancellation would
invalidate that argument.  Preconditioning and separating the fixed vector
\(v\) from a certified small error are essential.

Finally, \(\Phi\) and \(\widehat\Phi\) differ by a fixed translation, so their
pairwise differences agree.  The lower bound proves injectivity on the cube,
and rearranging it proves the inverse Lipschitz estimate on its image.
\(\square\)

## 5. Conditional observation-perturbation corollary

The theorem gives a precise deterministic perturbation statement, with an
important domain condition.

**Corollary 5.1.**  Let \(z,\widehat z\in Q_{R_*}\).  If

\[
\|\Phi(\widehat z)-\Phi(z)\|_\infty\le\varepsilon,
\]

then

\[
\boxed{
\|\widehat z-z\|_\infty\le C_*\varepsilon.}
\]

This is Theorem 4.1 rearranged.  Observation noise, solver error, or a
degree-five-and-higher truncation discrepancy may be inserted into
\(\varepsilon\) only conditionally on a separate bound for that discrepancy.
No such bound is proved here, and both coordinate vectors must already be in
the certified cube.

The subsequent theorem “Approximate recovery from actual responses”
(`thm:actual`) in the
[canonical manuscript](../paper/randomized_signature_identifiability.tex)
provides the separate fifth-order remainder bound for this scalar witness and
all eight experiments at sufficiently small common scale, under a known BV
total-variation bound. Its clipped decoder is defined for every observation
vector and gives explicit approximate-recovery bounds, including observation
noise and iteration residual. That additional proof resolves the corresponding
object-C approximation problem without asserting an exact full-CDE inverse.

## 6. Common-scale inverse-linearization amplification

Let \(h\ne0\) and replace the base design scales \(r_j\) by \(hr_j\).  Denote
the resulting zero-signature Jacobian by \(B(h)\).  Since a column associated
with Hall degree \(d_i\) gains the factor \(h^{d_i}\),

\[
B(h)=B(1)\operatorname{diag}
(h,h,h^2,h^3,h^3,h^4,h^4,h^4),
\]

and therefore

\[
\boxed{
B(h)^{-1}=\operatorname{diag}
(h^{-1},h^{-1},h^{-2},h^{-3},h^{-3},h^{-4},h^{-4},h^{-4})
B(1)^{-1}.}
\]

The exact absolute row sums of \(B^{-1}\), in Hall order, are

\[
\left(
\frac{139}{33},\frac{4547}{717},\frac{23812}{2151},
\frac{17552}{2151},\frac{824}{99},\frac{63644}{32265},
\frac{160}{33},\frac{31978}{32265}
\right).
\]

Thus the induced infinity norm is exactly

\[
\boxed{
\|B(h)^{-1}\|_\infty
=\max\left\{
\frac{4547}{717}|h|^{-1},
\frac{23812}{2151}|h|^{-2},
\frac{824}{99}|h|^{-3},
\frac{160}{33}|h|^{-4}
\right\}.}
\]

For \(0<|h|\le1\), the degree-four row gives the lower bound and the base
maximum row sum gives the upper bound

\[
\boxed{
\frac{160}{33}|h|^{-4}
\le\|B(h)^{-1}\|_\infty
\le\frac{23812}{2151}|h|^{-4}.}
\]

Consequently,

\[
\boxed{
\|B(h)^{-1}\|_\infty=\Theta(|h|^{-4})
\quad\text{as }h\to0.}
\]

This is an exact statement about zero-Jacobian inverse-linearization
amplification.  It is not a condition-number theorem: no product with
\(\|B(h)\|_\infty\) is being
asserted.  It is also not, by itself, a nonlinear stability certificate at the
same isotropic radius.  The nonlinear Jacobian variation changes under common
scale rescaling and would require a new exact bound or an explicitly
transformed anisotropic coordinate domain.

## 7. Assumptions and exclusions

The theorem assumes and proves exactly the following:

- two inputs, scalar state, and Hall degrees one through four only;
- the exact witness \(V_1(x)=\tanh x\), \(V_2(x)=1/2\);
- the fixed certified eight-pair design \(\mathcal D_*\);
- exact knowledge of the degree-four truncated polynomial \(\Phi\);
- the real Hall-coordinate cube \(Q_{1/2048}\);
- the induced vector and matrix infinity norms;
- exact rational coefficient sums, inverse, radius table, and constants.

The rational certificate proves inequalities for all real Hall coordinates in
the cube; it is not limited to rational coordinate vectors.

The quantitative theorem proved in this note does **not by itself** establish:

- stability or inversion of the actual full CDE response, which contains
  levels five and above;
- a remainder estimate transferring object B to object C;
- global injectivity on \(\mathbb R^8\), or any inverse statement outside the
  certified cube and its image;
- existence of a preimage for arbitrary observation noise;
- stability of a chosen path representative in a path topology;
- a parameter-uniform radius or constant for the generic shallow-tanh family;
- a finite-difference reconstruction scheme;
- a condition-number result or a nonlinear common-scale stability theorem;
- universal measurement, reservoir-run, state, or scale minimality.

In particular, the Gate-3 almost-everywhere fixed-design Jacobian theorem does
not make these Gate-4 constants uniform in the architecture parameters; the
inverse and stability neighborhood can degenerate near the analytic
determinant-zero set.

## Final status

EXACT_LOCAL_STABILITY
