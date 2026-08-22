# Eight exact observations of the degree-four truncated response

## 1. Scope and conventions

This note concerns only the degree-four truncated response (observation object
B). It does not identify that polynomial with the actual full CDE response
(object C), whose expansion contains levels at least five.

Let
\[
q=\frac{\log 3}{2},\qquad V_1(x)=\tanh x,\qquad V_2(x)=\frac12.
\]
The conventions are those proved in
research_output/graded_reconstruction_derivation.md:

- words are chronological tuples and tensor multiplication is concatenation;
- \(S^{\leq4}(z)=\pi_{\leq4}\exp_\otimes(z)\);
- \(\Gamma(i_1\cdots i_m)
  =\mathcal D_{V_{i_1}}\cdots\mathcal D_{V_{i_m}}\), where
  \(\mathcal D_Vf=f'V\);
- \(\Gamma\) is a homomorphism;
- \([X_1,X_2]=X_1X_2-X_2X_1\) corresponds to
  \([V,W]=DW\,V-DV\,W\);
- the coefficient of \(r^m\) is the normalized scale jet \(J_m\), with no
  additional factorial.

The ordered standard-Lyndon basis is
\[
\begin{aligned}
\mathcal H=\big(&X_1,X_2,[X_1,X_2],
[X_1,[X_1,X_2]],[[X_1,X_2],X_2],\\
&[X_1,[X_1,[X_1,X_2]]],
[X_1,[[X_1,X_2],X_2]],[[[X_1,X_2],X_2],X_2]\big),
\end{aligned}
\]
with degree vector
\[
d=(1,1,2,3,3,4,4,4).
\]
Its coefficients identify the step-four free-Lie coordinate space with
\(\mathbb R^8\). Write \(z=z_1+z_2+z_3+z_4\), degree by degree, and set
\[
S_m(z)=\pi_m\exp_\otimes(z).
\]

For a state-scale pair \((y,r)\), define
\[
F_y^{(4)}(r;z)
 =y+\sum_{m=1}^4r^m\Gamma(S_m(z))\operatorname{Id}(y).
\]
For an ordered eight-pair design
\[
\mathcal D=((y_1,r_1),\ldots,(y_8,r_8)),
\]
let
\[
\Phi_{\mathcal D}(z)
 =(F_{y_j}^{(4)}(r_j;z))_{j=1}^8.
\]

The literal constant terms \(y_j=\pm q\) are not rational. Exact code
therefore also uses the explicitly centered map
\[
\widehat\Phi_{\mathcal D}(z)
 :=\Phi_{\mathcal D}(z)-y_{\mathcal D},
\qquad y_{\mathcal D}=(y_1,\ldots,y_8).
\]
At the witness, every coefficient of \(\widehat\Phi_{\mathcal D}\) is
rational. The uncentered implementation records an exact scalar as
\[
(k,a)\in\{-1,0,1\}\times\mathbb Q,
\]
denoting \(kq+a\). Thus the code does not silently replace
\(F_y^{(4)}\) by its centered version. Translation by
\(y_{\mathcal D}\) changes neither the Jacobian nor local injectivity.

## 2. Polynomial observation map and its zero Jacobian

The Hall element \(z\) depends linearly on its eight coordinates, and
\[
\pi_{\leq4}\exp_\otimes(z)
 =1+z+\frac{z^2}{2}+\frac{z^3}{6}+\frac{z^4}{24}
\]
after truncation. Therefore \(\Phi_{\mathcal D}\) is a polynomial, hence
real analytic, map of the eight Hall coordinates. This statement is about
the finite tensor polynomial only; no CDE solver or remainder estimate is
involved.

Let \(h\in\mathcal H\) have degree \(m\). Since
\[
\pi_{\leq4}\exp_\otimes(\varepsilon h)
 =1+\varepsilon h+O(\varepsilon^2),
\]
differentiation at zero gives
\[
\boxed{
\partial_hF_y^{(4)}(r;0)
 =r^m\Gamma(h)\operatorname{Id}(y)
 =r^m\rho(h)(y).}
\]
Consequently,
\[
B_{jh}:=D\Phi_{\mathcal D}(0)_{jh}
 =r_j^{\deg h}\rho(h)(y_j).
\]

The implementation constructs this matrix twice in exact arithmetic. The
tensor route expands \(h\) into associative words and applies the
chronological word representation. The independent Lie route recursively
applies the certified vector-field bracket. Equality of the two
\(\operatorname{Fraction}\)-valued matrices is a required certificate check,
not a floating-point comparison.

## 3. General joint state-scale theorem

**Theorem 3.1 (joint state-scale independence).** Suppose that every
homogeneous restriction
\[
\rho_m:\mathcal L_{2,m}\longrightarrow\mathfrak X(\mathbb R),
\qquad 1\leq m\leq4,
\]
is injective. On every nonempty open interval
\(I\subset\mathbb R\setminus\{0\}\), the eight functions
\[
(r,y)\longmapsto r^{\deg h}\rho(h)(y),
\qquad h\in\mathcal H,
\]
are linearly independent on \(I\times\mathbb R\).

**Proof.** Suppose that a linear combination is zero and group its terms by
degree. With
\[
h_m=\sum_{\deg h=m}c_hh,
\]
one has
\[
\sum_{m=1}^4r^m\rho(h_m)(y)=0
\qquad\text{for every }(r,y)\in I\times\mathbb R.
\]
For each fixed \(y\), the left side is a polynomial in \(r\) that vanishes on
an open interval. Every coefficient therefore vanishes:
\(\rho(h_m)(y)=0\) for every \(y\). Homogeneous injectivity gives \(h_m=0\);
basis independence then gives every \(c_h=0\). \(\square\)

We use the following elementary finite-evaluation lemma.

**Lemma 3.2.** If \(f_1,\ldots,f_n:\Omega\to\mathbb R\) are linearly
independent scalar functions, then there exist
\(\omega_1,\ldots,\omega_n\in\Omega\) for which
\([f_k(\omega_j)]_{j,k=1}^n\) is nonsingular.

**Proof.** Let
\[
v(\omega)=(f_1(\omega),\ldots,f_n(\omega))\in\mathbb R^n.
\]
If the span of \(\{v(\omega):\omega\in\Omega\}\) were proper, a nonzero
linear functional annihilating that span would give a nontrivial linear
relation among the \(f_k\). Hence those row vectors span \(\mathbb R^n\), so
eight of them form a basis when \(n=8\). \(\square\)

Applying the lemma to Theorem 3.1 supplies eight points of
\(I\times\mathbb R\) whose scalar evaluation matrix is nonsingular. By
Section 2, those evaluations are precisely the rows of a zero-signature
Jacobian. Thus a dimension-matched eight-observation design exists before
any grid restriction is imposed.

## 4. Prescribed exact search

The primary candidate set is ordered first by state and then by scale:
\[
y\in\{-q,0,q\},\qquad r\in\{1,2,3,4\}.
\]
There are twelve rows and
\(\binom{12}{8}=495\) eight-row subsets. Every determinant is computed over
\(\mathbb Q\). Exactly \(486\) subsets have nonzero determinant, so the
prescribed signed fallback grid is not used.

The phrase “simpler scales” is made deterministic as follows. For a design
set define
\[
C(\mathcal D)
 =\left(
   \#\{|r_j|:1\leq j\leq8\},
   \operatorname{sort}\{|r_j|:1\leq j\leq8\}
  \right),
\]
where the second component contains distinct magnitudes. Designs are first
ordered by \(C(\mathcal D)\). Only among designs already certified by an
exact nonzero determinant is the binary64 two-norm condition number used.
Remaining ties are resolved by the lexicographic ordered-pair list in the
authoritative state and scale order.

This rule selects
\[
\boxed{
\mathcal D_*=((-q,1),(-q,2),(-q,3),(0,2),(0,3),
               (q,1),(q,2),(q,3)).}
\]
In the Hall order of Section 1, its exact zero Jacobian is
\[
B_*=
\begin{pmatrix}
-\frac12&\frac12&-\frac38&\frac{15}{32}&\frac3{16}&-\frac{57}{128}&-\frac3{32}&\frac3{64}\\
-1&1&-\frac32&\frac{15}{4}&\frac32&-\frac{57}{8}&-\frac32&\frac34\\
-\frac32&\frac32&-\frac{27}{8}&\frac{405}{32}&\frac{81}{16}&-\frac{4617}{128}&-\frac{243}{32}&\frac{243}{64}\\
0&1&-2&4&0&-8&0&4\\
0&\frac32&-\frac92&\frac{27}{2}&0&-\frac{81}{2}&0&\frac{81}{4}\\
\frac12&\frac12&-\frac38&\frac{15}{32}&-\frac3{16}&-\frac{57}{128}&\frac3{32}&\frac3{64}\\
1&1&-\frac32&\frac{15}{4}&-\frac32&-\frac{57}{8}&\frac32&\frac34\\
\frac32&\frac32&-\frac{27}{8}&\frac{405}{32}&-\frac{81}{16}&-\frac{4617}{128}&\frac{243}{32}&\frac{243}{64}
\end{pmatrix}.
\]
Exact elimination gives
\[
\boxed{\det B_*=-\frac{258733035}{524288}\ne0.}
\]

The exact inverse persisted in the certificate is
\[
\small
B_*^{-1}=
\begin{pmatrix}
-18/11&9/22&-2/33&0&0&18/11&-9/22&2/33\\
360/239&126/239&-88/717&-405/239&80/239&360/239&126/239&-88/717\\
-76/239&547/239&-964/2151&-990/239&1760/2151&-76/239&547/239&-964/2151\\
-760/717&1168/717&-664/2151&-432/239&256/717&-760/717&1168/717&-664/2151\\
-76/33&52/33&-28/99&0&0&76/33&-52/33&28/99\\
-3056/10755&476/1195&-2992/32265&-1244/3585&808/10755&-3056/10755&476/1195&-2992/32265\\
-40/33&32/33&-8/33&0&0&40/33&-32/33&8/33\\
-472/10755&646/3585&-2264/32265&-1018/3585&1276/10755&-472/10755&646/3585&-2264/32265
\end{pmatrix}.
\]
Both products with \(B_*\) are the exact identity. Since the eight rows of an
invertible matrix are independent, deleting any one observation leaves exact
row rank seven. As a separate singular regression design, the four scales at
\(-q\) together with the four scales at \(0\) have determinant zero and rank
seven.

The non-certifying binary64 diagnostics for \(B_*\) are
\[
\begin{aligned}
\sigma(B_*)\approx{}&(72.8678806904,13.4576527068,12.4585453832,
2.91100080890,\\
&1.23631176424,0.365987026873,0.197225996535,
0.155492337857),\\
\kappa_2(B_*)&\approx468.626825569.
\end{aligned}
\]
These numbers select among exact certificates; they do not certify rank.

## 5. Fixed-design genericity

Keep \(\mathcal D_*\) fixed, including the states \(-q,0,q\). For
\[
\theta=(a_1,b_1,a_2,b_2)\in\mathbb R^4,
\qquad V_i^\theta(x)=\tanh(a_ix+b_i),
\]
define
\[
\Delta_{\mathcal D_*}(\theta)
 :=\det\left[
 r_j^{\deg h}\rho_\theta(h)(y_j)
 \right]_{j,h}.
\]
Every entry is real analytic in \(\theta\): finite iterated brackets use only
finite sums, products, and state derivatives of the real-analytic tanh fields.
Therefore \(\Delta_{\mathcal D_*}\) is real analytic. At
\[
\theta_*=(1,0,0,q)
\]
its exact value is \(-258733035/524288\ne0\), so it is not identically zero.
The zero set of a nontrivial real-analytic function on connected
\(\mathbb R^4\) has Lebesgue measure zero. Hence the same fixed design has an
invertible zero Jacobian for Lebesgue-almost every parameter vector, and
almost surely under every parameter law absolutely continuous with respect
to four-dimensional Lebesgue measure.

This conclusion uses the fixed determinant \(\Delta_{\mathcal D_*}\)
directly. Generic homogeneous injectivity alone would not prove that one
fixed finite design works generically.

## 6. Local analytic inverse for object B

At the witness, \(\Phi_{\mathcal D_*}\) is polynomial in \(z\) and
\[
D\Phi_{\mathcal D_*}(0)=B_*
\]
is invertible. The real-analytic inverse function theorem therefore gives
open neighborhoods
\[
0\in U\subset\mathbb R^8,
\qquad
\Phi_{\mathcal D_*}(0)\in W\subset\mathbb R^8,
\]
such that
\[
\Phi_{\mathcal D_*}|_U:U\longrightarrow W
\]
is a real-analytic diffeomorphism. Thus observations in \(W\) have one
preimage in \(U\), depending real analytically on the observations. This is
local uniqueness only. It proves neither global injectivity nor recovery from
the actual full CDE response.

The same statement holds for the centered map after translating \(W\) by
\(-y_{\mathcal D_*}\).

## 7. Dimension lower bound

**Theorem 7.1.** Let \(U\subset\mathbb R^8\) be open with \(0\in U\). If
\(M<8\), no continuous map \(G:U\to\mathbb R^M\) is locally injective at
zero, where locally injective means injective on some open neighborhood of
zero.

**Proof.** Suppose \(G\) is injective on an open neighborhood
\(U_0\subset U\) of zero. Let
\[
\iota:\mathbb R^M\hookrightarrow\mathbb R^8,
\qquad \iota(x)=(x,0,\ldots,0).
\]
Then \(\iota\circ G:U_0\to\mathbb R^8\) is continuous and injective.
Brouwer's invariance-of-domain theorem says its image is open in
\(\mathbb R^8\). But that image lies in the \(M\)-dimensional coordinate
plane \(\iota(\mathbb R^M)\), which has empty interior when \(M<8\), a
contradiction. \(\square\)

Thus eight scalar observations match the topological dimension in this
specific model: continuous local injectivity on an open eight-dimensional
Hall-coordinate domain. This is not universal measurement minimality,
reservoir-run minimality, scale minimality, or a statement about terminal
observations of the full CDE.

## 8. Common scale rescaling

For \(h\ne0\), replace every selected pair \((y_j,r_j)\) by
\((y_j,hr_j)\). A column belonging to a degree-\(m\) Hall element is
multiplied by \(h^m\). Since the degree multiplicities are \(2,1,2,3\),
\[
\sum_{a=1}^8\deg H_a
 =2\cdot1+1\cdot2+2\cdot3+3\cdot4
 =22.
\]
Therefore
\[
\boxed{\det B_*(h)=h^{22}\det B_*(1).}
\]
This is an algebraic identity. The restriction \(h\ne0\) is needed to retain
nonsingularity.

## 9. Cross-degree scope boundary at the witness

Put \(u=\tanh y\). Direct use of
\([V,W]=DW\,V-DV\,W\) gives the three odd represented fields
\[
u,\qquad -\frac{u(1-u^2)}2,\qquad u^3(1-u^2),
\]
and the five even fields
\[
\frac12,\quad -\frac{1-u^2}{2},\quad
\frac{1-u^4}{2},\quad
-\frac{(1-u^2)(1+3u^4)}2,\quad
\frac{(1-u^2)(1-3u^2)}4.
\]
The odd fields have successive highest powers \(u,u^3,u^5\), hence rank
three. The first four even fields have successive highest even powers through
\(u^6\), hence rank four. All five even fields satisfy the unique relation
\[
\boxed{
\rho\left(
4[X_1,X_2]+3[X_1,[X_1,X_2]]
+2[[[X_1,X_2],X_2],X_2]
\right)=0.}
\]
Indeed, the three fields appearing in this relation are, coefficientwise as
polynomials in \(u\),
\[
\frac{u^2-1}{2},\qquad
\frac{1-u^4}{2},\qquad
\frac{(1-u^2)(1-3u^2)}4,
\]
and
\[
4\frac{u^2-1}{2}
+3\frac{1-u^4}{2}
+2\frac{(1-u^2)(1-3u^2)}4
=0.
\]
Thus the eight represented fields have cross-degree function rank exactly
seven. Independently, the exact normalized Taylor matrix at \(y=0\), using
orders zero through six, has a seven-column minor of determinant
\(-3/32\ne0\), which supplies the rank lower bound. The displayed polynomial
identity supplies the rank upper bound.

At one fixed scale \(r_0\ne0\), the eight columns are multiplied by the
invertible diagonal matrix
\[
\operatorname{diag}(r_0,r_0,r_0^2,r_0^3,r_0^3,
r_0^4,r_0^4,r_0^4).
\]
Their rank remains seven. Equivalently, the relation coefficients at scale
\(r_0\) are \(4r_0^{-2},3r_0^{-3},2r_0^{-4}\) in the three displayed
directions. Hence every all-at-one-scale eight-observation Jacobian is
singular at the witness, and every full-rank design must use at least two
distinct scale values. This is precisely why homogeneous injectivity must not
be promoted to ungraded injectivity.

## 10. Standard results used

The proof invokes three standard results outside the repository's exact
algebra:

1. the finite-dimensional real-analytic inverse function theorem;
2. the fact that the zero set of a nonzero real-analytic function on a
   connected open subset of \(\mathbb R^n\) has Lebesgue measure zero;
3. Brouwer's invariance-of-domain theorem.

The exhaustive grid calculation, determinant, inverse, two independent
Jacobian constructions, common-scale law, coefficientwise polynomial
relation, and Taylor-minor rank lower bound are reproduced by
research_output/reproduce_finite_observation.py. All certifying calculations
are exact.

## 11. Exact conclusion and limitations

The established result is:

- eight explicitly selected scalar state-scale evaluations give an exact
  nonsingular zero Jacobian for the degree-four truncated response at the
  tanh witness;
- the same fixed design works for almost every shallow-tanh parameter vector;
- the truncated polynomial map has a local real-analytic inverse near zero;
- fewer than eight scalar outputs cannot be continuously locally injective on
  an open eight-dimensional Hall-coordinate neighborhood;
- common rescaling obeys the exact exponent-22 determinant law.

No claim is made about global inversion, an explicit inverse-neighborhood
radius, finite-difference reconstruction, the full CDE response, levels at
least five, or universal measurement minimality.

**Final status: PROVED.**
