# Fixed-state Hall rank and finite-degree recovery for shallow tanh reservoirs

Status: **VERIFIED within the observation models stated below**.

## 1. Motivation and current contribution

For all positive integers \(d,D,L\), denoting input dimension, state
dimension and degree, shallow componentwise biased-tanh fields generically have fixed-zero
Hall evaluation rank \(\min(D,\nu_{d,L})\), where \(\nu_{d,L}\) is the
number of Hall coordinates through degree \(L\). Consequently a single
zero-initialized vector of the **degree-\(L\) truncated polynomial response**
locally determines all Hall coordinates when \(D\geq\nu_{d,L}\). This width
is sharp for continuous local recovery on an open Hall-coordinate domain
from one vector. The [authoritative Hall-rank note](intrinsic_hall_rank_theorem.md)
contains the complete proof and precise attribution.

The contribution is the explicit finite-parameter tanh realization and
its exact minors at a prescribed Hall-degree cutoff. Generic bracket
generation, witness-based analytic genericity, and the scalar constant/tanh
pair have prior uses described in Section 8. Neither the fixed initial
state alone nor the standard implication from rank to a local inverse
is presented as a new principle.

The existing scalar two-input package remains a separate exact base case:
homogeneous injectivity through degree four, ideal graded reconstruction,
the designed eight-output truncated inverse, and a rational local stability
certificate. A proved remainder bound and clipped decoder additionally give
approximate recovery from those eight actual CDE responses. The separate
sufficient-width theorem proves independence of the larger word-response
family at \(D\geq\sum_{m=0}^L d^m\).

These results concern the continuous-time vector-field representation and
the explicitly stated observation objects. The motivating finance
experiments do not supply their proofs, and the theorems do not establish
properties of the implemented discrete recurrence or explain financial
performance.

## 2. Setup and the three observation objects

Let

\[
V_i^\theta(x)=\tanh(A_i x+b_i),\qquad
\theta\in\mathbb R^{d(D^2+D)},\qquad i=1,\ldots,d,
\]

and define the free-Lie representation

\[
\rho:\mathcal L_d\longrightarrow\mathfrak X(\mathbb R^D),
\qquad \rho(X_i)=V_i,
\]

with bracket convention

\[
[V,W](x)=DW(x)V(x)-DV(x)W(x).
\]

Here \(A_i\in\mathbb R^{D\times D}\), \(b_i\in\mathbb R^D\), tanh
acts componentwise, and \(D_Vf=Df\,V\). Words are chronological, tensor
multiplication is concatenation, and
\(\Gamma(i_1\cdots i_m)=D_{V_{i_1}}\cdots D_{V_{i_m}}\)
has its first letter outermost. Thus \(\Gamma(uv)=\Gamma(u)\Gamma(v)\)
and \(\Gamma(H)\operatorname{Id}=\rho(H)\). The nonlinear representation
is defined on the full free Lie algebra; restriction by degree does not
assert factorization through its nilpotent quotient.

Write \(\rho_m=\rho|_{\mathcal L_{d,m}}\). For a continuous
bounded-variation input \(X\), the scaled response always means the response
to the scaled input path,

\[
Y^{y,r}(X):=Y^y(rX),
\qquad
dY_t^{y,r}
=r\sum_{i=1}^dV_i(Y_t^{y,r})\,dX_t^i,
\qquad Y_0^{y,r}=y.
\]

No state-rescaling identity is used.

Here \(\ell(z)\) denotes the Lie polynomial with Hall-coordinate vector
\(z\), ordered by degree and then Lyndon word.

Three observation objects must remain distinct:

- **A. Ideal normalized graded scale jets**
  \[
  J_m(y;X)
  =\frac1{m!}\left.\partial_r^mY_T^{y,r}(X)\right|_{r=0}.
  \]
- **B. The degree-\(L\) truncated finite-scale response polynomial**
  \[
  F_y^{\leq L}(r;z)=y+\sum_{m=1}^L r^m
  \Gamma\bigl(\pi_m\exp_\otimes\ell(z)\bigr)\operatorname{Id}(y),
  \qquad z\in\mathbb R^{\nu_{d,L}}.
  \]
  Truncation is by tensor word length, equivalently Hall weighted degree.
  The scalar specialization \(F_y^{(4)}\) uses eight Hall coordinates.
- **C. The actual full CDE response** \(Y_T^y(rX)\), which also contains
  levels above \(L\).

The ideal graded decoder concerns A. The fixed-state Hall-rank corollary and
the scalar eight-output inverse concern B. For the scalar witness and its
eight designed experiments only, the actual-response theorem bounds the
difference between B and C and gives approximate recovery; it does not
identify these objects exactly at a fixed nonzero scale.

## 3. Sharp fixed-zero Hall rank and local truncated recovery

Put \(\nu=\nu_{d,L}\). Let \(H_1,\ldots,H_\nu\) be the existing
degree-ordered Lyndon/Hall basis,
\(m_a=\deg H_a\), and
\[
H_\theta=[\rho_\theta(H_1)(0)\ \cdots\ \rho_\theta(H_\nu)(0)].
\]
Outside the zero set of a nontrivial real-analytic minor in the **full**
parameter space, \(\operatorname{rank}H_\theta=\min(D,\nu)\). This is an
open dense full-measure parameter set; every jointly absolutely continuous
parameter law gives it probability one. Singular bias-free, tied or sparse
sampling laws are not covered automatically.

At width \(\nu\), identify the generators \(e_i\) with their Hall-coordinate
vectors. The finite witness uses the grading derivation
\(\mathcal E|_{\mathfrak g_m}=mI\) on the free step-\(L\) Lie algebra and
\[
\Lambda_a=\mathcal E^{-1}\operatorname{ad}_a\mathcal E,\qquad
V_i(x)=\tanh(qe_i-\Lambda_{e_i}x),\qquad q=\operatorname{atanh}(1/2).
\]
The classical affine action \(W_a(x)=a/2-\Lambda_ax\) preserves brackets
with the repository sign. A finite weighted-jet argument transfers its
leading Hall evaluations to the actual tanh fields: a length-\(m\) Hall
bracket has no component above grade \(m\), and its grade-\(m\) component
is \(H/2\). At \(D=\nu\), the Hall matrix is block upper triangular
with diagonal blocks \(I/2\), so its determinant is \(2^{-\nu}\).
Lower-grade off-diagonal entries can be nonzero.

For \(D<\nu\), a graded ideal quotient retains all lower grades and the
required prefix of the final retained grade; this includes \(D<d\).
For \(D>\nu\), inactive coordinates embed the witness. The selected
\(\min(D,\nu)\)-square minor is exactly \(2^{-\min(D,\nu)}\). The
construction uses finite admissible weights and biases, not a limiting
parameter or a numerical rank sample.

For every fixed \(r\ne0\),
\[
D_zF_0^{\leq L}(r;0)
=H_\theta\operatorname{diag}(r^{m_1},\ldots,r^{m_\nu}).
\]
On the stated generic parameter set with \(D\geq\nu\), the analytic
inverse-function theorem gives a local inverse of the selected \(\nu\)
output coordinates near \(z=0\); for
\(D=\nu\) these are the whole vector. Invariance of domain excludes
continuous local injectivity into \(D<\nu\) outputs on an open
\(\nu\)-dimensional domain. This proves width sharpness for this
one-vector truncated observation model. It gives no uniform inverse
neighborhood, conditioning bound, linear readout, or exact full-CDE
endpoint decoder.

### Relationship to the scalar proof dependencies

The scalar shallow-tanh theorem supplies an explicit generic family
satisfying the homogeneous-injectivity premise. From that premise, the two
central consequences are siblings:

~~~text
homogeneous Lie injectivity
├── ideal graded reconstruction (object A)
└── joint state-scale independence
    └── explicit eight-output design (object B)
        └── exact zero Jacobian B_*
            ├── local analytic inverse
            └── quantitative stability from exact B_*^{-1} and Delta(R)
~~~

Joint state-scale independence follows directly by separating powers of the
scale variable; it is not deduced from the graded reconstruction theorem.
Likewise, quantitative stability is proved by an exact Jacobian-variation
estimate and a segment argument, not by the analytic inverse function theorem.
Every finite-scale claim in the lower branch concerns B only.
The actual scalar-response extension then combines this quantitative
certificate with a fifth-order remainder bound. The new fixed-state theorem
uses its own graded witness and local inverse argument; it does not inherit
the scalar constants or observation design.

## 4. Scalar homogeneous injectivity and its exact witness

The scalar results in Sections 4–7.1 specialize to \(d=2,D=1,L=4\), with
\(\theta=(a_1,b_1,a_2,b_2)\) and \(V_i(x)=\tanh(a_ix+b_i)\).

> **Homogeneous shallow-tanh theorem.** For Lebesgue-almost every
> \(\theta\in\mathbb R^4\), the restrictions
> \[
> \rho_m:\mathcal L_{2,m}\longrightarrow\mathfrak X(\mathbb R)
> \]
> are simultaneously injective, separately for \(m=1,2,3,4\).

Equivalently, the four homogeneous restrictions are simultaneously injective
almost surely under every parameter law absolutely continuous with respect to
four-dimensional Lebesgue measure. This is not injectivity on the ungraded
direct sum.

Set

\[
q=\frac{\log 3}{2}=\operatorname{atanh}(1/2),\qquad
\theta_*=(1,0,0,q).
\]

Then

\[
V_1(x)=\tanh x,\qquad V_2(x)=\frac12,\qquad
\tanh(\pm q)=\pm\frac12.
\]

The deterministic standard-Lyndon basis blocks are, degree by degree,
\((X_1,X_2)\), \(([X_1,X_2])\),
\(([X_1,[X_1,X_2]],[[X_1,X_2],X_2])\), and
\(([X_1,[X_1,[X_1,X_2]]],[X_1,[[X_1,X_2],X_2]],
[[[X_1,X_2],X_2],X_2])\). In that order, the exact homogeneous evaluation
matrices are

\[
M_1\big|_{(0,q)}
=\begin{pmatrix}
0&1/2\\
1/2&1/2
\end{pmatrix},
\qquad
M_2\big|_{(0)}
=\begin{pmatrix}-1/2\end{pmatrix},
\]

\[
M_3\big|_{(-q,q)}
=\begin{pmatrix}
15/32&3/16\\
15/32&-3/16
\end{pmatrix},
\]

and

\[
M_4\big|_{(-q,0,q)}
=\begin{pmatrix}
-57/128&-3/32&3/64\\
-1/2&0&1/4\\
-57/128&3/32&3/64
\end{pmatrix}.
\]

Their exact determinants are

\[
-\frac14,\qquad
-\frac12,\qquad
-\frac{45}{256},\qquad
\frac{135}{8192},
\]

so

\[
\prod_{m=1}^4\det M_m
=-\frac{6075}{16777216}\ne0.
\]

The finite-evaluation lemma turns each nonzero determinant into functional
linear independence at the witness. The determinant product is real analytic
in \(\theta\) and nonzero at \(\theta_*\), so its zero set is Lebesgue-null.
This gives the almost-everywhere theorem. Exact rational activation jets are
the proof path; floating-point atlas and autodifferentiation results are
non-certifying context only.

At the witness, the eight represented Hall fields are not independent across
degrees. Their exact relation is

\[
\rho\!\left(
4[X_1,X_2]
+3[X_1,[X_1,X_2]]
+2[[[X_1,X_2],X_2],X_2]
\right)=0,
\]

and their joint cross-degree function rank is exactly seven. Homogeneous
injectivity therefore must not be restated as ungraded injectivity.

## 5. Sibling consequence A: ideal graded reconstruction

For a degree-four log-signature

\[
\ell=\ell_1+\ell_2+\ell_3+\ell_4,
\qquad
S^{\le4}=\exp_\otimes(\ell),
\]

the Chen--Fliess expansion with the fixed repository word/operator convention
gives the exact triangular identities

\[
J_m(y;X)
=\rho(\ell_m)(y)
+\Psi_m(y;\ell_1,\ldots,\ell_{m-1}),
\qquad m=1,2,3,4.
\]

If each \(\rho_m\) is injective, finitely many state evaluations separate its
finite-dimensional image. Hence the Hall-coordinate blocks are recovered
recursively:

\[
\lambda_m
=M_m^{-1}
\bigl(\mathbf J_m-\boldsymbol\Psi_m
      (\lambda_1,\ldots,\lambda_{m-1})\bigr).
\]

The same criterion holds for arbitrary finite \(d,D,L\), assuming
homogeneous injectivity on the observation domain and the stated smooth
local-solution hypotheses; see [general graded identifiability](general_graded_identifiability.md).
The scalar shallow-tanh theorem supplies one concrete generic
architecture to which it applies, and the displayed witness matrices provide
an exact decoder. The result uses ideal derivatives at \(r=0\); it is not a
claim that one terminal state or finitely many ordinary full-CDE values recover
the log-signature.

The full convention check, tensor-exponential formulas, every
\(\Psi_m\), formal converse, and exact recursive decoder are in
[the graded reconstruction derivation](graded_reconstruction_derivation.md)
and its
[exact certificate](graded_reconstruction_certificate.json).

## 6. Sibling consequence B: finite state-scale observation

For a Hall element \(h\) of homogeneous degree \(m\), the differential at
zero of the degree-four truncated response satisfies

\[
\partial_hF_y^{(4)}(r;0)=r^m\rho(h)(y).
\]

If a linear combination over all Hall elements vanishes for every \((r,y)\),
then it has the form

\[
\sum_{m=1}^4r^mf_m(y)=0.
\]

The polynomial identity in \(r\) forces every \(f_m\) to vanish; homogeneous
injectivity then forces every Hall coefficient to vanish. This proves joint
state-scale independence directly from homogeneous injectivity.

For the witness, the certified ordered design is

\[
\mathcal D_*=
\bigl(
(-q,1),
(-q,2),
(-q,3),
(0,2),
(0,3),
(q,1),
(q,2),
(q,3)
\bigr).
\]

Let \(\Phi=\Phi_{\mathcal D_*}:\mathbb R^8\to\mathbb R^8\) be the corresponding
degree-four truncated observation polynomial. Its exact zero-signature
Jacobian \(B_*=D\Phi(0)\) has

\[
\det B_*=-\frac{258733035}{524288}\ne0.
\]

The same fixed states and scales give an invertible zero Jacobian for
Lebesgue-almost every shallow-tanh parameter vector, because the fixed-design
determinant is a nontrivial real-analytic function of \(\theta\). The local
inverse neighborhood may depend on \(\theta\) and can degenerate near its
determinant-zero set.

The analytic inverse function theorem gives neighborhoods \(U\ni0\) and
\(W\ni\Phi(0)\) such that
\(\Phi|_U:U\to W\) is a real-analytic diffeomorphism. This is local, not
global.

Eight scalar outputs match the Hall dimension

\[
2+1+2+3=8.
\]

If \(U\subset\mathbb R^8\) is open and \(M<8\), no continuous map
\(U\to\mathbb R^M\) can be locally injective. This is a dimension lower bound
for this continuous local observation model, not universal measurement
minimality.

At this scalar witness, the rank-seven cross-degree relation makes the zero
Jacobian of every all-at-one-nonzero-scale design singular. Multiple scales
are therefore necessary for this nonsingular-Jacobian construction; this
does not by itself disprove nonlinear injectivity. Under a common nonzero
multiplier \(h\),

\[
\det B(h)=h^{22}\det B(1),
\]

because the degree multiplicities are \(2,1,2,3\). The exact design and
Jacobian proof are in
[the finite-observation theorem](finite_observation_theorem.md) and its
[certificate](finite_observation_certificate.json).

## 7. Quantitative local stability for object B

The quantitative result uses the exact inverse \(B_*^{-1}\) and the exact
coefficient-sum majorant

\[
\Delta(R)\ge
\sup_{\|z\|_\infty\le R}
\|D\Phi(z)-B_*\|_\infty.
\]

It does not follow from the qualitative inverse function theorem. For the
witness and fixed design,

\[
R_*=\frac1{2048},\qquad
\|B_*^{-1}\|_\infty=\frac{23812}{2151},
\]

\[
c_*=\frac{21220001406063}{409087045009408},
\qquad
C_*=\frac{409087045009408}{21220001406063}.
\]

For all \(z,z'\) in the closed cube

\[
\|z\|_\infty,\ \|z'\|_\infty\le\frac1{2048},
\]

the exact segment estimate gives

\[
\|\Phi(z)-\Phi(z')\|_\infty
\ge c_*\|z-z'\|_\infty,
\]

and therefore

\[
\|z-z'\|_\infty
\le C_*\|\Phi(z)-\Phi(z')\|_\infty.
\]

The radius \(R_*\) is the largest passing member of the prescribed dyadic
certificate list under the chosen coefficient-sum majorant. It is a
conservative certificate, not a sharp or maximal stability radius. These
constants are witness-specific, not parameter-uniform.

For the common-scale design, the exact inverse row-sum identity gives

\[
\|B(h)^{-1}\|_\infty
=\Theta(|h|^{-4})
\qquad\text{as }h\to0.
\]

This is inverse-linearization amplification, not exact condition-number
scaling and not a full-CDE truncation-versus-conditioning theorem. The complete
bound and exact radius table are in
[the local-stability theorem](local_stability_theorem.md) and its
[certificate](local_stability_certificate.json).

### 7.1. Approximate recovery from the eight actual scalar responses

The canonical manuscript's actual-response theorem uses the same scalar
fields, Hall ordering and design. Let \(X_0=0\) be continuous and of bounded
variation, with \(\operatorname{TV}(X^1)+\operatorname{TV}(X^2)\leq M\).
Set
\[
D_h=\operatorname{diag}(h,h,h^2,h^3,h^3,h^4,h^4,h^4),\qquad
(c_1,c_2,c_3,c_4)=(1,1,1,13/12).
\]
For \(0<h\leq1\), require \(c_m(hM)^m\leq R_*\) for all four degrees.
For \(M>0\), \(h\leq\min(1,R_*/M)\) suffices; if \(M=0\), any such
\(h\) is allowed. The required cube condition is on \(D_hz(X)\), not on
the unscaled coordinates. The full polynomial identity is
\(\Phi_h(z)=\Phi(D_hz)\).

The actual centered responses \(b_h(X)_j=Y_T^{y_j,hr_j}(X)-y_j\) obey
\[
\|b_h(X)-\Phi(D_hz(X))\|_\infty\leq3402h^5M^5.
\]
This uses a finite integral remainder, the global word-field bound
\(K_5=1680\), and \(\max_j|r_j|=3\), giving \(3^5K_5/5!=3402\).
Let \(K=\|B_*^{-1}\|_\infty\), \(Q_{R_*}=[-R_*,R_*]^8\), and
\[
\kappa=K\Delta(R_*)=
\frac{1748210800969}{4105988734976}<\frac12.
\]
Coordinatewise clipping onto the cube makes
\[
T_b(u)=\Pi_{Q_{R_*}}[u+B_*^{-1}(b-\Phi(u))]
\]
a contraction for every \(b\in\mathbb R^8\). Its fixed point \(G(b)\)
agrees with the inverse on \(\Phi(Q_{R_*})\) and is
\(K/(1-\kappa)=C_*\)-Lipschitz. If observed responses satisfy
\(b=b_h(X)+e\), \(\|e\|_\infty\leq\delta\), then
\[
\|D_h^{-1}G(b)-z(X)\|_\infty
\leq C_*(3402M^5h+\delta h^{-4}).
\]
A computed iterate \(u_n\in Q_{R_*}\) adds
\(h^{-4}r_n/(1-\kappa)\), with exact fixed-point residual
\(r_n=\|u_n-T_b(u_n)\|_\infty\). This residual differs from response-fit
error and from floating-point roundoff. Here \(\delta\) means response
error, not the finance simulation's observation-noise parameter.

The proof, deterministic exact-flow validation and precision checks are
recorded in the [canonical manuscript](../paper/randomized_signature_identifiability.tex)
and the [actual-response audit addendum](paper_mathematical_audit.md#2026-09-08-approximate-recovery-from-eight-actual-cde-responses).
This is approximate recovery from eight designed responses, not exact
recovery at fixed scale, a global inverse, or recovery from one scalar
zero-initialized endpoint. Its constants do not transfer to Section 3's map.

### 7.2. Separate sufficient-width word-response independence

For every finite \(d,L\) and
\[
D\geq D_0(d,L)=\sum_{m=0}^L d^m,
\]
the [general sufficient-width theorem](general_shallow_tanh_width_theorem.md)
proves generic joint independence of all nonempty word-response fields
through length \(L\). The fixed-zero corollary in the
[canonical manuscript](../paper/randomized_signature_identifiability.tex)
(label \(\texttt{cor:zero}\)) proves independence of their evaluations at
zero in the full biased parameter space. There are
\(Q_{d,L}=\sum_{m=1}^L d^m\) such words. The proof transfers the classical
tensor system to finite nonzero-epsilon tanh fields and retains its
nonempty-word minor. The known empty coefficient is excluded.

This word-family conclusion implies joint Hall field independence and then
homogeneous injectivity. The width-\(\nu\) Hall theorem does not prove
independence of all \(Q_{d,L}\) word responses. At \(d=2,L=4\), the
respective counts are \(\nu=8\), \(D_0=31\), and \(Q=30\). No minimal
width is claimed for the larger word family.

## 8. Related work and attribution

Free-Lie/Hall--Lyndon algebra, signatures, tensor exponential and logarithm,
finite scale expansions, analytic genericity and local inverse methods are
classical. The grading-derived affine construction is attributed to
[Burde, Section 3.1, Proposition 3.3 and Corollaries 3.4–3.5](https://homepage.univie.ac.at/dietrich.burde/papers/burde_24_pre_lie.pdf).
The complete derivation, repository sign check and finite-jet shallow-tanh
realization are in the authoritative Hall-rank note.

[Cuchiero, Larsson and Teichmann, arXiv v2](https://arxiv.org/pdf/1908.07838v2),
Theorem 5.1 and Lemma 6.1, use nonzero minors of Lie-field evaluations
and analytic zero sets for generic interpolation, assuming an analytic
family with a specialization generating all polynomial vector fields.
Corollary 5.4 and Example 5.5 allow randomized mixtures with linear and
quadratic activations, including tanh mixtures. This directly precedes the
genericity method; it is not a fixed-tanh prescribed-degree Hall theorem.
[Monzio Compagnoni et al., arXiv v3](https://arxiv.org/pdf/2201.00384v3),
Definition II.5 and Theorem IV.1, use the same shallow biased architecture;
the proof asserts generic Hörmander spanning for a Brownian-density result.
It does not specify maximal Hall rank at each finite cutoff. Its broad
analytic-activation statement is not used as a premise here.
[Grong and Vega-Molino, arXiv v3](https://arxiv.org/html/2403.08090v3),
Remark 3.2, explicitly use the scalar constant/tanh pair and its polynomial
brackets for landmark controllability. The additional scalar content here
is the homogeneous degree-four certificate, exact mixed-degree relation
and state-scale decoder, not the generator pair itself.

[Akyildirim et al. (2022)](https://arxiv.org/abs/2201.02441) give the
randomized-signature architecture and a broad reconstruction statement;
its activation assumptions do not directly supply the tanh theorem here.
[Glückstad, Muça Cirone and Teichmann (2025), version 1](https://arxiv.org/html/2502.03163v1)
study word/tree independence and reconstruction using scales and selected
initial states, and discuss direct log-signature recovery in Section 5.
Direction-resolved projected tree terms must be distinguished from complete
summed fields: dependence among the former does not refute their complete
reconstruction theorem. No depth-two width comparison is used as a premise.
[Bailleul and Diehl, version 3 (14 November 2014)](https://arxiv.org/pdf/1407.2768v3)
supply prior reconstruction-matrix methodology for inverse rough CDEs.
Their Theorem 6 (page 7) includes one-state observations but concerns
small-time rough-flow reconstruction using generators and degree-two
brackets, rather than this all-degree shallow-tanh witness.
[Biagini, Gonon and Walter, version 2 (6 September 2024)](https://arxiv.org/pdf/2406.10214v2)
give an injective finite-datastream memory block (Lemma 7) and a terminal
increment linear-readout approximation theorem for continuous functions
of the preterminal history \(x_1,\ldots,x_{T-1}\) (Proposition 9), under their
structured discrete-time Sampling Scheme 4 and Assumption 6. These PDF
locators concern a different observation and sampling model; the local
nonlinear Hall inverse does not imply their linear-readout conclusion.

The [authoritative note](intrinsic_hall_rank_theorem.md#6-relationship-to-the-existing-results-and-sources)
records the narrow, checked comparison. The remaining result is the explicit
tanh realization with exact prescribed-degree Hall minors and the local
truncated inverse at the dimension threshold. Neither failure to locate a
matching statement nor a qualification of an earlier proof establishes
worldwide priority. The established methods are not presented as new.

## 9. Exact scope and current status

The general rank theorem covers arbitrary finite \(d,D,L\) at fixed zero
initialization in the full biased-tanh parameter family. Local recovery of
an unrestricted open Hall-coordinate neighborhood from one truncated vector
has sharp threshold \(D=\nu_{d,L}\). The scalar eight-query design remains
useful in its different observation model, with its own explicit constants
and proved approximate actual-response bound.

The package does not establish exact recovery from an actual full-CDE
endpoint at fixed nonzero scale, global inversion, linear-readout
approximation from the rank theorem, parameter-uniform stability, or a
conditioning estimate for the general Hall witness. It does not extend the
specific scalar homogeneous certificate beyond degree four. The general
word-family threshold remains separate. No discrete-time, rough-path or
financial-performance consequence is inferred from these results.

## 10. Detailed sources and artifacts

The proof and finite exact checks are separated in the source files:

- [the authoritative fixed-zero Hall-rank theorem](intrinsic_hall_rank_theorem.md)
  and [finite witness certificate](intrinsic_hall_rank_certificate.json);
- [general graded identifiability](general_graded_identifiability.md) and
  [the separate sufficient-width word theorem](general_shallow_tanh_width_theorem.md);
- [the homogeneous certificate note](certificate_note.md) and
  [exact certificate](certificate.json);
- [the graded reconstruction derivation](graded_reconstruction_derivation.md)
  and [exact reconstruction certificate](graded_reconstruction_certificate.json);
- [the finite-observation theorem](finite_observation_theorem.md) and
  [exact finite-observation certificate](finite_observation_certificate.json);
- [the local-stability theorem](local_stability_theorem.md) and
  [exact local-stability certificate](local_stability_certificate.json);
- [the canonical manuscript](../paper/randomized_signature_identifiability.tex)
  and [audit with actual-response validation](paper_mathematical_audit.md); and
- [the concise supervisor summary](graded_reconstruction_summary.md).

The new reproducer recomputes the entire finite certificate without writing:

```text
python -B research_output/reproduce_intrinsic_hall_rank.py
```

It checks 17 finite witness cases using exact rational arithmetic, including
graded quotients and inactive embeddings. These checks support the
implementation; the all-\(d,D,L\) theorem and genericity are proved in the
note. The existing reproducers retain their own artifacts. The numerical
rank atlas is non-certifying historical context. Actual-flow validation uses
`python -B research_output/validate_actual_response_recovery.py`; its
arbitrary-precision flow evaluations are numerical evidence, not exact
certificates.
