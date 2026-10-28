# Sharp fixed-state Hall rank for shallow biased-tanh fields

**Verdict: VERIFIED.** This note establishes generic maximal Hall evaluation
rank at initialization zero, and dimension-optimal local recovery for the
existing degree-\(L\) truncated polynomial response from one vector. It is the
authoritative note for this result; it does not replace the existing scalar
certificates or the sufficient-width word-response theorem.

## 1. Statement and conventions

Fix integers \(d,D,L\geq1\). Let
\[
 \mathfrak g=\bigoplus_{m=1}^L\mathfrak g_m,\qquad
 n_m=\dim\mathfrak g_m,\qquad
 \nu=\nu_{d,L}=\sum_{m=1}^L n_m
\]
be the real free step-\(L\) Lie algebra on generators \(e_1,\ldots,e_d\).
Use the repository's homogeneous Lyndon/Hall basis
\(H_1,\ldots,H_\nu\), ordered by increasing degree and then lexicographic
Lyndon word. Write \(m_a=\deg H_a\). The same symbols denote their canonical
homogeneous representatives in the full free Lie algebra inside the tensor
algebra, where \([a,b]=ab-ba\). These representatives, rather than a
representation of the step-\(L\) quotient, are used for nonlinear fields.

Consider the full parameter space
\[
 \mathcal P_{d,D}=\mathbb R^{d(D^2+D)},\qquad
 \theta=(A_1,b_1,\ldots,A_d,b_d),\qquad
 V_i^\theta(x)=\tanh(A_ix+b_i),
\]
where \(A_i\in\mathbb R^{D\times D}\), \(b_i\in\mathbb R^D\), and tanh acts
componentwise. Initial state is fixed at zero throughout.

For smooth functions, including vector-valued functions component by
component, the conventions are
\[
 [V,W]=DW\,V-DV\,W,\qquad
 D_Vf=Df\,V,\qquad
 \Gamma(i_1\cdots i_m)=D_{V_{i_1}}\cdots D_{V_{i_m}}.
\]
The first letter is the outermost operator. Tensor words are chronological,
with earliest time on the left, and multiplication concatenates left word
then right word. Thus \(\Gamma(uv)=\Gamma(u)\Gamma(v)\) on the full tensor
algebra. Cancellation of the Hessian terms gives
\[
 [D_V,D_W]=D_{[V,W]},\qquad
 \Gamma(H)=D_{\rho(H)},\qquad
 \Gamma(H)\operatorname{Id}=\rho(H).
\]
The last two identities follow by induction on brackets. They agree with
the existing word expansion and vector-field bracket implementations.

**Theorem (sharp fixed-state Hall rank).** For these fixed \(d,D,L\), define
\[
 H_\theta=
 \big[\rho_\theta(H_1)(0)\ \cdots\ \rho_\theta(H_\nu)(0)\big]
 \in\mathbb R^{D\times\nu}.
\]
There is a nontrivial real-analytic function
\(f_{d,D,L}:\mathcal P_{d,D}\to\mathbb R\), given by a fixed
\(\min(D,\nu)\)-square minor of this matrix, such that
\[
 f_{d,D,L}(\theta)\ne0
 \quad\Longrightarrow\quad
 \operatorname{rank}H_\theta=\min(D,\nu).
\]
The exceptional set \(\{f_{d,D,L}=0\}\) is Lebesgue null; its complement is
open and dense. Consequently the stated rank holds almost surely under
every jointly absolutely continuous law on the full parameter vector.
This conclusion does not automatically apply to bias-free, tied, sparse,
or other singular parameter laws.

The proof below constructs a finite parameter value at which the selected
minor equals \(2^{-\min(D,\nu)}\). No numerical rank or limiting parameter
value is used to establish its nontriviality.

## 2. The grading-derived affine construction

For a finite positively graded real Lie algebra, let
\(\mathcal E|_{\mathfrak g_m}=mI\). Positivity makes \(\mathcal E\)
invertible, and the grading identity makes it a derivation:
\[
 \mathcal E[a,b]=[\mathcal Ea,b]+[a,\mathcal Eb].
\]
For \(a\in\mathfrak g\) and \(c\in\mathbb R\), set
\[
 \Lambda_a=\mathcal E^{-1}\operatorname{ad}_a\mathcal E,\qquad
 W_a^c(x)=ca-\Lambda_ax.
\]
Here \(\operatorname{ad}_a b=[a,b]\); \(\mathcal E\) does not denote the
number of time steps. The usual matrix commutator satisfies
\[
 [\Lambda_a,\Lambda_b]
 =\mathcal E^{-1}[\operatorname{ad}_a,\operatorname{ad}_b]\mathcal E
 =\Lambda_{[a,b]}.
\]
The derivation identity also gives
\[
 \Lambda_ab-\Lambda_ba
 =\mathcal E^{-1}\big([a,\mathcal Eb]+[\mathcal Ea,b]\big)
 =[a,b].
\]
In particular, if \(a\) has degree \(s\) and \(b\) has degree \(t\),
\[
 \Lambda_ab=\frac{t}{s+t}[a,b],
\]
with both sides zero when the bracket is discarded by truncation.

Since \(DW_a^c=-\Lambda_a\), the repository's vector-field bracket gives
\[
 \begin{aligned}
 [W_a^c,W_b^c](x)
 &=c(\Lambda_ab-\Lambda_ba)
   +(\Lambda_b\Lambda_a-\Lambda_a\Lambda_b)x\\
 &=c[a,b]-\Lambda_{[a,b]}x
 =W_{[a,b]}^c(x).
 \end{aligned}
\]
Therefore \(a\mapsto W_a^c\) is a Lie algebra homomorphism, and for any
homogeneous Hall element \(H\),
\[
 \rho_W(H)(x)=cH-\Lambda_Hx,\qquad \rho_W(H)(0)=cH.
\]
In particular the evaluation factor is \(c\), not \(c^{\deg H}\).
This affine action respects the nilpotent Lie quotient.

The underlying construction is classical: apply Burde's nonsingular-cocycle
construction with the adjoint module and cocycle \(\mathcal E\).
Proposition 3.3 gives the transported left multiplication
\(\mathcal E^{-1}\operatorname{ad}_a\mathcal E\); Corollaries 3.4–3.5 cover
invertible derivations and positive gradings. These locators were verified
in Section 3.1, printed pages 14–16, with Proposition 3.3 on page 15 and
Corollary 3.5 on page 16. The vector-field sign above is checked directly
rather than identified with an affine-matrix sign convention.
[Burde, *Left-symmetric algebras, or pre-Lie algebras in geometry and physics*](https://homepage.univie.ac.at/dietrich.burde/papers/burde_24_pre_lie.pdf).

## 3. A finite shallow-tanh witness at width \(\nu\)

Identify \(\mathfrak g\) with \(\mathbb R^\nu\) in its ordered Hall basis.
Let
\[
 c=\frac12,\qquad q=\operatorname{atanh}(1/2)=\frac12\log3,\qquad
 A_i=-\Lambda_{e_i},\qquad b_i=qe_i.
\]
These are finite admissible parameters of the specified architecture:
\[
 V_i(x)=\tanh(qe_i-\Lambda_{e_i}x).
\]
There is no normalization, external output multiplier, or epsilon limit.

Give the state coordinate corresponding to a degree-\(k\) Hall element
weight \(k\). Since \(\Lambda_{e_i}\) raises grade by one, the grade-one
components of \(V_i\) are exactly the constant vector \(ce_i\). Each
individual grade-\(k\) component, \(k\geq2\), is
\[
 \tanh(\ell_{i,a}(x)),\qquad
 \ell_{i,a}(x)=-(\Lambda_{e_i}x)_a,
\]
where \(\ell_{i,a}\) is linear in grade-\((k-1)\) coordinates. Its linear
term is precisely the corresponding component of \(W_{e_i}^c\).

We justify the remaining calculation with finite jets. For a word of
length \(m\), the value \(\Gamma(w)\operatorname{Id}(0)\) depends only on
ordinary derivatives of the generator fields through order \(m-1\).
Indeed, the length-one expression is the field itself, and the recursion
\[
 F_{iv}=DF_v\,V_i,\qquad F_w=\Gamma(w)\operatorname{Id},
\]
increases the largest derivative order by at most one. Hence replacing
each \(V_i\) by its ordinary Taylor polynomial through degree \(L\) at zero
preserves all these evaluations for \(m\leq L\).

For the components of grade at least two, this Taylor polynomial consists
of \(\ell_{i,a}\) and finitely many terms
\(\alpha_n\ell_{i,a}^{\,n}\), with odd \(3\leq n\leq L\).
This uses only the finite Taylor jet of tanh at zero. Its remainder has
zero derivatives through the required orders; no globally convergent
Taylor expansion is asserted.

For clarity, a monomial differential operator \(p(x)\partial_{x_a}\)
has weight shift \(s-k\) when \(p\) has weighted degree \(s\) and \(x_a\)
has weight \(k\): it sends a weighted homogeneous polynomial of degree
\(t\) to one of degree \(t+s-k\), or to zero. These shifts add under
composition. Every affine term of \(D_{W_{e_i}^c}\) has shift \(-1\).
Every nonlinear Taylor term has shift
\[
 n(k-1)-k=-1+(n-1)(k-1)\geq1.
\]
Thus, when a length-\(m\) word acts on a grade-\(k\) coordinate, its
all-affine contribution has weighted degree \(k-m\). Every contribution
using at least one nonlinear term has strictly greater weighted degree.
All these statements concern finite polynomial expressions. Evaluation
at zero selects only their terms of weight zero.

It follows that for \(k>m\) the evaluation is zero, and for \(k=m\) it
agrees with the affine evaluation. Expand a homogeneous Hall element
\(H\in\mathfrak g_m\) into its existing length-\(m\) tensor words and use
the operator identities from Section 1. Writing \(P_k\) for projection
onto grade \(k\), we obtain
\[
 P_k\rho_V(H)(0)=0\quad(k>m),\qquad
 P_m\rho_V(H)(0)=P_m\rho_W(H)(0)=cH.
\]
Consequently, with increasing-degree rows and columns, \(H_\theta\) at
this witness is block **upper triangular**, with diagonal blocks
\(cI_{n_m}\). Empty grade blocks can simply be omitted. Therefore
\[
 \det H_\theta=c^\nu=2^{-\nu}.
\]
The proof does not assert \(H_\theta=cI\): evaluations in grades below
the Hall degree can be nonzero.

## 4. All state dimensions and analytic genericity

Suppose \(D<\nu\), and put \(s_0=0\),
\(s_p=\sum_{m=1}^p n_m\). Let \(p\) be the smallest grade for which
\(s_p\geq D\), and let \(t=D-s_{p-1}\). Retain all basis vectors below
grade \(p\) and the first \(t\) in grade \(p\). Let \(U_p\) span the
remaining grade-\(p\) basis vectors, and define
\[
 I=U_p\oplus\bigoplus_{k=p+1}^L\mathfrak g_k.
\]
This is a graded ideal. In fact a bracket of \(U_p\) with any positive
grade has degree above \(p\), and brackets with grades above \(p\) remain
above \(p\), or vanish. The quotient
\(\overline{\mathfrak g}=\mathfrak g/I\) therefore has dimension \(D\),
with the retained Hall images as its ordered homogeneous basis. Its
grading derivation \(\overline{\mathcal E}\) is invertible.

On this quotient use
\[
 \overline\Lambda_a
 =\overline{\mathcal E}^{-1}\operatorname{ad}_a\overline{\mathcal E},
 \qquad
 V_i(x)=\tanh(q\overline e_i-\overline\Lambda_{\overline e_i}x),
\]
where \(\overline e_i\) is the image of the original generator. The
affine identities and finite-jet proof apply on the quotient. Its
grade-one generator images are coordinate vectors or zero, so the
grade-one tanh components are exactly \(c\overline e_i\).
The \(D\) columns of the original Hall matrix indexed by the retained
basis vectors form a block upper triangular matrix with diagonal \(cI\).
With all \(D\) rows, this minor has determinant \(2^{-D}\).

If \(D<d\), then \(p=1\): the quotient retains the first \(D\) generators
and discards all other generators and higher grades. Its retained fields
are constant \(ce_i\), its other generator fields are zero, and the
selected minor is \(cI_D\). Thus this case is included. The argument uses
a proved quotient construction, rather than an arbitrary submatrix crop.

Only the selected columns are claimed to have the stated minor.
Discarded Hall elements can have nonzero represented fields for the
nonlinear tanh witness. Unlike the affine action, the nonlinear action
is not claimed to descend to this quotient.

For \(D>\nu\), put the width-\(\nu\) witness in the first \(\nu\)
coordinates, and set every remaining weight and bias entry to zero.
The fields then have the form \((V_i^{\rm act}(x^{\rm act}),0)\).
The word recursion preserves this form. The first \(\nu\) rows and all
Hall columns therefore give the same determinant \(2^{-\nu}\).

Fix the row and column choices just described: all rows and the retained
\(D\) Hall columns if \(D<\nu\), and the first \(\nu\) rows and all Hall
columns if \(D\geq\nu\). Define \(f_{d,D,L}\) as this minor for arbitrary
parameters in the **full** space \(\mathcal P_{d,D}\), still evaluated at
zero. The map \((\theta,x)\mapsto V_i^\theta(x)\) is jointly real analytic.
Finite differentiation, the word recursion, and the Hall word expansions
preserve this property. Evaluation at zero and taking the determinant
therefore give a real-analytic \(f_{d,D,L}\).

Its value at the finite witness is \(2^{-\min(D,\nu)}\), so it is not
identically zero on the connected Euclidean parameter space. The
real-analytic zero-set theorem makes its zero set Lebesgue null
([Mityagin, Proposition 0](https://arxiv.org/pdf/1512.07276)).
Continuity and nullity give an open dense complement. On that complement,
the selected minor supplies rank at least \(\min(D,\nu)\); matrix size
supplies the opposite bound. This proves the theorem. A zero of this
particular minor need not be a rank-deficient parameter.

## 5. Local recovery from one truncated-response vector

Write
\[
 \ell(z)=\sum_{a=1}^{\nu}z_aH_a,\qquad
 \delta_r\ell(z)=\sum_{a=1}^{\nu}r^{m_a}z_aH_a.
\]
Define the finite tensor polynomial
\[
 P_L(r,z)=\pi_{\leq L}
       \sum_{j=0}^{L}\frac{(\delta_r\ell(z))^j}{j!},
 \qquad
 F_0^{\leq L}(r;z)=\big(\Gamma_\theta(P_L(r,z))
                            \operatorname{Id}\big)(0).
\]
The projection discards tensor words of length above \(L\).
Equivalently it discards Hall weighted degree above \(L\) after expansion.
Apply \(\Gamma_\theta\) to this finite representative in the full tensor
algebra. No factorization of the nonlinear operator action through the
truncated tensor or Lie quotient is needed.

This is precisely the current manuscript's truncated response at \(y=0\):
\[
 F_0^{\leq L}(r;z)
 =\sum_{m=1}^L r^m
   \big(\Gamma_\theta(\pi_m\exp_\otimes\ell(z))
                    \operatorname{Id}\big)(0).
\]
The empty word has coefficient one and contributes
\(\Gamma(1)\operatorname{Id}(0)=0\). It is a known constant coefficient,
not an additional unknown Hall coordinate; no unital evaluation
injectivity is claimed. The factorials above are those of the tensor
exponential. The normalized scale-jet convention remains
\(J_m=(1/m!)\partial_r^mY|_{r=0}\), with no further factorial multiplying
a homogeneous response coefficient.

**Corollary (one-vector local truncated recovery).** Fix \(d,D,L\) with
\(D\geq\nu_{d,L}\), and a parameter with \(f_{d,D,L}(\theta)\ne0\).
For every fixed \(r\ne0\), the map
\(z\mapsto F_0^{\leq L}(r;z)\) admits local analytic recovery of all
\(\nu\) Hall coordinates near \(z=0\). If \(D=\nu\), the response map
itself has a local analytic inverse. If \(D>\nu\), projection onto the
selected \(\nu\) output coordinates has a local analytic inverse, which
recovers \(z\) from the full vector.

**Proof.** The response is a polynomial in \(z\), and its value at zero
is zero. Differentiation at \(z=0\) kills every term with at least two
Lie factors in the exponential. The term with one factor has coefficient
one and is retained because \(m_a\leq L\). Hence
\[
 \partial_{z_a}F_0^{\leq L}(r;0)
 =r^{m_a}\Gamma_\theta(H_a)\operatorname{Id}(0)
 =r^{m_a}\rho_\theta(H_a)(0),
\]
and
\[
 D_zF_0^{\leq L}(r;0)
 =H_\theta\operatorname{diag}(r^{m_1},\ldots,r^{m_\nu}).
\]
All diagonal scale factors are nonzero. Let \(P\) select the certified
\(\nu\) rows, taking \(P=I\) when \(D=\nu\). Then
\[
 \det D_z(PF_0^{\leq L})(r;0)
 =r^{\sum_a m_a}f_{d,D,L}(\theta)\ne0.
\]
The analytic inverse-function theorem gives neighborhoods \(U\) and
\(\Omega\) of zero in \(\mathbb R^\nu\), and an analytic map
\(g:\Omega\to U\), with
\[
 g(PF_0^{\leq L}(r;z))=z\qquad(z\in U).
\]
For \(D>\nu\), this is inversion of the projected response, not a
bijection between open subsets of \(\mathbb R^\nu\) and \(\mathbb R^D\).
The neighborhoods may depend on \(\theta\) and \(r\). The same certified
parameter set works for every nonzero \(r\), but no uniform inverse
neighborhood or stability constant is asserted. \(\square\)

**Dimension lower bound.** If \(D<\nu\), no continuous map from a nonempty
open Hall-coordinate neighborhood in \(\mathbb R^\nu\) to
\(\mathbb R^D\) is injective there. Otherwise, append \(\nu-D\) zero
coordinates to obtain a continuous injection into \(\mathbb R^\nu\).
Invariance of domain would make its image open, although that image lies
in a proper linear subspace with empty interior, a contradiction.
The required version is the continuous-injection statement of
[Hatcher, *Algebraic Topology*, Theorem 2B.3, printed page 172](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf).
Together with the corollary, this proves that \(D=\nu\) is the sharp
width for this one-vector, continuous local truncated-coordinate
observation model. It is not a minimum number of reservoir runs for
multiple-query designs, or a bound for restricted path families.

## 6. Relationship to the existing results and sources

This theorem concerns evaluations of Hall fields at one fixed state.
When \(D\geq\nu\), their independent evaluations also imply independence
as vector-valued functions, but the converse need not hold. In
particular a fixed-state rank bound of \(D\) does not impose that bound
on function-level Hall independence.

The existing sufficient-width theorem and the fixed-zero corollary in the
[canonical manuscript](../paper/randomized_signature_identifiability.tex)
(label \(\texttt{cor:zero}\)) concern the larger family of all
\(Q_{d,L}=\sum_{m=1}^L d^m\) nonempty word responses, using
\(D_0(d,L)=\sum_{m=0}^L d^m\). They remain separate statements with their
existing threshold. The present Hall minor does not establish
independence of that word family at width \(\nu\).

The certified scalar homogeneous theorem, its mixed-degree obstruction,
the designed eight scalar observations at states \(-q,0,q\), and their
local inverse and approximate actual-response recovery remain unchanged.
Their radius, inverse constants, scale exponents and full-response
remainder estimates do not apply to the new map. For example, when
\(d=2,L=4\), the new width threshold is \(\nu=8\) for one vector of
truncated Hall observations; the existing word construction has
\(D_0=31\) and \(Q=30\).

The corollary concerns the finite polynomial response, not exact recovery
from an actual full CDE endpoint. It establishes no linear-readout
approximation, practical conditioning, or global inverse. All genericity
arguments here hold the initial state at zero; no simultaneous
full-measure statement over all initial states is used.

The comparison with the reconstruction references is limited to the
following checked statements:

- Monzio Compagnoni et al., arXiv v3, Definition II.5 and Theorem IV.1
  (PDF page 4), use the same shallow biased architecture. The proof asserts
  generic Hörmander bracket generation for a Brownian-density conclusion.
  It gives no prescribed Hall cutoff or exact minor. Its broad analytic
  activation hypothesis is not adopted here: constant activations cannot
  span a state space of dimension greater than one.
  [Primary paper, version 3](https://arxiv.org/pdf/2201.00384v3).
- Cuchiero–Larsson–Teichmann, arXiv v2, Theorem 5.1 and Lemma 6.1,
  use witness families and analytic minors of bracket evaluations for
  generic interpolation. Their witness hypothesis is generation of all
  polynomial vector fields. Corollary 5.4 and Example 5.5 permit analytic
  activation families with linear/quadratic specializations, including
  tanh mixtures. This is direct methodological precedent, but not the
  prescribed-degree rank law for a fixed tanh activation.
  [Primary paper, version 2](https://arxiv.org/pdf/1908.07838v2).
- Grong–Vega-Molino, arXiv v3, Remark 3.2, use the scalar constant/tanh
  pair, polynomial brackets and Vandermonde evaluation for landmark
  controllability. The repository's scalar witness is this pair up to
  generator exchange and nonzero constant scaling. Its homogeneous
  degree-four injectivity and mixed-degree relation are separate
  calculations; unrestricted spanning does not establish them.
  [Primary paper, version 3](https://arxiv.org/html/2403.08090v3).
- Akyildirim et al., Proposition 2.10 (page 7), give the classical tensor
  signature system. Their Definition 2.12 (page 8) specifies the shallow
  architecture; Theorem 2.13 (page 9) concerns linear approximation using
  all initial values and an activation with infinite Taylor radius. It
  does not give the fixed-zero biased-tanh Hall-rank statement here.
  [Primary paper](https://arxiv.org/pdf/2201.02441).
- Glückstad–Muça Cirone–Teichmann, version 1, Section 3.3.2, Remark 3.10,
  Theorem 4.1 and Section 5 concern the word/tree framework and
  reconstruction. The reconstruction proof uses selected initial states
  and input-scale derivatives; Section 5 discusses direct log-signature
  recovery as open.
  These passages do not state the sharp fixed-state Hall-rank theorem.
  This comparison does not identify projected tree terms with complete
  summed fields, and does not refute their complete reconstruction theorem.
  [Primary paper, version 1](https://arxiv.org/html/2502.03163v1).
- Bailleul–Diehl, Theorem 6 (page 7), use generator and degree-two bracket
  evaluations at finitely many states to characterize small-time
  rough-flow reconstruction, including a one-state case. This is relevant
  reconstruction-matrix methodology, but does not supply the present
  arbitrary-\(L\) shallow-tanh witness and sharp rank theorem.
  [Primary paper, version 3](https://arxiv.org/pdf/1407.2768v3).

The grading-derived affine machinery, witness-and-analytic-minor method
and rank-based local inversion are established tools. The specific result
above is the finite-parameter shallow-tanh realization with exact maximal
Hall minors at every prescribed finite degree and width, and its local
truncated inverse. The comparison does not establish worldwide priority.
Fixing the state alone, absence of a matching located statement, or a
qualification of another proof is not evidence of novelty.

## 7. Verification and migration checklist

The proof uses the current conventions in
[free_lie.py](../lie_reservoir/free_lie.py),
[brackets.py](../lie_reservoir/brackets.py),
[graded_reconstruction.py](../lie_reservoir/graded_reconstruction.py),
and [tensor_witness.py](../lie_reservoir/tensor_witness.py).
The architecture matches
[vector_fields.py](../lie_reservoir/vector_fields.py).
The current [canonical manuscript](../paper/randomized_signature_identifiability.tex)
already supplies the finite polynomial definition and its Hall-coordinate
derivative. The technical and certificate notes, graded reconstruction
summary and derivation, general graded identifiability, finite observation,
local stability, general shallow-tanh width note, and mathematical audit
were read in their current repository versions. Earlier scope summaries
do not override the later actual-response result in the manuscript.

The finite implementation is
[intrinsic_hall_rank.py](../lie_reservoir/intrinsic_hall_rank.py), with
[focused tests](../tests/test_intrinsic_hall_rank.py),
[the exact certificate](intrinsic_hall_rank_certificate.json), and
[its reproducer](reproduce_intrinsic_hall_rank.py).
These are the canonical paths for this implementation.

The module reuses the existing Lyndon basis, Witt dimensions, tensor
bracket expansions and tanh recurrence from the certificate code.
Leading-word elimination obtains adjoint coordinates and checks their
full tensor reconstruction. Generator invariance of the discarded graded
ideal is checked before inducing quotient matrices. All original input
labels remain present, including zero projected generators. The grading
operator is inverted only on the active coordinates before any padding.

The actual tanh jets use exact rational coefficients from
\(T'=1-T^2\), with \(\tanh(q)=1/2\) and no numerical value of \(q\).
Generator jets through ordinary order \(L-1\) suffice. At word length
\(m\), the implementation retains ordinary polynomial degree \(L-m\):
each remaining outer derivative lowers ordinary degree by at most one,
and multiplication by a field coefficient cannot lower it. The Hall
matrix is then obtained from the existing tensor word expansions,
without inserting its predicted triangular entries.

The computational limits are \(1\leq d\leq3\), \(1\leq L\leq5\),
\(d^L\leq128\), and \(1\leq D\leq40\); invalid inputs are rejected.
These limits do not restrict the theorem. The coefficient recurrence
separately accepts any nonnegative order; its degree-seven coefficient
at zero, \(-17/315\), is checked by independent symbolic differentiation.
The production arithmetic is Fraction-based. SymPy is used in tests
for independent exact differentiation, polynomial interpolation and
matrix checks.

All six requested full-width reference cases were feasible and reproduced:

| \(d\) | \(L\) | \(D=\nu\) and exact rank | Selected minor determinant |
|--:|--:|--:|--:|
| 2 | 2 | 3 | \(1/8\) |
| 2 | 3 | 5 | \(1/32\) |
| 2 | 4 | 8 | \(1/256\) |
| 2 | 5 | 14 | \(1/16384\) |
| 3 | 3 | 14 | \(1/16384\) |
| 3 | 4 | 32 | \(1/4294967296\) |

The certificate additionally records \((d,L)=(2,4)\) at
\(D=1,4,7,9\); \((3,3)\) at \(D=2,8\); \((1,5)\) at \(D=1,2\);
and \((3,1)\) at \(D=2,3,4\). These 17 cases include \(D<d\),
quotients cutting through a homogeneous grade, and inactive embeddings.
Every computed selected minor equals \(2^{-\min(D,\nu)}\).

In the repository's degree-ordered Lyndon word labels
\((1,2,12,112,122,1112,1122,1222)\), the full width-eight evaluation matrix
was exactly
\[
 H_{\theta_*}=\tfrac12 I_8-\tfrac1{32}(E_{3,6}+E_{3,8}),
\]
where \(E_{a,b}\) is the matrix unit with entry one in row \(a\), column
\(b\). The complete matrix, including both \(-1/32\) entries, is a
regression fixture in the tests and is recomputed from actual tanh jets.
The embedded ninth row is zero. A test that sets the actual weight
matrices to zero checks that the evaluator responds to changed fields
instead of returning the fixture.

Further independent tests check the affine cocycle and commutator
identities on all 64 pairs of the width-eight Hall basis, direct
actual-tanh bracket signs, and the full \(d=2,L=3\) truncated response.
The latter is reconstructed by exact polynomial interpolation using the
existing tensor exponential and independently differentiated tanh fields,
including its nonlinear Hall-coordinate terms. Its zero Jacobian agrees
with the new routine at rational scales \(0,1,1/3,-2\).
Scale zero gives the zero derivative and provides no local inverse.

The reproducer recomputes and compares the entire deterministic JSON
payload. A tamper test removes one off-diagonal entry while preserving
rank, determinant and the selected-minor slice; verification rejects it
without rewriting the file. Neither default verification nor explicit
generation writes any old certificate. The all-\(d,D,L\) result and
genericity remain consequences of the proof above, not of these finite
examples.

From the repository root, the normal read-only command is:

    python -B research_output/reproduce_intrinsic_hall_rank.py

Explicit regeneration of only the new certificate is:

    python -B research_output/reproduce_intrinsic_hall_rank.py --write-artifacts

Both commands were run successfully. Default verification passed all
17 cases. The focused command

    python -B -m pytest -q tests/test_intrinsic_hall_rank.py

passed 38 tests in 11.88 seconds. All five existing reproduction commands
also passed:

    python -B research_output/reproduce_certificate.py
    python -B research_output/reproduce_graded_reconstruction.py
    python -B research_output/reproduce_finite_observation.py
    python -B research_output/reproduce_local_stability.py
    python -B research_output/reproduce_tensor_witness.py

The affected existing exact tests were run with:

    python -B -m pytest tests/test_free_lie.py tests/test_brackets.py tests/test_certificates.py tests/test_graded_reconstruction.py tests/test_finite_observation.py tests/test_local_stability.py tests/test_tensor_witness.py -q -p no:cacheprovider

All 135 passed in 76.75 seconds. The environment used Python 3.13.3,
pytest 9.1.1 and SymPy 1.14.0.

The repository-required full suite was run once:

    python -B -m pytest -q -p no:cacheprovider

All 271 tests passed in 213.80 seconds. Python syntax and scoped
new-file whitespace checks passed. The repository-wide git diff --check
still reports pre-existing whitespace in rsmc/__init__.py, rsmc/hedging.py
and tests/test_hedging.py; those files were not changed. Content-hash and
section checks confirmed that the other existing source files and
Sections 1–6 of this note were preserved.

The following migration checklist was recorded at the end of Stage 2:

1. Add the Hall-rank theorem, finite witness proof and one-vector local
   truncated inverse beside the general architecture discussion, using
   the existing Hall ordering, truncated-response definition and
   \(\mathcal E\) notation.
2. Add Burde's precise classical attribution and the observation-specific
   dimension lower bound; retain the verified comparison scope above.
3. Preserve the existing word-response theorem and fixed-zero corollary,
   scalar results, eight-observation design, constants, actual-response
   theorem, labels and accepted corrections. Adjust only surrounding
   scope summaries that need to mention this additional theorem.
4. Reference the new exact harness and certificate as finite checks,
   preserving the distinction between computation and the general proof.

Stage 2 adds only the new module, focused test file, reproducer and
certificate, and updates this implementation section. The theorem and
proof in Sections 1–6, canonical manuscript, other notes, existing APIs,
scalar certificates and experiments remain unchanged. Paper-wide and
notes-wide integration is deferred to Stage 3.

**Stage 3 integration, 13 September 2026.** The checklist above is now
complete. The canonical manuscript leads with the fixed-zero Hall-rank
theorem, contains this affine/finite-jet/quotient proof and the local
truncated inverse with its dimension bound, and attributes the classical
construction. Its computational discussion records the two nonlinear
width-eight entries and the finite harness limits separately from the
theorem. The overview and companion notes now state the relationship to the
preserved scalar and word-family results. Sections 1–6 of this authoritative
note and the exact certificate are unchanged. Current reproduction and
document checks, including the unavailable LaTeX build, are recorded in the
dated addendum to [the manuscript audit](paper_mathematical_audit.md).

**Final independent audit, 13 September 2026.** The central affine,
finite-jet, quotient and fixed-state genericity arguments were independently
checked, together with the truncated inverse and its dimension lower bound.
All 17 persisted full Hall matrices and selected minors reproduced in
read-only mode. Direct symbolic tanh brackets independently recovered the
width-eight matrix above; corruption preserving rank and determinant was
rejected. All five legacy reproducers passed, including generated-note
comparison, and the current full suite passed 271 tests in 127.32 seconds.
No implementation or certificate change was required. Source-scope and
cross-reference corrections are synchronized in the manuscript and notes.
A portable Tectonic 0.17.0 build produced 37 pages, all visually inspected;
there are no overfull boxes or unresolved references/citations. The
historical Stage 3 build limitation above is preserved as history. The
final audit entry records commands, minor remaining compiler warnings and
the unrelated pre-existing whitespace gate failure. The upgrade is complete
within the theorem's stated scope.
