# Exact graded reconstruction and local stability for a shallow tanh reservoir

Status: **READY FOR SUPERVISOR REVIEW**.

## 1. Motivation and current contribution

The pricing and partial-information hedging experiments in this repository
motivated a mathematical study of the continuous-time vector-field analogue of
the shallow biased randomized-signature reservoir. The result below concerns
that continuous-time representation only. It does not transfer to the
implemented discrete recurrence and does not prove a pricing, hedging, or
control-variate theorem.

For the scalar two-input shallow biased-tanh architecture, the completed
contribution is:

1. generic injectivity of each homogeneous free-Lie component through degree
   four;
2. exact recursive reconstruction of degree-at-most-four log-signature Hall
   coordinates from ideal normalized graded scale jets;
3. an explicit dimension-matched eight-output state-scale design for the
   degree-four truncated response polynomial; and
4. an exact quantitative local inverse-Lipschitz certificate for that
   truncated map.

These statements do not form one linear implication chain. Their proof
dependencies are recorded in Section 3.

## 2. Setup and the three observation objects

Let

\[
V_i^\theta(x)=\tanh(a_i x+b_i),\qquad
\theta=(a_1,b_1,a_2,b_2)\in\mathbb R^4,\qquad i=1,2,
\]

and define the free-Lie representation

\[
\rho:\mathcal L_2\longrightarrow\mathfrak X(\mathbb R),
\qquad \rho(X_i)=V_i,
\]

with bracket convention

\[
[V,W](x)=DW(x)V(x)-DV(x)W(x)
        =W'(x)V(x)-V'(x)W(x).
\]

Write \(\rho_m=\rho|_{\mathcal L_{2,m}}\). For a continuous
bounded-variation input \(X\), the scaled response always means the response
to the scaled input path,

\[
Y^{y,r}(X):=Y^y(rX),
\qquad
dY_t^{y,r}
=r\sum_{i=1}^2V_i(Y_t^{y,r})\,dX_t^i,
\qquad Y_0^{y,r}=y.
\]

No state-rescaling identity is used.

Three observation objects must remain distinct:

- **A. Ideal normalized graded scale jets**
  \[
  J_m(y;X)
  =\frac1{m!}\left.\partial_r^mY_T^{y,r}(X)\right|_{r=0}.
  \]
- **B. The degree-four truncated finite-scale response polynomial**
  \(F_y^{(4)}(r;z)\), where \(z\in\mathbb R^8\) contains the deterministic
  Hall coordinates of \(\ell^{\le4}\).
- **C. The actual full CDE response** \(Y_T^y(rX)\), which also contains
  homogeneous levels at least five.

The ideal graded decoder concerns A. The eight-output inverse and quantitative
stability theorems concern B. The present package does not identify B with C.

## 3. Correct proof dependency

The concrete shallow-tanh theorem supplies an explicit generic family
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

## 4. Homogeneous injectivity and its exact witness

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

This is a general graded-identifiability criterion within the stated
degree-four setting. The shallow-tanh theorem supplies one concrete generic
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

The rank-seven cross-degree relation makes every all-at-one-nonzero-scale
design singular, so multiple scales are essential. Under a common nonzero
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

## 8. Related work and novelty boundary

The randomized-signature setting predates this package. Akyildirim, Gambara,
Teichmann, and Zhou used randomized signatures as feature extractors and gave
a representation-theoretic justification in
[*Applications of Signature Methods to Market Anomaly Detection* (2022)](https://arxiv.org/abs/2201.02441).

The closest prior work identified in the targeted search is Mie Glückstad,
Nicola Muça Cirone, and Josef Teichmann,
[*Signature Reconstruction from Randomized Signatures* (2025),
arXiv:2502.03163v1](https://arxiv.org/html/2502.03163v1). The paper states a
reconstruction theorem for ordinary tensor-signature levels from
CDE/randomized-signature observations under a linear-independence hypothesis.
Its proof introduces a scalar scaling parameter, differentiates at zero,
separates homogeneous tensor levels, and selects finitely many initial states.
For an explicit depth-two nested-exponential architecture it states
almost-sure reconstruction through level \(L\) when the hidden dimension
satisfies \(N\ge L-1\). Section 5 identifies moving directly to the compressed
log-signature/Lie coordinates as a further direction; if the claimed truncated
tensor signature is obtained, it can also be postprocessed by the truncated
tensor logarithm.

There is a source-level qualification. In the Euclidean proof of Theorem 4.1,
the asserted identity \(Y^{\eta,r}=rY^{\eta/r}\), used to obtain the scaled
response family from unscaled responses over all initial states, is not valid
for general nonlinear vector fields in the displayed \(rV_i\)-scaled CDE.
The subsequent Taylor and finite-evaluation argument remains the relevant
prior strategy if that scaled family is independently available, but the
printed state-rescaling bridge does not establish it in the stated generality.
This package uses only responses to the scaled input path and no
state-rescaling identity. In either reading, the
2025 paper does not provide a finite set of scale values or the exact
eight-output and inverse-Lipschitz certificates established here.

Accordingly, this package claims no originality for introducing path or vector
field scaling, differentiating in scale, separating tensor levels, choosing
finite initial states, extracting information from CDE observations, or
reconstructing ordinary tensor signatures.

Inverse controlled differential equations are also prior art. Bailleul and
Diehl,
[*The Inverse Problem for Rough Controlled Differential Equations*,
SIAM Journal on Control and Optimization 53(5), 2762--2780
(2015)](https://doi.org/10.1137/140995982), give a necessary and sufficient
condition for a rough control to be reconstructable to finite order from the
observed controlled evolution. The present package is not claimed to be the
first inverse-CDE result.

The free-Lie/Hall--Lyndon algebra, group-like signatures and log-signatures,
tensor exponential and logarithm, and Chen--Fliess-type controlled-ODE
expansions are classical. Standard sources used here are Christophe
Reutenauer's
[*Free Lie Algebras* (1993)](https://academic.oup.com/book/53911),
Lyons, Caruana, and Lévy's
[*Differential Equations Driven by Rough Paths* (2007)](https://doi.org/10.1007/978-3-540-71285-5),
and Friz and Victoir's
[*Multidimensional Stochastic Processes as Rough Paths* (2010)](https://doi.org/10.1017/CBO9780511845079).
The abstract triangular passage between signature and log-signature
coordinates is therefore not presented as a priority claim.

Relative to the closest prior work identified, the present package differs in
working directly with homogeneous free-Lie/log-signature coordinates for the
scalar two-input shallow biased-tanh architecture and in certifying the exact
witness, the fixed eight-output degree-four truncated design, fixed-design
genericity, and an exact local inverse-Lipschitz bound. These are
package-specific proved statements, not assertions that a literature search
has established priority.

| Category | Boundary |
|---|---|
| Classical or prior | Free-Lie and Hall--Lyndon algebra; signature/log-signature and tensor algebra; scale/Taylor level separation; inverse-CDE problems; ordinary tensor-signature reconstruction from suitable CDE observations. |
| Proved in this package | Generic homogeneous injectivity through degree four for the scalar two-input shallow biased-tanh architecture; exact witness and homogeneous certificates; ideal graded reconstruction; explicit eight-output object-B design; exact Jacobian and inverse; fixed-design genericity; exact local inverse-Lipschitz certificate; exact common-scale inverse-linearization amplification. |
| Open or not claimed | Actual full-CDE reconstruction; degree at least five; input dimension at least three; state dimension at least two; parameter-uniform stability; generic ungraded injectivity; discrete-time recovery; finance or hedging consequences; rough-path extension. |

## 9. Exact scope and current status

The completed theorem package is confined to two inputs, scalar state, shallow
biased tanh, and homogeneous degrees one through four. Its reconstruction
claims are exact for A and local for B. It does not prove:

- injectivity on the ungraded direct sum;
- recovery from one terminal state;
- global invertibility;
- recovery of the full infinite log-signature;
- reconstruction from actual full-CDE observations;
- a degree-five or higher architecture theorem;
- higher input or state dimension;
- parameter-uniform stability;
- discrete-time reconstruction;
- finance or hedging consequences; or
- a rough-path extension.

The only substantive downstream mathematical bridge is from B to C: control
the degree-at-least-five remainder of the full CDE on an appropriately bounded
path class. Observation noise or solver error can enter the present stability
estimate only after separate bounds have been established; none is supplied
here.

## 10. Detailed sources and artifacts

The root theorem package is supported by:

- [the homogeneous certificate note](certificate_note.md) and
  [exact certificate](certificate.json);
- [the graded reconstruction derivation](graded_reconstruction_derivation.md)
  and [exact reconstruction certificate](graded_reconstruction_certificate.json);
- [the finite-observation theorem](finite_observation_theorem.md) and
  [exact finite-observation certificate](finite_observation_certificate.json);
- [the local-stability theorem](local_stability_theorem.md) and
  [exact local-stability certificate](local_stability_certificate.json); and
- [the concise supervisor summary](graded_reconstruction_summary.md).

The corresponding reproducers verify the persisted exact artifacts without
writing by default. The numerical rank atlas remains non-certifying historical
context and is not part of the exact proof.

## 11. Sole remaining mathematical question

Can the degree>=5 full-CDE remainder be bounded sharply enough, on an
appropriately bounded path class, to transfer the truncated local stability
theorem to actual shallow reservoir-flow observations?
