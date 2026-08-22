# Graded reconstruction and finite truncated observation

## Contribution

This package establishes one theorem package for a scalar, two-input shallow
biased-tanh reservoir through homogeneous degree four. Its proof dependencies
branch rather than form one linear implication chain:

```text
homogeneous Lie injectivity
|-- ideal graded reconstruction (object A)
`-- joint state-scale independence
    `-- explicit eight-output design (object B)
        `-- exact zero Jacobian B_*
            |-- local analytic inverse
            `-- quantitative stability from exact B_*^{-1} and Delta(R)
```

The shallow-tanh theorem supplies a concrete generic family satisfying the
homogeneous-injectivity premise. Graded reconstruction and joint state-scale
independence are sibling consequences of that premise. Quantitative stability
comes from the exact inverse and Jacobian-variation majorant, not from the
analytic inverse function theorem.

The package deliberately distinguishes three observation objects:

- **A. Ideal graded jets:** the normalized derivatives of the response to the
  scaled input path $rX$ at $r=0$, degree by degree.
- **B. Degree-four truncated response:** the finite-scale polynomial obtained
  by retaining homogeneous degrees one through four.
- **C. Actual full CDE response:** the reservoir-flow response, which also
  contains terms of degree at least five.

Graded reconstruction concerns A. The eight-output inverse and quantitative
stability theorems concern B. No result below identifies B with C.

## Homogeneous injectivity and graded reconstruction

For

\[
V_i^\theta(x)=\tanh(a_i x+b_i),\qquad
\theta=(a_1,b_1,a_2,b_2)\in\mathbb R^4,
\]

on a scalar state space, with two input letters and bracket convention

\[
[V,W]=DW\,V-DV\,W,
\]

the restrictions

\[
\rho_m:\mathcal L_{2,m}\longrightarrow\mathfrak X(\mathbb R)
\]

are simultaneously injective, separately for each $m=1,2,3,4$, for
Lebesgue-almost every $\theta$. The exact nonvanishing witness is

\[
q=\frac{\log 3}{2},\qquad
\theta_*=(1,0,0,q),\qquad
V_1(x)=\tanh x,\quad V_2(x)=\frac12.
\]

This is a homogeneous statement and does not assert injectivity on the
ungraded direct sum.

The reconstruction criterion is logically separate from this concrete tanh
theorem. For continuous bounded-variation inputs (in particular, continuous
piecewise-$C^1$ inputs) and known $C^4$ scalar vector fields satisfying the
local solution hypotheses, define the ideal normalized scale jets by

\[
J_m(y;X)=\frac1{m!}\left.\partial_r^mY_T^y(rX)\right|_{r=0}.
\]

If every homogeneous restriction $\rho_m$ is injective, then the intrinsic
Hall coordinates of the homogeneous log-signature components
$\ell_1,\ldots,\ell_4$ are recovered recursively from finitely many such jets. The
triangular identity is

\[
J_m(y;X)
=\rho(\ell_m)(y)
+\Psi_m(y;\ell_1,\ldots,\ell_{m-1}),
\]

so, at the certified degree-specific states,

\[
\lambda_m
=M_m^{-1}\bigl(\mathbf J_m-\boldsymbol\Psi_m\bigr).
\]

In deterministic degree-then-Lyndon order, the exact witness matrices are

\[
M_1\big|_{(0,q)}=
\begin{pmatrix}
0&1/2\\
1/2&1/2
\end{pmatrix},\qquad
M_2\big|_{(0)}=
\begin{pmatrix}-1/2\end{pmatrix},
\]

\[
M_3\big|_{(-q,q)}=
\begin{pmatrix}
15/32&3/16\\
15/32&-3/16
\end{pmatrix},
\]

and

\[
M_4\big|_{(-q,0,q)}=
\begin{pmatrix}
-57/128&-3/32&3/64\\
-1/2&0&1/4\\
-57/128&3/32&3/64
\end{pmatrix}.
\]

Their exact determinants are, respectively,

\[
-\frac14,\quad -\frac12,\quad -\frac{45}{256},\quad
\frac{135}{8192},
\]

and their exact inverses drive the certified triangular decoder. Thus the
general homogeneous-injectivity criterion becomes an exact decoder for the
concrete tanh witness.

## Eight-output degree-four truncated observation

The Hall-coordinate dimensions in degrees one through four are

\[
2+1+2+3=8.
\]

Let $z\in\mathbb R^8$ be ordered degree by degree and let $\Phi$ denote the
degree-four truncated observation polynomial for the ordered state-scale
design

\[
\mathcal D_*=\bigl(
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

The eight outputs are eight state-scale experiments applied to the same
unknown truncated Hall-coordinate vector. The exact zero-signature Jacobian
$B_*=D\Phi(0)$ satisfies

\[
\det B_*=-\frac{258733035}{524288}\ne0.
\]

Consequently, at the witness, there are neighborhoods $U$ of $z=0$ and $W$ of
$\Phi(0)$ such that $\Phi|_U:U\to W$ is a real-analytic diffeomorphism. This is
a local statement for the truncated polynomial, not global invertibility.

Keeping the same states and scales fixed, the zero-signature determinant is a
nontrivial real-analytic function of the four shallow-tanh parameters.
Therefore the same fixed design has an invertible zero-signature Jacobian for
Lebesgue-almost every parameter vector. The inverse neighborhood and its
stability constants may depend on the parameter and may degenerate near the
analytic determinant-zero set. The explicit constants below are for the
certified witness $\theta_*$.

The output count is dimension-matched in a precise topological sense: if
$U\subset\mathbb R^8$ is open and $M<8$, no continuous map
$U\to\mathbb R^M$ can be locally injective. This is a dimension lower bound
for continuous local injectivity on an open Hall-coordinate domain, not a
claim of universal measurement minimality.

## Cross-degree and scale structure

At the witness, the eight represented Hall fields through degree four have
cross-degree function rank exactly seven. Their exact relation is

\[
\rho\!\left(
4[X_1,X_2]
+3[X_1,[X_1,X_2]]
+2[[[X_1,X_2],X_2],X_2]
\right)=0.
\]

Thus homogeneous injectivity does not become ungraded injectivity, and an
all-at-one-nonzero-scale design is singular. A successful eight-output design
requires multiple scales; within the certified three-state grid, three
distinct scales are required and $\mathcal D_*$ uses scales 1, 2, and 3.

Under a common nonzero scale multiplier $h$, the degree multiplicities
$(2,1,2,3)$ give

\[
\det B(h)=h^{22}\det B(1).
\]

The exact inverse row-sum formula further yields

\[
\lVert B(h)^{-1}\rVert_\infty
=\Theta(|h|^{-4})
\qquad\text{as }h\to0.
\]

This is inverse-linearization amplification for the degree-four truncated
map, not exact condition-number scaling.

## Quantitative local stability

For the witness and the fixed design $\mathcal D_*$, exact coefficient-sum control of
$D\Phi(z)-B_*$, using the vector infinity norm and its induced matrix norm,
gives

\[
R_*=\frac1{2048},\qquad
\lVert B_*^{-1}\rVert_\infty=\frac{23812}{2151},
\]

\[
c_*=\frac{21220001406063}{409087045009408},\qquad
C_*=\frac{409087045009408}{21220001406063}.
\]

For every $z,z'\in\mathbb R^8$ satisfying

\[
\lVert z\rVert_\infty,\ \lVert z'\rVert_\infty\le\frac1{2048},
\]

the degree-four truncated observation obeys

\[
\lVert\Phi(z)-\Phi(z')\rVert_\infty
\ge c_*\lVert z-z'\rVert_\infty,
\]

and hence

\[
\lVert z-z'\rVert_\infty
\le C_*\lVert\Phi(z)-\Phi(z')\rVert_\infty.
\]

The radius $R_*$ is the largest passing radius in the prescribed dyadic
certificate list under the chosen coefficient-sum majorant. It is a
conservative certified radius and is not claimed to be sharp, truly maximal,
or the largest possible stable radius.

## Related work and novelty boundary

The closest prior work identified is Glückstad, Muça Cirone, and Teichmann,
[*Signature Reconstruction from Randomized Signatures* (2025)](https://arxiv.org/html/2502.03163v1).
It reconstructs finite tensor-signature levels from CDE/randomized-signature
observations under a word/tree-level linear-independence condition. Its proof
uses a scale parameter, differentiation at zero, homogeneous-level separation,
and finite initial-state selection, and its explicit positive construction
uses depth-two exponential vector fields.

Section 5 identifies direct recovery in compressed log-signature/Lie
coordinates as a further direction. Where its tensor-signature reconstruction
theorem applies, the corresponding truncated log-signature can also be
obtained indirectly through the classical tensor logarithm.

Bailleul and Diehl's
[*The Inverse Problem for Rough Controlled Differential Equations* (2015)](https://doi.org/10.1137/140995982)
establishes important prior art for inverse controlled/rough differential
equations. Free-Lie and Hall--Lyndon algebra, signature/log-signature algebra,
tensor exponential and logarithm, and Chen--Fliess expansions are classical.

Accordingly, no originality is claimed here for scale/Taylor arguments,
homogeneous-level separation, finite initial-state selection, inverse-CDE
problems, or ordinary tensor-signature reconstruction.

Relative to the closest prior work identified, this package works directly
with homogeneous free-Lie coordinates for the scalar shallow biased-tanh
architecture and establishes the exact witness, the fixed dimension-matched
eight-output truncated design, fixed-design genericity, and an exact
quantitative local inverse-Lipschitz certificate. This comparison delimits the
proved contribution and is not a literature-priority claim.

## Scope and limitations

The exact result concerns object B, the degree-four truncated observation
polynomial. Object C, the actual full CDE response, contains degree-five and
higher terms that the present theorem does not control. In particular, the
package does not establish reconstruction from ordinary full-CDE
observations.

The genuine limitations are:

- the concrete architecture theorem is restricted to two inputs, scalar
  state, and degrees at most four;
- the finite observation theorem is truncated rather than a theorem for the
  full CDE;
- stability is local rather than global;
- the explicit stability certificate is not parameter-uniform;
- there is no discrete-time theorem;
- there is no financial theorem; and
- there is no rough-path extension.

The detailed proofs and exact artifacts are recorded in
[the homogeneous certificate note](certificate_note.md) and
[exact certificate](certificate.json),
[the graded reconstruction derivation](graded_reconstruction_derivation.md),
[the finite-observation theorem](finite_observation_theorem.md), and
[the local-stability theorem](local_stability_theorem.md).

## Next mathematical question

Can the degree>=5 full-CDE remainder be bounded sharply enough, on an appropriately bounded path class, to transfer the truncated local stability theorem to actual shallow reservoir-flow observations?
