# Generic finite-degree word-field independence for shallow componentwise-tanh vector fields at an explicit sufficient state dimension

This note addresses one architecture question only: whether an explicit
sufficient state dimension guarantees generic finite-degree word-field
independence for shallow componentwise-tanh vector fields. It neither uses nor
proves a CDE reconstruction theorem. The existing scalar theorem and its
reconstruction and stability consequences remain separate.

## 1. Precise setting

Fix finite integers

\[
d\geq 1,
\qquad
L\geq 1.
\tag{1}
\]

Let \(e_1,\ldots,e_d\) be the standard basis of \(\mathbb R^d\), with ordered
alphabet \(1<\cdots<d\). Let

\[
T(\mathbb R^d):=\bigoplus_{m=0}^{\infty}(\mathbb R^d)^{\otimes m}
\]

be the tensor algebra with ordered concatenation
\(e_u e_v=e_{uv}\). Set \(X_i:=e_i\), and define

\[
\mathcal L_d
:=\operatorname{Lie}\langle X_1,\ldots,X_d\rangle
\subset T(\mathbb R^d),
\qquad
[a,b]:=ab-ba.
\tag{1a}
\]

Thus the associative-word realization is part of the convention. Define

\[
\mathcal L_{d,m}
:=\mathcal L_d\cap(\mathbb R^d)^{\otimes m},
\qquad
\mathcal L_d^{\leq L}
:=\bigoplus_{m=1}^L\mathcal L_{d,m}.
\tag{1b}
\]

For a word \(w=i_1\cdots i_m\) on \(\{1,\ldots,d\}\), put

\[
e_w:=e_{i_1}\otimes\cdots\otimes e_{i_m},
\qquad |w|=m,
\]

and let \(e_\varnothing\) denote the empty word. Define the truncated tensor
space

\[
E_{d,L}:=T^{\leq L}(\mathbb R^d)
=\bigoplus_{m=0}^L(\mathbb R^d)^{\otimes m}.
\tag{2}
\]

Its word basis is \(\{e_w:|w|\leq L\}\), and

\[
D_0(d,L):=\dim E_{d,L}=\sum_{m=0}^L d^m.
\tag{3}
\]

Order all nonempty words first by degree and then lexicographically within
each degree, using \(1<\cdots<d\), and write

\[
\mathcal W_{d,L}:=(w:1\leq |w|\leq L),
\qquad
Q_{d,L}:=|\mathcal W_{d,L}|=\sum_{m=1}^L d^m=D_0(d,L)-1.
\tag{4}
\]

This tensor-word count is distinct from the intrinsic Hall-coordinate count

\[
N_{d,L}:=\sum_{m=1}^L\dim\mathcal L_{d,m}
\]

used in the
[*general graded-identifiability note*](general_graded_identifiability.md).
Thus \(Q_{d,L}\) counts all nonempty associative words through degree \(L\),
whereas \(N_{d,L}\) counts free-Lie/Hall coordinates; the two symbols are not
interchangeable.

For a general state dimension \(D\), the full shallow componentwise-tanh
parameter space is

\[
\mathcal P_{d,D}
:=\prod_{i=1}^d
  \bigl(\mathbb R^{D\times D}\times\mathbb R^D\bigr)
\cong\mathbb R^{d(D^2+D)}.
\tag{5}
\]

For

\[
\theta=(A_1,b_1,\ldots,A_d,b_d)\in\mathcal P_{d,D},
\]

set

\[
V_i^\theta(x):=\tanh(A_i x+b_i),
\qquad i=1,\ldots,d,
\tag{6}
\]

where \(\tanh\) acts componentwise. Let \(\mathfrak X(\mathbb R^D)\) be the
Lie algebra of smooth vector fields with the repository bracket

\[
[V,W]:=DW\,V-DV\,W.
\tag{7}
\]

By the standard freeness of the Lie subalgebra in (1a), there is a unique
Lie-algebra homomorphism

\[
\rho_\theta:\mathcal L_d\longrightarrow\mathfrak X(\mathbb R^D)
\tag{8}
\]

satisfying

\[
\rho_\theta(X_i)=V_i^\theta.
\tag{9}
\]

Its homogeneous restrictions are

\[
\rho_{\theta,m}
:=\left.\rho_\theta\right|_{\mathcal L_{d,m}},
\qquad 1\leq m\leq L.
\tag{10}
\]

For a smooth vector-valued function \(F\), define

\[
\mathcal D_VF(x):=DF(x)V(x).
\tag{11}
\]

Operator multiplication is ordinary composition, \((AB)F=A(BF)\). For a
chronological word \(w=i_1\cdots i_m\), define

\[
\Gamma_\theta(w)
:=\mathcal D_{V_{i_1}^\theta}\cdots
  \mathcal D_{V_{i_m}^\theta},
\qquad
F_w^\theta:=\Gamma_\theta(w)\operatorname{Id}_{\mathbb R^D}.
\tag{12}
\]

Thus the first letter is the outermost derivation and the last letter is the
innermost derivation.

The four notions used below mean the following.

1. **Finite-degree word-field independence through \(L\).** The vector fields
   \(\{F_w^\theta:w\in\mathcal W_{d,L}\}\) are jointly linearly independent
   over \(\mathbb R\):

   \[
   \sum_{w\in\mathcal W_{d,L}}c_wF_w^\theta=0
   \quad\Longrightarrow\quad
   c_w=0\ \text{for every }w.
   \tag{13}
   \]

2. **Injectivity on the truncated free-Lie vector space.** The linear
   restriction

   \[
   \rho_\theta^{\leq L}
   :=\left.\rho_\theta\right|_{\mathcal L_d^{\leq L}}:
   \mathcal L_d^{\leq L}\longrightarrow
   \mathfrak X(\mathbb R^D),
   \qquad
   h\longmapsto\rho_\theta(h),
   \tag{14}
   \]

   is injective. Its domain is the indicated finite-dimensional graded vector
   subspace. In general, \(\rho_\theta\) need not annihilate
   \(\bigoplus_{m>L}\mathcal L_{d,m}\), so it need not factor through the free
   step-\(L\) nilpotent quotient.

3. **Homogeneous free-Lie injectivity through degree \(L\).** Every map
   \(\rho_{\theta,m}\) in (10) is injective for \(1\leq m\leq L\).

4. **Generic and Lebesgue-almost-every.** In this note a generic certified
   parameter set is the complement of the zero set of one nonzero
   real-analytic function on the full Euclidean space \(\mathcal P_{d,D}\).
   Such a complement is open, dense, and of full Lebesgue measure. Thus
   “generic” implies “Lebesgue-almost-every” here, with Lebesgue measure taken
   in all matrix and bias entries in (5).

No CDE, path, or path-regularity hypothesis occurs in this architecture
problem.

## 2. Exact linear word representation

Identify \(E_{d,L}\) with \(\mathbb R^{D_0(d,L)}\) through its word basis and
regard basis vectors as columns. Define truncated **right** concatenation by

\[
R_i e_w:=
\begin{cases}
e_{wi},&|w|<L,\\
0,&|w|=L.
\end{cases}
\tag{15}
\]

Equivalently, the matrix entry in row \(wi\), column \(w\), is one whenever
\(|w|<L\). Set

\[
W_i(x):=R_i x,
\qquad
x_0:=e_\varnothing.
\tag{16}
\]

These are the standard truncated-signature, or truncated tensor-system,
linear fields. Indeed, for a bounded-variation control \(X\), the controlled
linear system

\[
dY_t=\sum_{i=1}^d R_iY_t\,dX_t^i
=\pi_{\leq L}(Y_t\otimes dX_t),
\qquad
Y_s=e_\varnothing,
\]

where \(\pi_{\leq L}\) is truncation to degrees at most \(L\), has solution
\(Y_t=S_{\leq L}(X)_{s,t}\). In particular,

\[
[Y_t]_{i_1\cdots i_m}
=\int_{s<t_1<\cdots<t_m<t}
dX_{t_1}^{i_1}\cdots dX_{t_m}^{i_m},
\qquad m\leq L.
\]

Thus, from the empty word, the system records the truncated tensor signature
in its word coordinates. This classical identification is stated directly in
Akyildirim--Gambara--Teichmann--Zhou,
[Proposition 2.10](https://arxiv.org/abs/2201.02441). The standard reference
Friz--Victoir,
[*Multidimensional Stochastic Processes as Rough Paths*, Chapter 7, Section
7.2](https://doi.org/10.1017/CBO9780511845079.010), gives the step-\(L\)
signature equation in Proposition 7.8 and the corresponding linear fields in
Remark 7.9. The tensor/free-Lie setting is treated by Reutenauer,
[*Free Lie Algebras*, Theorem 0.5](https://academic.oup.com/book/53911).
No controlled-system statement is used in the tanh determinant proof below.

Let \(\Gamma_{\mathrm{lin}}\) be (12) with \(V_i=W_i\).

### Lemma 2.1 (repository-convention check for the standard linear system)

For every word \(w=i_1\cdots i_m\) with \(1\leq m\leq L\),

\[
\boxed{
\Gamma_{\mathrm{lin}}(i_1\cdots i_m)
\operatorname{Id}(x_0)=e_{i_1\cdots i_m}.}
\tag{17}
\]

**Proof.** Every operator-composition step can be displayed by introducing,
for \(1\leq r\leq m\),

\[
G_r(x):=
\mathcal D_{W_{i_r}}\mathcal D_{W_{i_{r+1}}}\cdots
\mathcal D_{W_{i_m}}\operatorname{Id}(x).
\tag{18}
\]

At the innermost step,

\[
G_m(x)
=D\operatorname{Id}(x)W_{i_m}(x)
=R_{i_m}x.
\tag{19}
\]

Suppose the already-composed suffix is

\[
G_{r+1}(x)=R_{i_m}R_{i_{m-1}}\cdots R_{i_{r+1}}x.
\tag{20}
\]

The next, outer derivation gives exactly

\[
\begin{aligned}
G_r(x)
&=\mathcal D_{W_{i_r}}G_{r+1}(x)\\
&=DG_{r+1}(x)W_{i_r}(x)\\
&=\bigl(R_{i_m}R_{i_{m-1}}\cdots R_{i_{r+1}}\bigr)R_{i_r}x\\
&=R_{i_m}R_{i_{m-1}}\cdots R_{i_r}x.
\end{aligned}
\tag{21}
\]

Induction from \(r=m\) down to \(r=1\) therefore yields

\[
\Gamma_{\mathrm{lin}}(i_1\cdots i_m)
\operatorname{Id}(x)
=R_{i_m}\cdots R_{i_2}R_{i_1}x.
\tag{22}
\]

The order in (22) is reversed at the matrix-product level because the
rightmost matrix acts first. At \(x_0\), the actual actions are

\[
e_\varnothing
\xmapsto{R_{i_1}}e_{i_1}
\xmapsto{R_{i_2}}e_{i_1i_2}
\xmapsto{R_{i_3}}\cdots
\xmapsto{R_{i_m}}e_{i_1\cdots i_m}.
\tag{23}
\]

No truncation occurs before the last displayed action because \(m\leq L\).
Equations (22)--(23) prove (17). \(\square\)

Accordingly, Lemma 2.1 is not claimed as a new representation-theoretic
result. Its role is to verify that this classical system agrees exactly with
the repository's chronological-word, operator-composition, and right-
concatenation conventions.

Using \(R_{i_1}\cdots R_{i_m}\) instead of (22) would reverse a two-letter
word and is not the repository convention.

The evaluation state \(x_0=e_\varnothing\) is the natural cyclic vector of
this truncated tensor module through degree \(L\): successive right
concatenations generate the exact chronological word basis as in (23).
Consequently the linear word-evaluation matrix at the normalized limit
\(\epsilon=0\) is the identity.

Define the square matrix \(A_0\in\mathbb R^{Q_{d,L}\times Q_{d,L}}\) by

\[
(A_0)_{u,w}
:=\left[
\Gamma_{\mathrm{lin}}(w)\operatorname{Id}(x_0)
\right]_u,
\qquad u,w\in\mathcal W_{d,L},
\tag{24}
\]

where \([\,\cdot\,]_u\) is the coordinate along \(e_u\), and rows and columns
use the same word order. By (17), column \(w\) is exactly \(e_w\) restricted
to the nonempty coordinates. Hence

\[
\boxed{A_0=I_{Q_{d,L}}.}
\tag{25}
\]

In particular, the linear word fields are jointly independent. Indeed, if

\[
\sum_{w\in\mathcal W_{d,L}}c_w
\Gamma_{\mathrm{lin}}(w)\operatorname{Id}=0
\]

as a vector field, evaluation at \(x_0\) and restriction to the nonempty word
coordinates give \(A_0c=c=0\).

This is finite-degree word-field independence. No tree-indexed object has
been introduced or identified.

## 3. Exact linear-to-tanh transfer

On \(E_{d,L}\), define the actual shallow-tanh fields

\[
V_i^\epsilon(x):=\tanh(\epsilon R_i x).
\tag{26}
\]

For \(\epsilon\neq0\), introduce the auxiliary normalized fields

\[
\widetilde V_i^\epsilon(x)
:=\frac{\tanh(\epsilon R_i x)}{\epsilon},
\tag{27}
\]

and set

\[
\widetilde V_i^0(x):=R_i x.
\tag{28}
\]

Only \(V_i^\epsilon\), not \(\widetilde V_i^\epsilon\), is asserted to be an
element of the architecture (6).
As \(\epsilon\to0\), the normalized fields and all finite state jets used
below converge to the standard linear fields \(W_i\), while

\[
V_i^\epsilon=\epsilon\widetilde V_i^\epsilon\longrightarrow0.
\]

Hence the actual tanh family is a vanishing-amplitude degeneration of the
standard truncated-signature linear system, with that system visible only
after normalization by \(\epsilon\).

### Lemma 3.1 (analytic normalized finite jets)

For every state multi-index \(\alpha\) needed through word length \(L\)—in
particular for every \(|\alpha|\leq L-1\)—the finite jet

\[
(\epsilon,x)\longmapsto
\partial_x^\alpha\widetilde V_i^\epsilon(x)
\tag{29}
\]

is real analytic near \((0,x_0)\).

**Proof.** It is enough to consider the scalar function

\[
q(\epsilon,z):=
\begin{cases}
\tanh(\epsilon z)/\epsilon,&\epsilon\neq0,\\
z,&\epsilon=0.
\end{cases}
\tag{30}
\]

The odd Taylor expansion of \(\tanh\) at zero gives a convergent expansion

\[
q(\epsilon,z)=
\sum_{r=0}^\infty a_r\epsilon^{2r}z^{2r+1},
\qquad a_0=1,
\tag{31}
\]

on a neighborhood of every point \((0,z_*)\). Thus \(q\) is jointly real
analytic there. Componentwise,

\[
\widetilde V_i^\epsilon(x)_a
=q\bigl(\epsilon,(R_ix)_a\bigr),
\tag{32}
\]

so it is jointly real analytic near \((0,x_0)\). Finite differentiation in
\(x\) preserves real analyticity, proving (29). \(\square\)

Let \(\Gamma_\epsilon\) denote the word operators generated by the actual
fields \(V_i^\epsilon\), and let \(\widetilde\Gamma_\epsilon\) denote those
generated by the auxiliary fields \(\widetilde V_i^\epsilon\).

### Lemma 3.2 (exact word-length factorization)

For every \(w\in\mathcal W_{d,L}\), and every \(\epsilon\neq0\),

\[
\boxed{
\Gamma_\epsilon(w)\operatorname{Id}
=\epsilon^{|w|}
\widetilde\Gamma_\epsilon(w)\operatorname{Id}.}
\tag{33}
\]

**Proof.** The equality \(V_i^\epsilon=\epsilon\widetilde V_i^\epsilon\)
implies, for every smooth \(F\),

\[
\mathcal D_{V_i^\epsilon}F(x)
=DF(x)\bigl(\epsilon\widetilde V_i^\epsilon(x)\bigr)
=\epsilon\mathcal D_{\widetilde V_i^\epsilon}F(x).
\tag{34}
\]

The scalar \(\epsilon\) is independent of \(x\). Consequently, when the next
outer derivation differentiates an already-composed function, it does not
differentiate that scalar. Factoring one scalar from each of the \(|w|\)
derivations in the ordinary operator product proves (33). \(\square\)

For \(\epsilon\neq0\), define the normalized square matrix

\[
A(\epsilon)_{u,w}
:=\left[
\epsilon^{-|w|}
\Gamma_\epsilon(w)\operatorname{Id}(x_0)
\right]_u,
\qquad u,w\in\mathcal W_{d,L}.
\tag{35}
\]

By (33), its \(w\)-column is

\[
\left[
\widetilde\Gamma_\epsilon(w)\operatorname{Id}(x_0)
\right]_{u\in\mathcal W_{d,L}}.
\tag{36}
\]

Use (36) to define \(A(0)\). The recursion

\[
\widetilde F_{i_1\cdots i_m}^\epsilon(x)
=D_x\widetilde F_{i_2\cdots i_m}^\epsilon(x)
\widetilde V_{i_1}^\epsilon(x),
\qquad
\widetilde F_i^\epsilon=\widetilde V_i^\epsilon,
\tag{37}
\]

uses only finitely many of the analytic jets in Lemma 3.1. Induction on word
length therefore proves that every entry of \(A(\epsilon)\) is real analytic
near zero. At zero the auxiliary fields are the linear fields (16), so the
cyclic-vector identity of Section 2 gives the exact normalized linear-limit
basis identity

\[
\boxed{A(0)=A_0=I_{Q_{d,L}},\qquad \det A(0)=1.}
\tag{38}
\]

Continuity of the determinant gives some \(\epsilon_0>0\) such that

\[
\det A(\epsilon)\neq0
\qquad\text{whenever}\qquad
0<|\epsilon|<\epsilon_0.
\tag{39}
\]

The family of architecture parameters

\[
A_i=\epsilon R_i,
\qquad b_i=0,
\qquad i=1,\ldots,d,
\]

is explicit as a one-parameter family, and continuity proves that every
nonzero member with sufficiently small \(|\epsilon|\) is admissible. The proof
does not supply a numerical value of \(\epsilon_0\), however, and it certifies
no particular exact or numerical nonzero value of \(\epsilon\). Thus it gives
existence inside this family, not a fully explicit tanh parameter point. Also,
\(A(0)=I_{Q_{d,L}}\) is the normalized auxiliary limit: the actual fields
\(V_i^0\) vanish and \(A_{\mathrm{raw}}(0)=0\).

Let \(A_{\mathrm{raw}}(\epsilon)\) be the same matrix without the column
normalization in (35). Then

\[
A_{\mathrm{raw}}(\epsilon)
=A(\epsilon)
\operatorname{diag}_{w\in\mathcal W_{d,L}}
  \bigl(\epsilon^{|w|}\bigr).
\tag{40}
\]

There are \(d^m\) words of length \(m\). Therefore the exact determinant
scaling is

\[
\boxed{
\det A_{\mathrm{raw}}(\epsilon)
=\epsilon^{S(d,L)}\det A(\epsilon),
\qquad
S(d,L):=\sum_{m=1}^L m d^m.}
\tag{41}
\]

Since \(\det A(\epsilon)\to1\), equation (41) also shows that the raw
certificate degenerates to zero at the rate \(\epsilon^{S(d,L)}\) as
\(\epsilon\to0\). This observation supplies no conditioning or stability
estimate and is not used in place of the continuity argument (39).

In particular, (39) implies

\[
\det A_{\mathrm{raw}}(\epsilon)\neq0
\qquad (0<|\epsilon|<\epsilon_0).
\tag{42}
\]

## 4. Strongest consequences of the same matrix

Fix one \(\epsilon\) satisfying (39).
Let

\[
\rho_\epsilon:\mathcal L_d\longrightarrow
\mathfrak X(E_{d,L})
\tag{43}
\]

be the unique Lie-algebra homomorphism satisfying
\(\rho_\epsilon(X_i)=V_i^\epsilon\), and write
\(\rho_{\epsilon,m}:=\rho_\epsilon|_{\mathcal L_{d,m}}\).

First, the finite family of word fields

\[
F_w^\epsilon
:=\Gamma_\epsilon(w)\operatorname{Id},
\qquad w\in\mathcal W_{d,L},
\tag{44}
\]

are jointly linearly independent. Indeed, a vanishing constant linear
combination of these vector fields, evaluated at \(x_0\) in the nonempty word
coordinates, lies in the kernel of \(A_{\mathrm{raw}}(\epsilon)\), which is
zero by (42).

Second, this same certificate gives injectivity on the truncated free-Lie
vector space. To verify compatibility between the commutator convention (1a)
and the vector-field sign, for scalar \(f\) compute

\[
\begin{aligned}
[\mathcal D_V,\mathcal D_W]_{\mathrm{op}}f
&=\mathcal D_V(\mathcal D_Wf)
  -\mathcal D_W(\mathcal D_Vf)\\
&=D^2f(V,W)+Df\,DW\,V
  -D^2f(W,V)-Df\,DV\,W\\
&=Df\,(DW\,V-DV\,W)\\
&=\mathcal D_{[V,W]}f.
\end{aligned}
\tag{45}
\]

Extend \(\Gamma_\epsilon\) linearly from words to finite tensor polynomials,
with \(\Gamma_\epsilon(1)\) equal to the identity operator. The chronological
definition (12) first gives the associative operator identity

\[
\Gamma_\epsilon(uv)
=\Gamma_\epsilon(u)\Gamma_\epsilon(v)
\tag{45a}
\]

for all words \(u,v\), and hence for their linear extensions.

We now prove the operator-level identity

\[
\Gamma_\epsilon(h)=\mathcal D_{\rho_\epsilon(h)}
\tag{45b}
\]

on scalar smooth functions, for every Lie polynomial \(h\in\mathcal L_d\).
For a generator, (45b) is

\[
\Gamma_\epsilon(X_i)
=\mathcal D_{V_i^\epsilon}
=\mathcal D_{\rho_\epsilon(X_i)}.
\]

If (45b) holds for Lie polynomials \(a\) and \(b\), then (1a), (45a), and
(45) give

\[
\begin{aligned}
\Gamma_\epsilon([a,b])
&=\Gamma_\epsilon(a)\Gamma_\epsilon(b)
  -\Gamma_\epsilon(b)\Gamma_\epsilon(a)\\
&=[\mathcal D_{\rho_\epsilon(a)},
    \mathcal D_{\rho_\epsilon(b)}]_{\mathrm{op}}\\
&=\mathcal D_{[\rho_\epsilon(a),\rho_\epsilon(b)]}\\
&=\mathcal D_{\rho_\epsilon([a,b])}.
\end{aligned}
\]

Structural induction on bracketed Lie monomials, followed by linearity, proves
(45b). Only \(h\in\mathcal L_d^{\leq L}\) is used in the independence argument.
Applying (45b) coordinatewise to the identity map now gives, and only now
gives,

\[
\rho_\epsilon(h)
=\mathcal D_{\rho_\epsilon(h)}\operatorname{Id}
=\Gamma_\epsilon(h)\operatorname{Id},
\qquad h\in\mathcal L_d^{\leq L}.
\tag{46}
\]

The same parameter-independent induction applies to
\(\Gamma_\theta,\rho_\theta\) for every \(\theta\in\mathcal P_{d,D}\). In
particular, (46) is not being propagated directly under brackets; it is a
consequence of the stronger operator identity (45b).

Now take

\[
0\neq h\in\mathcal L_d^{\leq L}.
\]

Because \(\mathcal L_d\) was defined as a subspace of the tensor algebra,
\(h\) has a unique nonzero word-basis expansion

\[
h=\sum_{w\in\mathcal W_{d,L}}c_we_w\neq0.
\tag{47}
\]

Equations (44), (46), and (47) give

\[
\rho_\epsilon(h)=\sum_{w\in\mathcal W_{d,L}}c_wF_w^\epsilon\neq0.
\tag{48}
\]

Thus \(\rho_\epsilon^{\leq L}\) is injective on the truncated free-Lie vector
space. Restricting an injective linear map to any homogeneous summand proves
homogeneous free-Lie injectivity through degree \(L\).

The order of strength is therefore

\[
\boxed{
\text{finite-degree word-field independence}
\Longrightarrow
\text{injectivity on the truncated free-Lie vector space}
\Longrightarrow
\text{homogeneous free-Lie injectivity through degree }L.}
\tag{49}
\]

Finite-degree word-field independence is stronger than required for direct
Hall/log-signature reconstruction. No Hall-minor construction is needed here,
and no tree-indexed independence statement is made. The certified
\(d=2,D=1,L\leq4\) witness has a known nonzero cross-degree kernel through
degree four without contradicting (49): it is a different, substantially
smaller-state-dimension construction, whereas (49) concerns the tensor-space
family at dimension \(D_0(d,L)\), with its admissible nonzero \(\epsilon\)
selected existentially by continuity.

## 5. Genericity in the full shallow-tanh family at \(D=D_0(d,L)\)

Choose existentially one sufficiently small nonzero \(\epsilon\) supplied by
(39), without assigning it a numerical or exact value, set
\(x_0^{(D_0)}:=x_0=e_\varnothing\), and define the architecture parameter

\[
\theta_\epsilon:
\qquad
A_i=\epsilon R_i,
\qquad
b_i=0,
\qquad i=1,\ldots,d.
\tag{50}
\]

Then (6) is exactly (26). Keep the evaluation state
\(x_0^{(D_0)}=e_\varnothing\), the word columns \(\mathcal W_{d,L}\), and the
same nonempty word-coordinate rows. For \(u\in\mathcal W_{d,L}\), define the
fixed output-coordinate evaluation

\[
\lambda_u^{(D_0)}(F):=\bigl[F(x_0^{(D_0)})\bigr]_u.
\]

For an arbitrary \(\theta\in\mathcal P_{d,D_0}\), define

\[
M(\theta)_{u,w}
:=\lambda_u^{(D_0)}
\bigl(\Gamma_\theta(w)\operatorname{Id}\bigr),
\qquad u,w\in\mathcal W_{d,L},
\tag{51}
\]

and

\[
\Delta_{d,L,D_0}(\theta):=\det M(\theta).
\tag{52}
\]

Every entry in (51) is real analytic in the complete parameter vector. Here
is a direct finite proof. The map

\[
(\theta,x)\longmapsto V_i^\theta(x)=\tanh(A_ix+b_i)
\]

is jointly real analytic. For \(w=i_1v\), ordinary operator composition gives
the exact recursion

\[
F_{i_1v}^\theta(x)
=D_xF_v^\theta(x)V_{i_1}^\theta(x),
\qquad
F_i^\theta=V_i^\theta.
\tag{53}
\]

Finite differentiation, multiplication, and composition preserve real
analyticity. Induction on \(|w|\leq L\) therefore shows that
\((\theta,x)\mapsto F_w^\theta(x)\) is real analytic. Evaluation at the fixed
state and projection to a fixed coordinate prove the asserted
analyticity of (51). A finite determinant of analytic entries is analytic, so
(52) is real analytic on all of \(\mathcal P_{d,D_0}\).

At the constructed point,

\[
M(\theta_\epsilon)=A_{\mathrm{raw}}(\epsilon),
\qquad
\Delta_{d,L,D_0}(\theta_\epsilon)\neq0
\tag{54}
\]

by (42). Thus \(\Delta_{d,L,D_0}\) is not identically zero. The standard
real-analytic zero-set theorem says that the zero set of a nonzero
real-analytic function on a connected open subset of Euclidean space has
Lebesgue measure zero. Hence

\[
\mathcal G_{d,L,D_0}^{\mathrm{cert}}
:=\{\theta\in\mathcal P_{d,D_0}:
      \Delta_{d,L,D_0}(\theta)\neq0\}
\tag{55}
\]

is an open, dense, full-Lebesgue-measure certificate set. For every parameter
in this set, the same evaluation argument as in Section 4 proves finite-degree
word-field independence, injectivity on the truncated free-Lie vector space,
and homogeneous free-Lie injectivity through degree \(L\).

Let \(\mathcal B_{d,L,D_0}\) denote the actual parameter locus where
finite-degree word-field independence fails. The fixed determinant proves only

\[
\mathcal B_{d,L,D_0}
\subseteq\{\Delta_{d,L,D_0}=0\}.
\]

Equality is not asserted: when this particular determinant vanishes, another
evaluation state or another coordinate minor may still certify independence.

This conclusion is qualitative. It gives no conditioning estimate; the fixed
determinant may deteriorate rapidly with \(d\) and \(L\); and it supplies no
parameter-uniform neighborhood or stability constant. The theorem gives the
explicit sufficient state dimension \(D_0(d,L)\), which is only a conservative
sufficient width, and proves no width lower bound. In particular, it does not
address compressed state dimensions \(D\ll d^L\).

## 6. Extension to every \(D>D_0(d,L)\)

Let

\[
D=D_0(d,L)+q,
\qquad q\geq1,
\]

and decompose

\[
\mathbb R^D=E_{d,L}\oplus\mathbb R^q.
\]

Block-embed the tensor-space parameter family by

\[
A_i^{(D)}
:=
\begin{pmatrix}
\epsilon R_i&0\\
0&0_q
\end{pmatrix},
\qquad
b_i^{(D)}=0,
\tag{56}
\]

and use

\[
x_0^{(D)}:=(e_\varnothing,0).
\tag{57}
\]

Writing a state as \((x,z)\in E_{d,L}\oplus\mathbb R^q\), componentwise
application of \(\tanh\) gives

\[
V_i^{(D)}(x,z)
=\bigl(\tanh(\epsilon R_ix),0\bigr)
=\bigl(V_i^\epsilon(x),0\bigr).
\tag{58}
\]

Let \(\Gamma_D\) be the chronological word operators generated by the block
fields \(V_1^{(D)},\ldots,V_d^{(D)}\) using the convention (12).
For every \(w\in\mathcal W_{d,L}\), induction on word length using (53) yields
the identity

\[
\Gamma_D(w)\operatorname{Id}_{\mathbb R^D}(x,z)
=\bigl(
  \Gamma_\epsilon(w)\operatorname{Id}_{E_{d,L}}(x),
  0
 \bigr).
\tag{59}
\]

Indeed, (58) is the one-letter case. If (59) holds for a suffix \(v\), its
derivative maps \((u,v')\) to
\((D_xF_v^\epsilon(x)u,0)\); applying it to
\(V_{i_1}^{(D)}(x,z)=(V_{i_1}^\epsilon(x),0)\) proves the case \(i_1v\).

At \(x_0^{(D)}\), use the same \(Q_{d,L}\) output coordinates belonging to
the nonempty word basis in the first block. Equation (59) says that the
resulting minor is exactly

\[
A_{\mathrm{raw}}(\epsilon),
\tag{60}
\]

so it is nonzero.

For \(u\in\mathcal W_{d,L}\), let

\[
\lambda_u^{(D)}(F):=\bigl[F(x_0^{(D)})\bigr]_u,
\]

where \([\,\cdot\,]_u\) is the coordinate in the embedded \(E_{d,L}\) block.
For a general parameter \(\theta\in\mathcal P_{d,D}\), define

\[
M_D(\theta)_{u,w}
:=\lambda_u^{(D)}
\bigl(\Gamma_\theta(w)\operatorname{Id}_{\mathbb R^D}\bigr),
\qquad u,w\in\mathcal W_{d,L},
\]

using these same fixed rows, columns, and evaluation state, and set

\[
\Delta_{d,L,D}(\theta):=\det M_D(\theta).
\tag{61}
\]

The finite recursion (53) proves that (61) is real analytic in all entries of
all full \(D\times D\) matrices and all full \(D\)-dimensional biases. Its
value at (56) is the nonzero determinant in (60). Therefore it is not
identically zero, and the same analytic zero-set theorem shows that

\[
\mathcal G_{d,L,D}^{\mathrm{cert}}
:=\{\theta\in\mathcal P_{d,D}:
       \Delta_{d,L,D}(\theta)\neq0\}
\tag{62}
\]

is an open, dense, full-Lebesgue-measure certificate set. Finite-degree
word-field independence, injectivity on the truncated free-Lie vector space,
and homogeneous free-Lie injectivity through degree \(L\) follow exactly as in
Section 4. As at \(D_0\), the actual failure locus is contained in
\(\{\Delta_{d,L,D}=0\}\), but equality is not asserted.

This is an inactive-coordinate block embedding at the constructed parameter
point: only there are the added coordinates forced to be inactive. It
preserves the selected nonzero minor and thereby proves non-identical
vanishing and genericity in the complete larger-dimensional parameter space;
generic parameters need not retain the block form. The argument does not show
that added coordinates improve expressivity, monotonicity in \(D\), a width
lower bound, or parameter-uniform conditioning. It is not the old scalar
\(d=2,D=1\) state-dimension lift.

## 7. Exact theorem and limitations

### Theorem 7.1 (explicit sufficient state dimension for generic finite-degree word-field independence)

Let \(d,L\geq1\) be finite and let

\[
D\geq D_0(d,L)=\sum_{k=0}^Ld^k.
\tag{63}
\]

For the complete shallow componentwise-tanh parameter space
\(\mathcal P_{d,D}\), there exist one fixed evaluation state
\(x_0^{(D)}\) and the fixed output-coordinate evaluations

\[
\lambda_u^{(D)}(F)=\bigl[F(x_0^{(D)})\bigr]_u,
\qquad u\in\mathcal W_{d,L},
\]

where the coordinates belong to the embedded \(E_{d,L}\) block, such that

\[
\boxed{
\Delta_{d,L,D}(\theta)
:=\det\!\left[
\lambda_u^{(D)}\!\left(
\Gamma_\theta(w)\operatorname{Id}_{\mathbb R^D}
\right)
\right]_{u,w\in\mathcal W_{d,L}}}
\tag{64}
\]

is a nontrivial real-analytic function of the complete parameter vector

\[
\theta=(A_1,b_1,\ldots,A_d,b_d)\in\mathcal P_{d,D}.
\]

The fixed-design certificate set

\[
\mathcal G_{d,L,D}^{\mathrm{cert}}
:=\{\theta\in\mathcal P_{d,D}:\Delta_{d,L,D}(\theta)\neq0\}
\tag{65}
\]

is open, dense, and of full Lebesgue measure. For every
\(\theta\in\mathcal G_{d,L,D}^{\mathrm{cert}}\), the finite family

\[
\left\{
\Gamma_\theta(w)\operatorname{Id}_{\mathbb R^D}:
1\leq|w|\leq L
\right\}
\tag{66}
\]

is jointly linearly independent. Consequently,

\[
\rho_\theta^{\leq L}:
\mathcal L_d^{\leq L}\longrightarrow\mathfrak X(\mathbb R^D)
\tag{67}
\]

is injective on the truncated free-Lie vector space. In particular, each
\(\rho_{\theta,m}\) is injective for \(1\leq m\leq L\); this is homogeneous
free-Lie injectivity through degree \(L\).

If \(\mathcal B_{d,L,D}\) is the actual parameter locus where finite-degree
word-field independence fails, then

\[
\mathcal B_{d,L,D}\subseteq\{\Delta_{d,L,D}=0\}.
\tag{68}
\]

Equality is not asserted. A parameter can fail this fixed determinant
certificate while a different evaluation state or coordinate minor still
proves finite-degree word-field independence.

**Proof.** At \(D=D_0(d,L)\), the normalized word matrix \(A(\epsilon)\) is
real analytic near zero by Lemma 3.1 and the finite recursion (37). The cyclic
linear identity gives \(A(0)=I\). Hence continuity, and only continuity, gives
\(\det A(\epsilon)\neq0\) for all sufficiently small nonzero \(\epsilon\), as
in (39). This proves existence of a shallow-tanh parameter point in the
one-parameter family, without supplying a specific numerical or exact
\(\epsilon\), where the raw fixed minor is nonzero. Section 4 gives the three
stated independence implications,
and Section 5 applies analytic zero-set genericity in the complete parameter
space. The inactive-coordinate block embedding in Section 6 preserves the
same nonzero minor and repeats the full-parameter argument for every
\(D>D_0(d,L)\). No all-\(\epsilon\) triangular determinant formula or
nonlinear triangularity observation is used. \(\square\)

The conclusion is qualitative and finite-degree. Precisely:

- no result is claimed for words of length greater than \(L\); in particular,
  no word-field or free-Lie injectivity claim is made there, and the tanh
  system need not factor through the free step-\(L\) nilpotent quotient;
- no tree-level independence statement is made;
- \(D_0(d,L)\) is the explicit sufficient state dimension and only a
  conservative sufficient width; no width lower bound is proved;
- no exact or numerical admissible nonzero \(\epsilon\) is certified by the
  general proof;
- no conditioning estimate is obtained, and the determinant may deteriorate
  rapidly with \(d\) and \(L\);
- no parameter-uniform neighborhood or stability constant is proved;
- the inactive-coordinate extension gives no expressivity monotonicity or
  lower bound in \(D\);
- the compressed regime \(D\ll d^L\) is untouched;
- no ranking against a depth-two or any other architecture is asserted; and
- no actual full-CDE reconstruction, discrete-time reconstruction, finance
  consequence, or parameter-uniform stability statement is made.

## 8. Related work and novelty boundary

### 8.1 Classical inputs

The following ingredients are classical and are not architecture-specific.

1. The truncated-signature linear system, its tensor-word basis, and its
   unit/empty-word initial state are standard. Precise standard locators are
   Friz--Victoir, [Chapter 7, Section 7.2, Proposition 7.8 and Remark
   7.9](https://doi.org/10.1017/CBO9780511845079.010), and
   Akyildirim--Gambara--Teichmann--Zhou,
   [Proposition 2.10](https://arxiv.org/abs/2201.02441). Lemma 2.1 checks only
   the repository-specific chronological and right-concatenation orientation.
2. The constructive nilpotent antecedent is the Birkhoff truncated-
   enveloping-module construction. If \(\mathfrak g\) is finite-dimensional
   and nilpotent of class \(c\), left multiplication gives a faithful action on
   the finite-dimensional weighted quotient
   \(U(\mathfrak g)/U^{c+1}(\mathfrak g)\); see Garrett
   Birkhoff, [*Representability of Lie Algebras and Lie Groups by
   Matrices*](https://doi.org/10.2307/1968569), *Annals of Mathematics* 38
   (1937), 526--532, and Dietrich Burde, Bettina Eick, and Willem de Graaf,
   [*Computing Faithful Representations for Nilpotent Lie
   Algebras*](https://arxiv.org/abs/0807.2345), *Journal of Algebra* 322
   (2009), 602--612, Section 2, Theorem 2.1. For the free step-\(L\) algebra,
   the free-Lie/PBW identification identifies the underlying weighted quotient
   through degree \(L\) with the truncated tensor-word space used here. The
   cited construction uses left multiplication
   and the usual endomorphism commutator; this note uses right concatenation
   together with (7), so its action orientation is opposite and is still
   proved directly in Section 2.
3. Reutenauer,
   [*Free Lie Algebras*](https://academic.oup.com/book/53911), Oxford
   University Press, 1993, Theorem 0.5, identifies
   \(U(\mathcal L_d)\) with \(T(\mathbb R^d)\) and realizes
   \(\mathcal L_d\) as the Lie subalgebra generated by the letters. Chapters
   1, 4, and 5 treat the word algebra, Hall bases, and Lyndon words and bases,
   respectively. Those results supply
   the classical algebraic setting and standard coordinates; they are not the
   tanh-transfer argument.
4. Classical finite-dimensional actions can be substantially smaller than the
   full truncated tensor module. Burde--Eick--de Graaf, Examples 2.2--2.3,
   treat the free two-generator, step-two algebra: the standard weighted
   enveloping quotient has dimension \(7=D_0(2,2)\), while a faithful
   three-dimensional module is obtained. Hence its minimal faithful
   representation dimension is at most \(3<7\). This does not imply that such
   a smaller module realizes finite-degree word-field independence or admits a
   shallow componentwise-tanh realization.
5. A nontrivial real-analytic function on a connected Euclidean open set has a
   Lebesgue-null zero set; see Steven G. Krantz and Harold R. Parks,
   *A Primer of Real Analytic Functions*, second edition, Birkhäuser, 2002.
   Sections 3, 5, and 6 prove the architecture-specific analyticity and
   non-identical vanishing required before this theorem is invoked.

Thus the classical linear module is prior. The architecture-specific content
here is only its normalized transfer into the shallow componentwise-tanh
family and the resulting nontrivial generic finite-degree determinant.

### 8.2 Akyildirim--Gambara--Teichmann--Zhou (2022)

Erdinc Akyildirim, Matteo Gambara, Josef Teichmann, and Syang Zhou,
[*Applications of Signature Methods to Market Anomaly Detection*](https://arxiv.org/abs/2201.02441)
(2022), record in Proposition 2.10 the standard truncated-signature linear
system used for context in Section 2. They also introduce the shallow
randomized-signature architecture as a feature extractor and state a broad
tensor-algebra injectivity/signature-recovery claim. Their proof invokes
real-analytic zero-set genericity but does not
establish the load-bearing non-identical-vanishing step by exhibiting a
parameter point at which the relevant relation or determinant is nonzero.
The later 2025 paper below identifies iterated-field independence as the
delicate premise in that sketch and notes that it is generally false without
finite-degree and dimensional qualifications. Moreover, the 2022 theorem
assumes a bounded real-analytic activation with infinite radius of convergence;
those assumptions do not directly provide the present componentwise-tanh
theorem.

### 8.3 Glückstad--Muça Cirone--Teichmann (2025)

Mie Glückstad, Nicola Muça Cirone, and Josef Teichmann,
[*Signature Reconstruction from Randomized Signatures*](https://arxiv.org/abs/2502.03163v1)
(2025), state a finite-level tensor-signature reconstruction theorem under a
tree-like vector-field independence hypothesis. The logical implication used there
is

\[
\text{tree-level independence}
\quad\Longrightarrow\quad
\text{word-level independence}.
\]

More precisely, their Lemma 3.4 expands each word operator into tree-like
operators, and Proposition 3.6 proves that per-degree independence of the
tree-like vector fields implies per-degree independence of the associated
iterated word operators. The present theorem instead proves joint
finite-degree word-field independence for
\(F_w^\theta=\Gamma_\theta(w)\operatorname{Id}\); it does not prove
tree-level independence.

They construct depth-two nested-exponential vector fields with exponential
activation for which their tree-level condition holds through level \(L\) when
their hidden-state dimension \(N\) satisfies \(N\geq L-1\). That symbol \(N\)
is analogous to the state dimension \(D\) here and is unrelated to either the
word count \(Q_{d,L}\) or the intrinsic Hall count \(N_{d,L}\). Their no-depth
analysis also exhibits a tree-level dependence obstruction for the classical
shallow bias-free form \(V_i(x)=\sigma(A_i x)\). This does not contradict the
present word-level result, because failure of tree-level independence does not
imply failure of word-level independence.

Their Remark 3.7 also gives a universal finite-dimensional iterated
word-operator relation: for
\(d=3\), hidden dimension \(1\), and word length \(3\), every choice of smooth
one-dimensional vector fields satisfies, in their notation,

\[
V_{123}+V_{231}+V_{312}
=V_{132}+V_{213}+V_{321}.
\]

This example confirms that some dimension/order restriction is necessary in
general; it does not establish sharp dimensional bounds. Section 5 of
that paper leaves direct recovery in Lie/log-signature coordinates open. The
depth-two construction and the present shallow componentwise-tanh construction
use different architectures and different independence conditions; no ranking
between them is asserted.

### 8.4 Constrained contribution statement

The targeted primary-source audit identified no prior proof of this
finite-degree shallow componentwise-tanh word-field independence and
genericity result at the stated sufficient width.

The result is technically new but conceptually expected and serves as the
general structural theorem supporting the direct Hall-coordinate
identifiability package. No priority claim is made.

## 9. Relation to the existing \(d=2,D=1,L\leq4\) result

The existing scalar result is a separate construction at the much smaller
state dimension \(D=1\) for the homogeneous-injectivity objective in the
\(d=2,L\leq4\) regime. It is not used anywhere in the proof of Theorem 7.1.
It establishes homogeneous
free-Lie injectivity through degree four at \(D=1\), and its eight-output Hall
decoder and quantitative local stability are separate strengths that are not
redeveloped here.

For the homogeneous-injectivity objective, the comparison with \(D=1\) shows
that \(D_0(2,4)=31\) is a conservative sufficient width. It does not give
a smaller state dimension for the stronger finite-degree word-field
independence conclusion of Theorem 7.1. Its certified cross-degree relation is
therefore consistent with the present theorem. The two results complement
rather than replace one another.

## 10. Final status

PROVED

The continuity-based proof is complete: \(A(\epsilon)\) is real analytic,
\(A(0)=I\), and hence \(\det A(\epsilon)\neq0\) for all sufficiently small
nonzero \(\epsilon\). All standard external inputs and scope limitations are
identified above.
