# General graded identifiability and state-dimension lifting

## Scope and exact status

This note proves one general structural theorem and one concrete lifting
corollary.

1. For arbitrary finite input dimension \(d\), state dimension \(D\), and
   truncation level \(L\), homogeneous free-Lie injectivity through level
   \(L\) is sufficient for intrinsic, degree-by-degree reconstruction from
   finitely many ideal normalized scale jets. The same premise gives a
   dimension-matched finite state-scale-coordinate observation map for the
   degree-\(L\) truncated response, with a nonsingular zero-signature
   Jacobian and a local analytic inverse.
2. The frozen scalar \(d=2,\ L\leq4\) shallow-tanh witness embeds into every
   state dimension \(D\geq1\). This gives explicit nonzero homogeneous minors
   and almost-everywhere separate homogeneous injectivity in the full
   \(D\)-state shallow-tanh parameter space.

The first result is conditional on the injectivity, on the observation domain,
of every homogeneous map \(\rho_{\theta,m}\). It does not prove that arbitrary
shallow-tanh architectures satisfy that premise. The second result is an
embedding theorem: the added coordinates are inactive at the witness.

Three observation objects are used and kept distinct:

- **A. Ideal normalized graded scale jets**, obtained by differentiating the
  actual response to the scaled input \(rX\) at \(r=0\).
- **B. The degree-\(L\) truncated response**, a finite tensor polynomial in
  truncated Hall coordinates.
- **C. The actual full CDE terminal response** \(Y_T^{y,r}\) at nonzero
  scale, which contains all homogeneous levels.

Nothing below identifies object B at nonzero scale with object C.

## Part I. General setting and repository conventions

### 1. State space, regularity, and local solutions

Fix integers

\[
d\geq1,\qquad D\geq1,\qquad L\geq1,
\]

let \(U\subseteq\mathbb R^D\) be nonempty and open, fix a parameter
\(\theta\), and let

\[
V_1^\theta,\ldots,V_d^\theta
\in C^\infty(\mathbb R^D;\mathbb R^D)
\]

be known global vector fields. Write
\(V_i:=V_i^\theta|_U\). Global smoothness ensures that the canonical
free-Lie representation below is defined on all of \(\mathcal L_d\). The
finite-level CDE argument uses only \(C^L\) regularity on \(U\): a level-\(m\)
word coefficient applied to a state coordinate uses derivatives only through
order \(m-1\), while derivatives through order \(L\) give the \(C^L\)
scale-dependence and the order-\(L+1\) remainder. No lower-regularity theorem
is claimed here.

Let \(X=(X^1,\ldots,X^d):[0,T]\to\mathbb R^d\) be continuous and of bounded
variation, and put

\[
\omega_X:=\sum_{i=1}^d\lVert X^i\rVert_{1\text{-var};[0,T]}.
\]

For \(y\in U\), choose a closed Euclidean ball \(K_y\subset U\) centered at
\(y\). Since the \(V_i\) are bounded and locally Lipschitz on \(K_y\), there
is \(\varepsilon_y>0\) such that, for \(|r|<\varepsilon_y\), the
Riemann--Stieltjes controlled ODE

\[
\begin{aligned}
dY_t^{y,r}
  &=\sum_{i=1}^d V_i(Y_t^{y,r})\,d(rX_t^i)
    =r\sum_{i=1}^d V_i(Y_t^{y,r})\,dX_t^i,\\
Y_0^{y,r}&=y
\end{aligned}
\tag{1}
\]

has a unique solution on \([0,T]\) whose range stays in \(K_y\). Indeed, if
\(K_y\) has radius \(a_y\), put
\[
M_y:=\max_{1\leq i\leq d}\sup_{x\in K_y}\lVert V_i(x)\rVert.
\]
Up to a first exit from \(K_y\),
\[
\sup_{t\leq T}\lVert Y_t^{y,r}-y\rVert
\leq |r|M_y\omega_X.
\tag{2}
\]
Choosing \(|r|M_y\omega_X<a_y\) rules out that exit.

On the open subset of \(C([0,T];\mathbb R^D)\) consisting of paths with range
in the interior of \(K_y\), the equation is the zero set of

\[
\mathcal F(r,Z)_t
=Z_t-y-r\sum_{i=1}^d\int_0^tV_i(Z_s)\,dX_s^i.
\]

Near the constant path \(Z^0_t=y\), this map is \(C^L\) and
\(D_Z\mathcal F(0,Z^0)=I\). The Banach-space implicit-function theorem
therefore shows that \(r\mapsto Y^{y,r}\) is \(C^L\) near \(0\). Define the
normalized vector scale jets

\[
J_m(y;X)
:=\frac1{m!}\left.\frac{\partial^m}{\partial r^m}
Y_T^{y,r}\right|_{r=0}\in\mathbb R^D,
\qquad 1\leq m\leq L.
\tag{3}
\]

For \(1\leq k\leq D\), the scalar \(J_m(y;X)_k\) is one observation
functional. No state-rescaling identity is used.

### 2. Tensor words, signatures, and Hall coordinates

Let \(E=\operatorname{span}\{X_1,\ldots,X_d\}\). A word
\(w=i_1\cdots i_m\) denotes the tensor monomial
\(X_w=X_{i_1}\cdots X_{i_m}\). Tensor multiplication is ordered
concatenation:

\[
X_uX_v=X_{uv}.
\tag{4}
\]

Let \(\pi_m\) denote projection onto homogeneous tensor degree \(m\), and
let \(\pi_{\leq L}:=\sum_{m=0}^L\pi_m\) denote truncation through degree
\(L\).

The chronological signature convention is

\[
S^{i_1\cdots i_m}(X)
:=\int_{0<t_1<\cdots<t_m<T}
dX_{t_1}^{i_1}\cdots dX_{t_m}^{i_m},
\tag{5}
\]

so the earliest differential is the leftmost letter. There is no factorial
in (5). Write

\[
S_m(X)=\sum_{|w|=m}S^w(X)X_w,
\qquad
S^{\leq L}(X)=1+\sum_{m=1}^L S_m(X).
\]

Input dilation gives \(S_m(rX)=r^mS_m(X)\).

The tensor exponential and logarithm use

\[
\exp_\otimes(a)=\sum_{q\geq0}\frac{a^q}{q!},
\qquad
\log_\otimes(1+a)
=\sum_{q\geq1}\frac{(-1)^{q+1}}q a^q,
\tag{6}
\]

with concatenation multiplication and truncation after level \(L\).
Chen's group-like theorem gives

\[
\ell^{\leq L}(X)
:=\log_\otimes S^{\leq L}(X)
=\ell_1(X)+\cdots+\ell_L(X),
\qquad
\ell_m(X)\in\mathcal L_{d,m},
\tag{7}
\]

where \(\mathcal L_{d,m}\) is the degree-\(m\) component of the free Lie
algebra on \(X_1,\ldots,X_d\).

For each degree, use the repository-compatible standard-bracketed Lyndon
basis, ordered lexicographically within degree; it is a homogeneous Hall
basis. Denote it by

\[
\mathcal H_{d,m}=\{H_{m,1},\ldots,H_{m,n_m}\},
\qquad
n_m:=\dim\mathcal L_{d,m},
\tag{8}
\]

and put

\[
\mathcal H_{d,L}:=\bigcup_{m=1}^L\mathcal H_{d,m},
\qquad
\mathfrak g_d^{(L)}
:=\pi_{\leq L}\mathcal L_d
=\bigoplus_{m=1}^L\mathcal L_{d,m},
\qquad
N_{d,L}:=\sum_{m=1}^L n_m.
\tag{9}
\]

Here degrees above \(L\) are quotiented out, and the Lie bracket on
\(\mathfrak g_d^{(L)}\) is
\([a,b]_{(L)}:=\pi_{\leq L}(ab-ba)\). Thus
\(\mathfrak g_d^{(L)}\) is the free step-\(L\) nilpotent Lie algebra.

Witt's formula is

\[
n_m=\frac1m\sum_{q\mid m}\mu(q)d^{m/q}.
\tag{10}
\]

The associative bracket is

\[
[u,v]=uv-vu.
\tag{11}
\]

### 3. Operator representation and bracket sign

For a vector field \(V\), let

\[
\mathcal D_V f(y):=Df(y)V(y)
\]

be its pullback derivation on scalar functions. Operator multiplication is
ordinary composition, \((AB)f=A(Bf)\). Define, for chronological words,

\[
\Gamma(X_i)=\mathcal D_{V_i},
\qquad
\Gamma(X_{i_1}\cdots X_{i_m})
=\mathcal D_{V_{i_1}}\circ\cdots\circ\mathcal D_{V_{i_m}}.
\tag{12}
\]

Extend \(\Gamma\) linearly to tensor polynomials and set \(\Gamma(1)=I\).

Thus

\[
\Gamma(uv)=\Gamma(u)\Gamma(v);
\tag{13}
\]

\(\Gamma\) is a homomorphism for concatenation, not an anti-homomorphism.
All operator identities below are understood on scalar test functions with
enough derivatives for the displayed total word degree; the coordinate maps
\(p_k\) are smooth.

For \(V,W\), cancellation of the Hessian terms gives

\[
\begin{aligned}
[\mathcal D_V,\mathcal D_W]_{\mathrm{op}}f
&=\mathcal D_V(\mathcal D_Wf)-\mathcal D_W(\mathcal D_Vf)\\
&=Df\bigl(DW\,V-DV\,W\bigr).
\end{aligned}
\tag{14}
\]

Accordingly, the vector-field bracket is exactly

\[
[V,W]:=DW\,V-DV\,W.
\tag{15}
\]

Let \(\mathfrak X(\mathbb R^D)\) be the Lie algebra of smooth vector fields
on \(\mathbb R^D\) with bracket (15). By the universal property of the free
Lie algebra,

\[
\rho_\theta:\mathcal L_d\longrightarrow\mathfrak X(\mathbb R^D)
\tag{16}
\]

is the unique Lie-algebra homomorphism satisfying

\[
\rho_\theta(X_i)=V_i^\theta,\qquad 1\leq i\leq d.
\]

Its homogeneous restrictions are

\[
\boxed{
\rho_{\theta,m}
:=\left.\rho_\theta\right|_{\mathcal L_{d,m}},
\qquad 1\leq m\leq L.}
\tag{16a}
\]

If
\(\operatorname{Id}=(p_1,\ldots,p_D)\) is the vector of state-coordinate
maps, then, coordinatewise,

\[
\Gamma(h)=\mathcal D_{\rho_\theta(h)|_U},
\qquad
\Gamma(h)\operatorname{Id}=\rho_\theta(h)|_U
\tag{17}
\]

for every Lie polynomial \(h\) of degree at most \(L\). This follows by
induction from (14). In particular, (11)--(15) reproduce the repository sign
without a word reversal.

## Part II. Arbitrary-level triangular reconstruction

### 4. Chen--Fliess coefficients and scale-jet normalization

For a scalar state coordinate \(p_k(y)=y_k\), the
Riemann--Stieltjes chain rule gives

\[
p_k(Y_t^{y,r})
=p_k(y)+r\sum_{i=1}^d\int_0^t
(\mathcal D_{V_i}p_k)(Y_s^{y,r})\,dX_s^i.
\tag{18}
\]

Repeated substitution fixes both orders: an earlier differential is placed
to the left in the word and its derivation is placed to the left in the
operator product. Through level \(L\),

\[
p_k(Y_T^{y,r})
=p_k(y)+\sum_{m=1}^L r^m
\sum_{|w|=m}S^w(X)\,[\Gamma(w)p_k](y)
+R_{L+1,k}(r).
\tag{19}
\]

The remainder is an \((L+1)\)-fold iterated integral with an overall factor
\(r^{L+1}\). More precisely,

\[
\begin{aligned}
R_{L+1,k}(r)
={}&r^{L+1}
\sum_{i_1,\ldots,i_{L+1}=1}^d
\int_{0<t_1<\cdots<t_{L+1}<T}\\
&\quad
\bigl[
\Gamma(X_{i_1}\cdots X_{i_{L+1}})p_k
\bigr](Y_{t_1}^{y,r})
\,
dX_{t_1}^{i_1}\cdots dX_{t_{L+1}}^{i_{L+1}}.
\end{aligned}
\]

On the compact ball \(K_y\), put

\[
C_{y,k}:=
\max_{1\leq i_1,\ldots,i_{L+1}\leq d}
\sup_{x\in K_y}
\left|
\bigl[
\Gamma(X_{i_1}\cdots X_{i_{L+1}})p_k
\bigr](x)
\right|.
\]

This is finite because \(V_i\in C^L\). Since a continuous BV control has
nonatomic variation measure,

\[
|R_{L+1,k}(r)|
\leq
\frac{C_{y,k}}{(L+1)!}|r|^{L+1}\omega_X^{L+1}
\tag{20}
\]

Combining (19), (20), and the \(C^L\) dependence on \(r\) proves

\[
\boxed{
J_m(y;X)
=\bigl[\Gamma(S_m(X))\operatorname{Id}\bigr](y),
\qquad 1\leq m\leq L.}
\tag{21}
\]

The normalization \(1/m!\) in (3) removes the Taylor factorial. The simplex
normalization already contained in the signature is not repeated.

### 5. Ordered-composition formula

Let

\[
\ell=\ell_1+\cdots+\ell_L,
\qquad \ell_j\in\mathcal L_{d,j}.
\]

Taking the homogeneous degree-\(m\) part of (6) yields

\[
\boxed{
\pi_m\exp_\otimes(\ell)
=\sum_{q=1}^m\frac1{q!}
\sum_{\substack{j_1+\cdots+j_q=m\\j_a\geq1}}
\ell_{j_1}\ell_{j_2}\cdots\ell_{j_q}.}
\tag{22}
\]

The inner sum is over ordered compositions of \(m\). Its factors cannot be
commuted. The unique \(q=1\) term is \(\ell_m\); in every term with
\(q\geq2\), each \(j_a\leq m-1\).

Define the lower-degree correction, in vector form, by

\[
\boxed{
\Psi_m(y;\ell_1,\ldots,\ell_{m-1})
:=
\sum_{q=2}^m\frac1{q!}
\sum_{\substack{j_1+\cdots+j_q=m\\j_a\geq1}}
\bigl[
\Gamma(\ell_{j_1}\cdots\ell_{j_q})
\operatorname{Id}
\bigr](y),}
\tag{23}
\]

with the empty sum \(\Psi_1=0\). Equations (7), (21), and (22) give the
arbitrary-level triangular identity

\[
\boxed{
J_m(y;X)
=\rho_\theta(\ell_m(X))(y)
+\Psi_m(y;\ell_1(X),\ldots,\ell_{m-1}(X)),
\quad 1\leq m\leq L.}
\tag{24}
\]

The degree-\(m\) unknown enters linearly through
\(\rho_{\theta,m}\); all other terms depend only on strictly lower
homogeneous degrees.

### 6. Finite scalar evaluations and recursive recovery

We use one elementary lemma.

**Finite point-coordinate evaluation lemma.** Let
\(f_1,\ldots,f_n:\Omega\to\mathbb R^D\) be linearly independent as
vector-valued functions. Then there are \(n\) point-coordinate pairs

\[
(\omega_a,k_a)\in\Omega\times\{1,\ldots,D\}
\]

such that \(\bigl[f_j(\omega_a)_{k_a}\bigr]_{a,j=1}^n\) is nonsingular.

**Proof.** On \(E_0=\operatorname{span}\{f_1,\ldots,f_n\}\), the scalar
evaluations \(f\mapsto f(\omega)_k\) have common kernel zero. Their span is
therefore all of the finite-dimensional dual \(E_0^*\); otherwise a nonzero
element of \(E_0\) would annihilate that span. Select \(n\) evaluations
forming a dual basis. \(\square\)

We can now state the first main result.

**Theorem 6.1 (general graded identifiability, conditional form).** Fix
finite integers \(d,D,L\geq1\), \(T>0\), a nonempty open observation domain
\(U\subseteq\mathbb R^D\), a parameter \(\theta\), and the Hall, tensor,
operator, and bracket conventions of Parts I--II. Assume:

1. the global fields \(V_1^\theta,\ldots,V_d^\theta\) belong to
   \(C^\infty(\mathbb R^D;\mathbb R^D)\), and the fields, their derivatives
   needed through level \(L\), and the representation \(\rho_\theta\) are
   known exactly;
2. \(X:[0,T]\to\mathbb R^d\) is continuous and of bounded variation;
3. at every observation state used below, the scaled Riemann--Stieltjes CDE
   (1) has a unique solution on \([0,T]\) for \(r\) near zero, its range
   remains in a compact subset of \(U\), and
   \(r\mapsto Y_T^{y,r}\) is \(C^L\);
4. the observer has exact access to the normalized coordinate jets
   \(J_m(y;X)_k\) at the finitely many state-coordinate pairs selected below;
5. for every \(1\leq m\leq L\), the observation-domain restriction

   \[
   h\longmapsto \left.\rho_{\theta,m}(h)\right|_U
   \tag{25}
   \]

   is injective.

Then there are \(n_m\) state-coordinate pairs
\((y_{m,a},k_{m,a})\in U\times\{1,\ldots,D\}\) for each \(m\) such that all
Hall coordinates of
\(\ell_1(X),\ldots,\ell_L(X)\) are recovered uniquely and recursively from
the \(N_{d,L}\) ideal scalar jets

\[
J_m(y_{m,a};X)_{k_{m,a}}.
\]

The selected pairs depend only on the known vector fields, not on the
unknown path. The conclusion concerns object A only.

**Proof.** Write

\[
\ell_m=\sum_{j=1}^{n_m}\lambda_{m,j}H_{m,j}.
\tag{26}
\]

Injectivity in (25) makes the vector fields
\(\rho_\theta(H_{m,1})|_U,\ldots,\rho_\theta(H_{m,n_m})|_U\) linearly
independent. Apply the finite evaluation lemma and define

\[
(M_m)_{aj}
:=\rho_\theta(H_{m,j})(y_{m,a})_{k_{m,a}}.
\tag{27}
\]

Then \(M_m\) is nonsingular. At degree one, (24) gives
\(M_1\lambda_1=\mathbf J_1\). If the blocks below degree \(m\) have already
been recovered, (23) determines the vector

\[
(\boldsymbol\Psi_m)_a
:=\Psi_m(y_{m,a};\ell_1,\ldots,\ell_{m-1})_{k_{m,a}}.
\]

Evaluation of (24) gives

\[
\mathbf J_m-\boldsymbol\Psi_m=M_m\lambda_m,
\]

and hence

\[
\boxed{
\lambda_m
=M_m^{-1}
\bigl(\mathbf J_m-\boldsymbol\Psi_m\bigr),
\qquad 1\leq m\leq L.}
\tag{28}
\]

Induction proves the claim. \(\square\)

If \(n_m=0\), the degree-\(m\) selection and recursion are vacuous. Since
shallow componentwise-tanh fields are \(C^\infty\), Theorem 6.1 applies
verbatim to every such parameter vector for which (25) holds.

This is global algebraic recovery on the formal Hall-coordinate space for
object A. It assumes exact derivatives at \(r=0\); it is not a
finite-difference procedure. It neither replaces those ideal jets by the
finite-scale truncated map B nor asserts reconstruction from actual full-CDE
values C.

### 7. Formal necessity and bounded-variation realizability

The homogeneous injectivity premise also has a precise converse.

**Theorem 7.1 (homogeneous kernel invisibility).** Suppose that, for some
\(m\leq L\),

\[
0\neq h\in\mathcal L_{d,m},
\qquad
\left.\rho_{\theta,m}(h)\right|_U=0.
\tag{29}
\]

Then for every \(z\in\mathfrak g_d^{(L)}\) and every \(t\in\mathbb R\), the
formal coordinates \(z\) and \(z+th\) produce identical graded observations
at every level through \(L\), at every state \(y\in U\), and in every state
coordinate. They also produce the same degree-\(L\) truncated response on
\(U\).

**Proof.** By (17), the vanishing in (29) gives
\(\Gamma(h)=\mathcal D_{\rho_\theta(h)|_U}=0\) as an operator. Introduce a
grading variable \(u\) and the homogeneous dilation

\[
\delta_u z:=\sum_{j=1}^L u^jz_j.
\tag{30}
\]

Every term in

\[
\pi_{\leq L}\exp_\otimes(\delta_u(z+th))
-\pi_{\leq L}\exp_\otimes(\delta_u z)
\]

contains at least one factor \(h\). Since \(\Gamma\) is an algebra
homomorphism and \(\Gamma(h)=0\), applying \(\Gamma\) annihilates the entire
difference. Thus

\[
\Gamma\!\left(
\pi_{\leq L}\exp_\otimes(\delta_u(z+th))\right)
=
\Gamma\!\left(
\pi_{\leq L}\exp_\otimes(\delta_u z)\right).
\tag{31}
\]

The coefficient of \(u^j\) is
\(\Gamma(\pi_j\exp_\otimes(z+th))\) on the left and
\(\Gamma(\pi_j\exp_\otimes z)\) on the right. Equality for
\(j=1,\ldots,L\), followed by application to every coordinate map and
evaluation at every state, proves the claim. \(\square\)

Consequently, if one observation-domain restriction
\(\rho_{\theta,m}(\,\cdot\,)|_U\) has a nonzero kernel, even the complete
graded observation system is not locally injective on any open subset of
\(\mathfrak g_d^{(L)}\): for any interior \(z\), sufficiently small nonzero
\(t\) keeps \(z+th\) in the same open set.

The standard finite-step path-realizability theorem transfers this formal
necessity statement to bounded-variation path signatures:

\[
\boxed{
\{S^{\leq L}(X):X\text{ is continuous and piecewise linear}\}
=G_d^{(L)}
=\exp_\otimes(\mathfrak g_d^{(L)}).}
\tag{32}
\]

Here \(G_d^{(L)}\) is the connected, simply connected free step-\(L\)
nilpotent group in the truncated tensor algebra. A segment of increment
\(v\in\mathbb R^d\) has truncated signature \(\exp_\otimes(v)\), and Chen
concatenation turns a piecewise-linear path into a finite product of such
exponentials. The horizontal first layer generates the whole free nilpotent
Lie algebra; the Rashevsky--Chow/orbit theorem, equivalently the corresponding
Lie-subgroup theorem, turns this Lie generation into finite-product
reachability of the whole group. The nilpotent exponential is a global
diffeomorphism. Hence separate piecewise-linear BV paths realize \(z\) and
\(z+th\).

Thus (29) gives genuine nonidentifiability on the realized truncated
signature domain for objects A and B. The realization theorem does not give
a continuous local right inverse into a chosen path topology.

## Part III. General state-scale-coordinate observation

### 8. Joint independence across homogeneous degrees

Fix any nonempty open interval

\[
I\subset\mathbb R\setminus\{0\}.
\]

For \(h\in\mathcal H_{d,L}\), define the scalar function on

\[
\Omega:=I\times U\times\{1,\ldots,D\}
\]

by

\[
f_h(r,y,k):=r^{\deg h}\rho_\theta(h)(y)_k.
\tag{33}
\]

**Theorem 8.1 (joint state-scale-coordinate independence).** Under (25),
the \(N_{d,L}\) functions \(f_h\), \(h\in\mathcal H_{d,L}\), are linearly
independent.

**Proof.** Suppose that \(\sum_hc_hf_h=0\), and group the coefficients by
degree:

\[
h_m:=\sum_{h\in\mathcal H_{d,m}}c_hh.
\]

For every fixed \(y,k\),

\[
\sum_{m=1}^L r^m\rho_\theta(h_m)(y)_k=0
\qquad\text{for every }r\in I.
\tag{34}
\]

The left side is a polynomial in \(r\) vanishing on an interval, so every
coefficient vanishes. Hence \(\rho_\theta(h_m)(y)_k=0\) for every \(y,k\);
therefore \(\rho_{\theta,m}(h_m)|_U=0\). Assumption (25) gives \(h_m=0\),
and Hall-basis independence gives all \(c_h=0\). \(\square\)

This argument uses separate homogeneous injectivity only. It makes no claim
of ungraded independence of the fields
\(y\mapsto\rho_\theta(h)(y)\).

### 9. A dimension-matched truncated observation map

For \(z=z_1+\cdots+z_L\in\mathfrak g_d^{(L)}\), define object B by

\[
\begin{aligned}
F_y^{(L)}(r;z)
&:=
\left[
\Gamma\!\left(
\pi_{\leq L}\exp_\otimes(\delta_rz)\right)
\operatorname{Id}
\right](y)\\
&=
y+\sum_{m=1}^Lr^m
\left[
\Gamma\!\left(\pi_m\exp_\otimes z\right)
\operatorname{Id}
\right](y).
\end{aligned}
\tag{35}
\]

Apply the scalar finite-evaluation lemma to the independent family in
Theorem 8.1. There are exactly \(N_{d,L}\) triples

\[
(r_a,y_a,k_a)\in I\times U\times\{1,\ldots,D\},
\qquad 1\leq a\leq N_{d,L},
\tag{36}
\]

such that

\[
A_{a,h}:=
r_a^{\deg h}\rho_\theta(h)(y_a)_{k_a}
\tag{37}
\]

is a nonsingular \(N_{d,L}\times N_{d,L}\) matrix. Define

\[
\Phi(z)_a:=F_{y_a}^{(L)}(r_a;z)_{k_a}.
\tag{38}
\]

If \(h\) is homogeneous of degree \(m\), then

\[
\pi_{\leq L}\exp_\otimes(\delta_r(\varepsilon h))
=1+\varepsilon r^mh+O(\varepsilon^2),
\]

so

\[
\partial_hF_y^{(L)}(r;0)_k
=r^m\rho_\theta(h)(y)_k.
\tag{39}
\]

Equations (37)--(39) imply

\[
D\Phi(0)=A,
\qquad
\det D\Phi(0)\neq0.
\tag{40}
\]

For fixed fields and a fixed design (36), \(\Phi\) is a finite polynomial in
the \(N_{d,L}\) Hall coordinates: analyticity of the vector fields in the
state variable is not required. The real-analytic inverse-function theorem
therefore gives neighborhoods

\[
0\in\mathcal O\subset\mathfrak g_d^{(L)},
\qquad
\Phi(0)\in\mathcal W\subset\mathbb R^{N_{d,L}},
\]

such that

\[
\boxed{
\Phi|_{\mathcal O}:\mathcal O\longrightarrow\mathcal W
\text{ is a real-analytic diffeomorphism}.}
\tag{41}
\]

The design is existential and representation-dependent. No fixed generic
design, quantitative radius, or parameter-uniform stability constant is
deduced for arbitrary \(d,D,L\). This local inverse concerns the truncated
finite-scale map B only; it is not an inverse for the actual full CDE response
C.

### 10. The exact codomain-dimension statement

Let \(\mathcal O_0\) be a nonempty open subset of
\(\mathfrak g_d^{(L)}\simeq\mathbb R^{N_{d,L}}\). If a continuous map

\[
G:\mathcal O_0\longrightarrow\mathbb R^M
\]

is injective on some open neighborhood of a point, then

\[
\boxed{M\geq N_{d,L}.}
\tag{42}
\]

Indeed, if \(M<N_{d,L}\), compose the injective restriction with the
coordinate embedding
\(\mathbb R^M\hookrightarrow\mathbb R^{N_{d,L}}\). Brouwer invariance of
domain would make the image open in \(\mathbb R^{N_{d,L}}\), but the image
lies in an \(M\)-dimensional coordinate plane with empty
\(N_{d,L}\)-dimensional interior.

This counts scalar codomain dimensions only. It is not a lower bound on
distinct runs, initial states, scale values, or vector-valued observations,
and it is not a universal measurement-minimality claim.

## Part IV. Lifting the exact scalar witness to every state dimension

### 11. The block-embedded shallow-tanh fields

Now fix \(d=2\), work through degree four (and hence through every
truncation \(L\leq4\)), and let \(D\geq1\). The full \(D\)-state shallow
componentwise-tanh family is

\[
V_i^{A,b}(x)=\tanh(A_ix+b_i),
\qquad
A_i\in\mathbb R^{D\times D},\quad b_i\in\mathbb R^D,
\quad i=1,2,
\tag{43}
\]

where \(\tanh\) acts componentwise. Its full parameter space is

\[
\Theta_D
=\bigl(\mathbb R^{D\times D}\times\mathbb R^D\bigr)^2
\simeq\mathbb R^{2D(D+1)}.
\tag{44}
\]

Put

\[
q:=\frac{\log3}{2},
\qquad
\theta_*^{(1)}
:=(a_1,b_1,a_2,b_2)=(1,0,0,q),
\tag{45}
\]

and define the block witness

\[
A_i^{(D)}=\operatorname{diag}(a_i,0,\ldots,0),
\qquad
b_i^{(D)}=(b_i,0,\ldots,0).
\tag{46}
\]

Write
\(\theta_*^{(D)}:=(A_1^{(D)},b_1^{(D)},A_2^{(D)},b_2^{(D)})\).
With \(e_1=(1,0,\ldots,0)\), the two fields are

\[
V_1^{(D)}(x)=\tanh(x_1)e_1,
\qquad
V_2^{(D)}(x)=\frac12e_1.
\tag{47}
\]

Define the global embedding of scalar fields by

\[
E(f)(x):=f(x_1)e_1.
\tag{48}
\]

If \(f,g:\mathbb R\to\mathbb R\), then

\[
D(Eg)(x)E(f)(x)=g'(x_1)f(x_1)e_1,
\]

and hence, with the repository bracket,

\[
[E(f),E(g)]
=E(g'f-f'g)
=E([f,g]_{\mathrm{scalar}}).
\tag{49}
\]

Thus \(E\) is an injective Lie homomorphism. Since the generator fields in
(47) are the \(E\)-images of the scalar witness fields, induction on every
Hall bracket gives the stronger global identity. Write
\(\rho_*^{(1)}:=\rho_{\theta_*^{(1)}}\) and
\(\rho_*^{(D)}:=\rho_{\theta_*^{(D)}}\). Then

\[
\boxed{
\rho_*^{(D)}(h)(x)
=E\!\left(\rho_*^{(1)}(h)\right)(x)
=\rho_*^{(1)}(h)(x_1)e_1}
\tag{50}
\]

for every Hall element \(h\). In particular,

\[
\boxed{
\bigl(\rho_*^{(D)}(h)(xe_1)\bigr)_1
=\rho_*^{(1)}(h)(x).}
\tag{51}
\]

Proving the global identity (50), rather than assuming only equality on the
axis, is essential because a bracket depends on state derivatives.

The embedding also preserves the frozen scalar cross-degree relation. With

\[
H_2=[X_1,X_2],\qquad
H_{31}=[X_1,[X_1,X_2]],\qquad
H_{43}=[[[X_1,X_2],X_2],X_2],
\]

equation (50) gives

\[
\boxed{
\rho_{\theta_*^{(D)}}(4H_2+3H_{31}+2H_{43})
=E\!\left(
\rho_{\theta_*^{(1)}}(4H_2+3H_{31}+2H_{43})
\right)
=0.}
\tag{51a}
\]

The Lie element in (51a) is nonzero because its nonzero components have
distinct degrees \(2,3,4\). Thus the lift has homogeneous injectivity
separately through degree four, but the restriction of
\(\rho_{\theta_*^{(D)}}\) to
\(\bigoplus_{m=1}^4\mathcal L_{2,m}\) is not injective.

### 12. Exact lifted homogeneous minors and full-parameter genericity

Use the frozen scalar standard-Lyndon columns and replace every scalar
evaluation at \(y\) by the point-coordinate evaluation

\[
F\longmapsto F(ye_1)_1.
\tag{52}
\]

By (50), the lifted matrices are exactly

\[
M_1^{(D)}=
\begin{pmatrix}
0&1/2\\
1/2&1/2
\end{pmatrix}
\quad\text{at }0e_1,\ qe_1,
\qquad
M_2^{(D)}=
\begin{pmatrix}-1/2\end{pmatrix}
\quad\text{at }0e_1,
\tag{53}
\]

\[
M_3^{(D)}=
\begin{pmatrix}
15/32&3/16\\
15/32&-3/16
\end{pmatrix}
\quad\text{at }-qe_1,\ qe_1,
\tag{54}
\]

and

\[
M_4^{(D)}=
\begin{pmatrix}
-57/128&-3/32&3/64\\
-1/2&0&1/4\\
-57/128&3/32&3/64
\end{pmatrix}
\quad\text{at }-qe_1,\ 0e_1,\ qe_1.
\tag{55}
\]

Their exact determinants are

\[
-\frac14,\qquad
-\frac12,\qquad
-\frac{45}{256},\qquad
\frac{135}{8192},
\tag{56}
\]

with product

\[
-\frac{6075}{16777216}\neq0.
\tag{57}
\]

These are complete-basis homogeneous minors of sizes \(2,1,2,3\).
Therefore \(\rho_{\theta_*^{(D)},m}\) is injective separately for
\(m=1,2,3,4\).

For a general parameter
\(\theta=(A_1,b_1,A_2,b_2)\in\Theta_D\), let
\(\Delta_{m,D}(\theta)\) be the determinant formed from the same Hall
columns and the same point-coordinate rows (52). Each entry is real analytic
jointly in all \(2D(D+1)\) parameter coordinates: it is obtained from
finitely many state derivatives, sums, and products of the
componentwise-tanh fields. Thus

\[
P_D(\theta):=\prod_{m=1}^4\Delta_{m,D}(\theta)
\tag{58}
\]

is real analytic on the connected space \(\Theta_D\). Equation (57) shows
that \(P_D\) is not identically zero. The zero set of a nontrivial
real-analytic function has Lebesgue measure zero. Outside that zero set, all
four fixed minors are nonsingular, and therefore:

\[
\boxed{
\text{For Lebesgue-almost every }\theta\in\Theta_D,\quad
\rho_{\theta,m}\text{ is injective separately for }m=1,2,3,4.}
\tag{59}
\]

The noninjective locus is contained in \(\{P_D=0\}\); equality is not
asserted. The fact that the displayed witness is sparse causes no problem:
one nonzero value proves nonidentity on the full parameter space.

### 13. Lift of the eight-output truncated design and exact constants

The Hall-bracket identity (50) proves equality of the zero Jacobians, but
retaining the nonlinear truncated stability constants requires equality of
the complete truncated polynomials. Let \(p_1(x)=x_1\). For a scalar
function \(\varphi\),

\[
\mathcal D_{E(f)}(\varphi\circ p_1)
=(\mathcal D_f\varphi)\circ p_1.
\tag{60}
\]

Write \(\Gamma_1\) and \(\Gamma_D\) for the associative operator
representations at the scalar and block witnesses. Induction on chronological
word length, using (12), gives the global function identity

\[
\Gamma_D(w)p_1
=
\bigl(\Gamma_1(w)\operatorname{Id}_{\mathbb R}\bigr)\circ p_1.
\]

In particular,

\[
\bigl[\Gamma_D(w)p_1\bigr](ye_1)
=\bigl[\Gamma_1(w)\operatorname{Id}_{\mathbb R}\bigr](y)
\tag{61}
\]

for every word \(w\), hence for every tensor polynomial through level four.

Lift the frozen ordered scalar design by using coordinate one at every row:

\[
\begin{aligned}
\mathcal D_*^{(D)}=\bigl(
&(-qe_1,1,1),(-qe_1,2,1),(-qe_1,3,1),\\
&(0,2,1),(0,3,1),\\
&(qe_1,1,1),(qe_1,2,1),(qe_1,3,1)
\bigr),
\end{aligned}
\tag{62}
\]

where each triple records an initial state, a scale, and the observed
coordinate. Let
\(\Phi_D:\mathbb R^8\to\mathbb R^8\) be the associated degree-four
truncated observation map at the block witness, and let \(\Phi_1\) be the
frozen scalar map. Equation (61) proves the polynomial identity

\[
\boxed{\Phi_D(z)=\Phi_1(z)\quad\text{for every }z\in\mathbb R^8.}
\tag{63}
\]

Consequently the lifted zero-signature Jacobian is exactly the scalar matrix
\(B_*\), with

\[
\boxed{\det B_*=-\frac{258733035}{524288}\neq0.}
\tag{64}
\]

The determinant is the frozen value from the
[finite-observation theorem](finite_observation_theorem.md). The exact
scalar witness-specific infinity-norm certificate from the
[local-stability theorem](local_stability_theorem.md) is retained without
change:

\[
R_*=\frac1{2048},
\qquad
\lVert B_*^{-1}\rVert_\infty=\frac{23812}{2151},
\tag{65}
\]

\[
c_*=
\frac{21220001406063}{409087045009408},
\qquad
C_*=
\frac{409087045009408}{21220001406063}.
\tag{66}
\]

Thus, for the lifted coordinate-one output map and
\(\lVert z\rVert_\infty,\lVert z'\rVert_\infty\leq1/2048\),

\[
\lVert\Phi_D(z)-\Phi_D(z')\rVert_\infty
\geq c_*\lVert z-z'\rVert_\infty,
\tag{67}
\]

and

\[
\lVert z-z'\rVert_\infty
\leq C_*
\lVert\Phi_D(z)-\Phi_D(z')\rVert_\infty.
\tag{68}
\]

These constants remain witness-specific. No quantitative
parameter-uniform stability statement follows from the almost-everywhere
injectivity result (59).

## Exact theorem summary

### General theorem

For every finite \(d,D,L\), the known global smooth vector fields satisfying
the observation-domain homogeneous injectivity hypothesis (25) have:

1. the arbitrary-\(L\) ordered-composition identity (22);
2. the triangular jet identity (24);
3. recursive recovery of all \(N_{d,L}\) intrinsic Hall coordinates from
   finitely many state-coordinate ideal scale jets by (28);
4. formal and BV-signature necessity of every homogeneous injectivity
   condition by Theorem 7.1 and (32);
5. joint state-scale-coordinate independence by Theorem 8.1;
6. an existential \(N_{d,L}\)-output degree-\(L\) truncated observation map
   with nonsingular zero Jacobian and a local analytic inverse by
   (36)--(41); and
7. the scalar codomain lower bound (42) for continuous local injectivity on
   an open \(N_{d,L}\)-dimensional Hall-coordinate domain.

### State-dimension lifting corollary

For \(d=2,\ L\leq4\), and every \(D\geq1\), the block parameters (46):

1. reproduce the scalar represented Hall fields in coordinate one by
   (50)--(51);
2. reproduce the exact cross-degree relation (51a), so the block witness is
   not injective on
   \(\bigoplus_{m=1}^4\mathcal L_{2,m}\);
3. retain the exact nonzero homogeneous minors (53)--(57);
4. imply separate homogeneous injectivity through degree four for
   Lebesgue-almost every full \(D\)-state shallow-tanh parameter by (59);
5. lift the frozen eight-output degree-four truncated design by (62)--(63);
   and
6. retain the exact scalar determinant and witness-specific stability
   constants (64)--(68).

This is an embedding/lifting theorem. It does not show that increasing the
state dimension improves expressivity, and it makes no minimal-width claim.
All coordinates after the first are inactive at the block witness.
The lifting proves no ungraded direct-sum injectivity; at the block witness it
fails by (51a).

## Standard results requiring citation

The following inputs are standard and should be cited in a paper version.

1. **Free Lie algebra, Lyndon--Hall bases, primitive/group-like series, and
   tensor exponential/logarithm.** Christophe Reutenauer,
   [*Free Lie Algebras*](https://academic.oup.com/book/53911), Oxford
   University Press, 1993, especially Chapters 3 and 5.
2. **Bounded-variation signatures, Chen concatenation, controlled-ODE
   expansions, and exact finite-step realizability.** Peter K. Friz and
   Nicolas B. Victoir,
   [*Multidimensional Stochastic Processes as Rough Paths*](https://doi.org/10.1017/CBO9780511845079),
   Cambridge University Press, 2010, especially Chapters 1--3, Section 7.1,
   and Theorems 7.28 and 7.30 for (32).
3. **Banach-space implicit-function theorem.** Klaus Deimling,
   [*Nonlinear Functional Analysis*](https://doi.org/10.1007/978-3-662-00547-7),
   Springer, 1985, Section 15.1.
4. **Real-analytic inverse-function theorem and zero sets of nontrivial
   real-analytic functions.** Steven G. Krantz and Harold R. Parks,
   *A Primer of Real Analytic Functions*, second edition, Birkhäuser, 2002,
   for the standard results used at (41) and (59).
5. **Brouwer invariance of domain.** Allen Hatcher,
   [*Algebraic Topology*](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf),
   Section 2.B, Theorem 2B.3.

The repository-specific word/operator order, bracket sign, ordered
composition formula, finite-evaluation lemma, recursive decoder, kernel
argument, polynomial-in-scale independence proof, and block-embedding proof
were established directly above.

## What remains confined to the explicit witness

The exact point sets, rational matrices, determinant
\(-258733035/524288\), radius \(1/2048\), and constants in (65)--(66) come
from the frozen scalar \(d=2,\ L=4\) certificate. They extend only to its
exact inactive-coordinate block embeddings through the polynomial identity
(63). They are not consequences of the general conditional theorem and are
not asserted for arbitrary shallow-tanh parameters.

The underlying frozen scalar statements remain those in the
[technical note](technical_note.md),
[graded reconstruction derivation](graded_reconstruction_derivation.md),
[graded reconstruction summary](graded_reconstruction_summary.md),
[finite-observation theorem](finite_observation_theorem.md), and
[local-stability theorem](local_stability_theorem.md); this note neither
rewrites nor weakens them.

No unconditional homogeneous-injectivity theorem for arbitrary
\((d,D,L)\) is proved here. No generic explicit observation design and no
quantitative stability theorem are proved for arbitrary \((d,D,L)\).

## Final status

| Claim | Status |
|---|---|
| Arbitrary-\(L\) ordered-composition and triangular scale-jet identities under the stated \(C^L\)/BV hypotheses | **PROVED** |
| Arbitrary-\((d,D,L)\) reconstruction, finite truncated observation, and local inverse under homogeneous injectivity (25) | **CONDITIONALLY PROVED** |
| State-dimension lifting of the exact \(d=2,\ L\leq4\) scalar tanh witness to every \(D\geq1\) | **PROVED** |
| Unconditional homogeneous free-Lie injectivity for arbitrary shallow-tanh \((d,D,L)\) | **NOT PROVED** |
