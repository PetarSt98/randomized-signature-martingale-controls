# From homogeneous Lie injectivity to graded log-signature reconstruction

This note proves the degree-by-degree reconstruction bridge for two inputs and
degrees at most four.  The path-level argument is first made for continuous
bounded-variation paths (hence for piecewise-\(C^1\) paths).  No rough-path
extension is used in any proof below.

This note gives the detailed proof of the graded-reconstruction component
summarized in `research_output/technical_note.md`. It preserves the certified
homogeneous-injectivity theorem and develops the object-A reconstruction
result in detail.

The separate [fixed-state Hall-rank theorem](intrinsic_hall_rank_theorem.md)
now gives generic rank \(\min(D,\nu_{d,L})\) at initialization zero in the
full biased shallow-tanh parameter space, where \(\nu_{d,L}\) is the total
Hall dimension through degree \(L\). For \(D\geq\nu_{d,L}\), it gives
local recovery near zero Hall coordinates from one degree-\(L\) truncated
terminal vector at any fixed nonzero scale, with the sharp dimension
threshold for that observation model. The scalar jet reconstruction proved
here remains a different observation design; its exact constants are not
constants for the new vector map.

The three observation objects used in the note are different:

1. **A. Ideal normalized scale jets** \(J_m(y;X)\), which are exact derivatives
   at input scale \(r=0\).
2. **B. The degree-four truncated finite-scale response**
   \(F_y^{(4)}(r;z)\), a polynomial obtained by retaining tensor degrees at most
   four.
3. **C. The actual full CDE response** \(Y_T^y(rX)\), whose Chen--Fliess
   expansion also has levels at least five.

Only A is used in the recursive reconstruction theorem.  B is used in the
joint state--scale theorem.  Neither theorem identifies B with C at nonzero
scale.

## 1. Setting and conventions

### 1.1 Scaled CDE and regularity

Fix \(T>0\), an open interval \(U\subseteq\mathbb R\), and a continuous path
\(X=(X^1,X^2):[0,T]\to\mathbb R^2\) of bounded variation.  Let
\(V_1,V_2\in C^4(U;\mathbb R)\), and put

\[
 \omega_X:=\|X^1\|_{1\text{-var};[0,T]}
            +\|X^2\|_{1\text{-var};[0,T]}.
\]

For each \(y\in U\), choose \(d_y>0\) so that
\(K_y=[y-d_y,y+d_y]\subset U\), and set

\[
 M_y:=\max_{i=1,2}\sup_{x\in K_y}|V_i(x)|,
 \qquad
 \varepsilon_y:=
 \begin{cases}
 d_y/(M_y\omega_X),&M_y\omega_X>0,\\
 +\infty,&M_y\omega_X=0.
 \end{cases}
\]

For \(|r|<\varepsilon_y\), the Riemann--Stieltjes CDE

\[
 dY_t^{y,r}
   =\sum_{i=1}^2 V_i(Y_t^{y,r})\,d(rX_t^i)
   =r\sum_{i=1}^2 V_i(Y_t^{y,r})\,dX_t^i,
 \qquad Y_0^{y,r}=y,
\]

has a unique solution that remains in a compact subset of \(U\) on which the
derivatives of the \(V_i\) through order four are bounded.  Indeed, up to a
hypothetical first exit from \(K_y\),

\[
 \sup_{t\le T}|Y_t^{y,r}-y|
 \le |r|M_y\omega_X<d_y,
\]

which rules out that exit; standard local existence and uniqueness for BV
controlled ODEs then gives the solution on \([0,T]\).  Thus the local
hypotheses hold for every \(y\in U\), rather than only for states selected
later.  For any finite selected state set one may use the minimum of its
positive \(\varepsilon_y\)'s.  The certified tanh fields are globally defined
and have bounded derivatives of every order.

The notation is, by definition,

\[
Y^{y,r}(X):=Y^y(rX).
\]

No state-rescaling identity is asserted or used.  Define

\[
J_m(y;X)
  :=\frac1{m!}\left.\frac{\partial^m}{\partial r^m}
      Y_T^{y,r}(X)\right|_{r=0},\qquad 1\le m\le4.
\]

The existence of these classical derivatives, not merely asymptotic
coefficients, is proved at the start of Section 2.

### 1.2 Associative words, tensors, and signature

Use the ordered alphabet \(1<2\), corresponding to the repository's internal
zero-based letters \(0<1\) and displayed generators \(X_1,X_2\).  A word is
\(w=i_1\cdots i_m\), its length is \(|w|=m\), and the empty word is denoted by
\(\varnothing\).

Let \(E=\operatorname{span}\{X_1,X_2\}\).  In the tensor algebra

\[
T((E))=\prod_{m\ge0}E^{\otimes m},
\]

the monomial associated with \(w=i_1\cdots i_m\) is
\(X_w=X_{i_1}\cdots X_{i_m}\).  Tensor multiplication is ordered
concatenation:

\[
X_u\otimes X_v=X_{uv}.
\]

Products below are this noncommutative product; the tensor symbol is often
suppressed.  Truncation after degree \(L\) uses the same multiplication followed
by deletion of terms of degree greater than \(L\).

The chronological signature-word convention is

\[
S^{i_1\cdots i_m}(X)
 :=\int_{0<t_1<\cdots<t_m<T}
      dX_{t_1}^{i_1}\cdots dX_{t_m}^{i_m},
\]

so the earliest differential is the leftmost letter.  There is no extra
factorial in this definition.  For example, a straight segment with increment
\(a\in E\) has \(S_m=a^{\otimes m}/m!\).  Set

\[
S(X)=1+\sum_{m\ge1}S_m(X),\qquad
S_m(X)=\sum_{|w|=m}S^w(X)X_w.
\]

With this convention, Chen concatenation places the earlier path on the left.
Input dilation acts homogeneously:

\[
S_m(rX)=r^mS_m(X).
\]

### 1.3 Tensor logarithm and deterministic Lyndon basis

The tensor logarithm convention is

\[
\log_{\otimes}S
 =\sum_{k\ge1}\frac{(-1)^{k+1}}{k}(S-1)^{\otimes k},
\qquad
S=\exp_{\otimes}(\ell)
 =\sum_{k\ge0}\frac{\ell^{\otimes k}}{k!}.
\]

For a bounded-variation signature, Chen's group-like theorem implies that
\(\ell=\log_{\otimes}S\) is a Lie series.  Write

\[
\ell=\ell_1+\ell_2+\ell_3+\ell_4+\text{terms of degree at least five},
\qquad \ell_m\in\mathcal L_{2,m}.
\]

The repository ordering is deterministic.  Words are lexicographically
ordered.  A Lyndon word is strictly smaller than each nontrivial proper suffix.
For a nontrivial Lyndon word \(w=uv\), \(v\) is its longest proper Lyndon
suffix, and the standard bracketing is \(H(w)=[H(u),H(v)]\).  Brackets expand
associatively by

\[
[u,v]=uv-vu.
\]

In degree order, followed by lexicographic Lyndon-word order within a degree,
the basis through degree four is

\[
\begin{array}{c|l}
1&X_1,\ X_2,\\
2&[X_1,X_2],\\
3&[X_1,[X_1,X_2]],\ [[X_1,X_2],X_2],\\
4&[X_1,[X_1,[X_1,X_2]]],\
  [X_1,[[X_1,X_2],X_2]],\
  [[[X_1,X_2],X_2],X_2].
\end{array}
\]

The standard-bracketed Lyndon basis is a Hall basis (often called the
Lyndon--Hall basis).  Its coefficients are the Hall coordinates meant below;
no second basis convention is introduced.

### 1.4 Differential operators, composition order, and bracket sign

For a scalar vector field \(V\), let its pullback derivation be

\[
\mathcal D_V f(x):=Df(x)V(x).
\]

Operator multiplication means composition in the usual order,
\((AB)f=A(Bf)\).  Define the unital associative representation

\[
\Gamma(X_i)=\mathcal D_{V_i},\qquad
\Gamma(X_{i_1}\cdots X_{i_m})
 =\mathcal D_{V_{i_1}}\circ\cdots\circ\mathcal D_{V_{i_m}}.
\]

Consequently,

\[
\Gamma(uv)=\Gamma(u)\Gamma(v),
\]

so \(\Gamma\) is a homomorphism, not an anti-homomorphism, for the repository
word order.

The sign can be checked without convention-dependent shorthand.  Hessian terms
cancel to give

\[
\begin{aligned}
[\mathcal D_V,\mathcal D_W]_{\mathrm{op}}f
 &=\mathcal D_V(\mathcal D_Wf)-\mathcal D_W(\mathcal D_Vf)\\
 &=Df\bigl(DW\,V-DV\,W\bigr).
\end{aligned}
\]

Thus the induced vector-field bracket is exactly

\[
[V,W]_{\mathrm{repo}}=DW\,V-DV\,W,
\]

and

\[
\Gamma(X_1X_2-X_2X_1)\operatorname{Id}
 =DV_2\,V_1-DV_1\,V_2
 =[V_1,V_2]_{\mathrm{repo}}.
\]

For the exact witness \(V_1(x)=\tanh x\), \(V_2(x)=1/2\), this is

\[
\Gamma(X_1X_2-X_2X_1)\operatorname{Id}(x)
 =-\tfrac12\operatorname{sech}^2x,
\]

which equals \(-1/2\) at \(x=0\), exactly the certified degree-two entry.
Reversing the word-to-operator order would give the opposite sign and would
conflict with the verified certificate.  Therefore no reversal or sign
correction is required.

Define the represented Lie field

\[
\rho(h):=\Gamma(h)\operatorname{Id},\qquad
\rho_m:=\rho|_{\mathcal L_{2,m}}.
\]

## 2. Chen--Fliess coefficient and the normalization factorial

First establish the scale regularity required by the definition of \(J_m\).
Fix \(y\in U\), use \(K_y\) from Section 1.1, and let
\(\mathcal B=C([0,T];\mathbb R)\) with the uniform norm.  On the open subset
\(\mathcal O_y\subset\mathcal B\) of paths with range in the interior of
\(K_y\), define

\[
 \Phi(r,Z)_t
 :=Z_t-y-r\sum_{i=1}^2\int_0^t V_i(Z_s)\,dX_s^i.
\]

Riemann--Stieltjes integration against a fixed BV path is a bounded linear map
on continuous integrands, and the Nemytskii maps \(Z\mapsto V_i\circ Z\) are
\(C^4\) there.  Hence
\(\Phi:\mathbb R\times\mathcal O_y\to\mathcal B\) is
\(C^4\) near \((0,Z^0)\), where \(Z^0_t=y\), and

\[
 D_Z\Phi(0,Z^0)=I_{\mathcal B}.
\]

The Banach-space implicit-function theorem gives a unique \(C^4\) map
\(r\mapsto Z(r)\) satisfying \(\Phi(r,Z(r))=0\) for sufficiently small
\(|r|\).  By uniqueness this is the CDE solution from Section 1.1.  Therefore
\(r\mapsto Y_T^{y,r}\) is classically \(C^4\), and all four derivatives in
the definition of \(J_m\) exist.

For any sufficiently smooth scalar test function \(f\), the
Riemann--Stieltjes chain rule gives

\[
f(Y_t^{y,r})
 =f(y)+r\sum_{i=1}^2\int_0^t
   (\mathcal D_{V_i}f)(Y_s^{y,r})\,dX_s^i.
\]

Substituting the same identity into the integrand fixes both orders.  At degree
two, for example, the earlier differential is \(dX_u^j\), the later one is
\(dX_s^i\), and the associated operator is
\(\mathcal D_{V_j}\mathcal D_{V_i}\):

\[
r^2\sum_{j,i=1}^2
 \int_{0<u<s<T}dX_u^j\,dX_s^i\,
 (\mathcal D_{V_j}\mathcal D_{V_i}f)(y).
\]

Induction by repeated substitution therefore yields, through degree four,

\[
f(Y_T^{y,r})
 =f(y)+\sum_{m=1}^4 r^m
   \sum_{i_1,\ldots,i_m=1}^2
   S^{i_1\cdots i_m}(X)
   (\mathcal D_{V_{i_1}}\cdots\mathcal D_{V_{i_m}}f)(y)
   +R_5(r).
\]

The remainder is not only formal.  Repeating the same substitution once more
gives the exact integral

\[
 R_5(r)=r^5\sum_{i_1,\ldots,i_5=1}^2
 \int_{0<t_1<\cdots<t_5<T}
 (\mathcal D_{V_{i_1}}\cdots\mathcal D_{V_{i_5}}f)
       (Y_{t_1}^{y,r})
 \,dX_{t_1}^{i_1}\cdots dX_{t_5}^{i_5}.
\]

For \(|r|<\varepsilon_y\), set

\[
 C_{5,f,y}:=
 \max_{i_1,\ldots,i_5\in\{1,2\}}
 \sup_{x\in K_y}
 \left|\mathcal D_{V_{i_1}}\cdots
             \mathcal D_{V_{i_5}}f(x)\right|.
\]

The total-variation measure
\(|dX^1|+|dX^2|\) is nonatomic because \(X\) is continuous.  Summing the
absolute fivefold iterated integrals and partitioning the product domain among
the \(5!\) strict time orderings gives

\[
 \boxed{
 |R_5(r)|\le
 \frac{C_{5,f,y}}{5!}\,|r|^5\omega_X^5.}
\]

For \(f=\operatorname{Id}\), the degree-four coefficient uses derivatives of
the \(V_i\) only through order three, while \(C_{5,f,y}\) uses derivatives
through order four.  Thus \(C^4\) is used precisely for the displayed
remainder; \(C^3\) would suffice for the four coefficients alone.  This local
bound identifies derivatives at \(r=0\); it is not a uniform finite-scale
error theorem for object C over an unrestricted path class.

Taking \(f=\operatorname{Id}\) gives

\[
Y_T^{y,r}
 =y+\sum_{m=1}^4 r^m\Gamma(S_m(X))\operatorname{Id}(y)+O(r^5).
\]

In particular, degree by degree,

\[
\begin{aligned}
[r]Y_T^{y,r}
 &=\sum_{i_1}S^{i_1}\mathcal D_{V_{i_1}}\operatorname{Id}(y),\\
[r^2]Y_T^{y,r}
 &=\sum_{i_1,i_2}S^{i_1i_2}
   \mathcal D_{V_{i_1}}\mathcal D_{V_{i_2}}\operatorname{Id}(y),\\
[r^3]Y_T^{y,r}
 &=\sum_{i_1,i_2,i_3}S^{i_1i_2i_3}
   \mathcal D_{V_{i_1}}\mathcal D_{V_{i_2}}
   \mathcal D_{V_{i_3}}\operatorname{Id}(y),\\
[r^4]Y_T^{y,r}
 &=\sum_{i_1,i_2,i_3,i_4}S^{i_1i_2i_3i_4}
   \mathcal D_{V_{i_1}}\mathcal D_{V_{i_2}}
   \mathcal D_{V_{i_3}}\mathcal D_{V_{i_4}}
   \operatorname{Id}(y).
\end{aligned}
\]

The terminal map is \(C^4\) in \(r\), as proved above.  Uniqueness of its
degree-four asymptotic polynomial therefore identifies
\(a_m=[r^m]Y_T^{y,r}\) with
\(\partial_r^mY_T^{y,r}|_{r=0}/m!\).  The normalization in \(J_m\) removes
exactly this Taylor factorial.  Hence, with no remaining factorial,

\[
\boxed{J_m(y;X)=\Gamma(S_m(X))\operatorname{Id}(y),\qquad 1\le m\le4.}
\]

The simplex factorial already present for straight-line signatures is part of
\(S_m\); it is unrelated to, and not duplicated by, the Taylor normalization.

## 3. Homogeneous tensor-exponential formulas

For this section write simply

\[
\ell=\ell_1+\ell_2+\ell_3+\ell_4
\]

inside the degree-four truncated tensor algebra.  From the tensor exponential,

\[
S_m
 =\sum_{k=1}^m\frac1{k!}
   \sum_{\substack{j_1+\cdots+j_k=m\\j_a\ge1}}
   \ell_{j_1}\ell_{j_2}\cdots\ell_{j_k}.
\]

The inner sum is over ordered compositions of \(m\), so none of its products
may be commuted.  Enumerating those compositions independently in each degree
gives

\[
\boxed{S_1=\ell_1,}
\]

\[
\boxed{S_2=\ell_2+\frac12\ell_1\ell_1,}
\]

\[
\boxed{
S_3=\ell_3
 +\frac12\bigl(\ell_1\ell_2+\ell_2\ell_1\bigr)
 +\frac16\ell_1\ell_1\ell_1,}
\]

and

\[
\boxed{\begin{aligned}
S_4={}&\ell_4
 +\frac12\bigl(\ell_1\ell_3+\ell_2\ell_2+\ell_3\ell_1\bigr)\\
&+\frac16\bigl(
  \ell_1\ell_1\ell_2
 +\ell_1\ell_2\ell_1
 +\ell_2\ell_1\ell_1\bigr)
 +\frac1{24}\ell_1\ell_1\ell_1\ell_1.
\end{aligned}}
\]

No commutative multinomial coefficients have been inserted: each displayed
ordered product arises once from an ordered composition.

## 4. Triangular graded identities

Combining the scale-jet identity with the four tensor-exponential formulas
gives

\[
J_m(y;X)
 =\rho(\ell_m(X))(y)
  +\Psi_m\bigl(y;\ell_1(X),\ldots,\ell_{m-1}(X)\bigr).
\]

Every lower-degree correction is explicit:

\[
\boxed{\Psi_1(y)=0,}
\]

\[
\boxed{
\Psi_2(y;\ell_1)
 =\frac12\Gamma(\ell_1\ell_1)\operatorname{Id}(y),}
\]

\[
\boxed{\begin{aligned}
\Psi_3(y;\ell_1,\ell_2)
={}&\frac12\Gamma(\ell_1\ell_2+\ell_2\ell_1)
       \operatorname{Id}(y)\\
&+\frac16\Gamma(\ell_1\ell_1\ell_1)
       \operatorname{Id}(y),
\end{aligned}}
\]

and

\[
\boxed{\begin{aligned}
\Psi_4(y;\ell_1,\ell_2,\ell_3)
={}&\frac12\Gamma(
   \ell_1\ell_3+\ell_2\ell_2+\ell_3\ell_1)
   \operatorname{Id}(y)\\
&+\frac16\Gamma(
   \ell_1\ell_1\ell_2
  +\ell_1\ell_2\ell_1
  +\ell_2\ell_1\ell_1)
   \operatorname{Id}(y)\\
&+\frac1{24}\Gamma(\ell_1\ell_1\ell_1\ell_1)
   \operatorname{Id}(y).
\end{aligned}}
\]

Thus the unknown degree-\(m\) Lie component enters \(J_m\) linearly through
\(\rho_m\), while every nonlinear term depends only on already lower degrees.
This is a graded triangular statement; it is not injectivity of \(\rho\) on an
ungraded direct sum.

## 5. Precise definition of the truncated finite-scale object

Let

\[
\mathfrak g^{(4)}
 =\mathcal L_{2,1}\oplus\mathcal L_{2,2}
  \oplus\mathcal L_{2,3}\oplus\mathcal L_{2,4},
\qquad z=z_1+z_2+z_3+z_4,
\]

and let the homogeneous dilation be

\[
\delta_rz=r z_1+r^2z_2+r^3z_3+r^4z_4.
\]

Define object B by

\[
\begin{aligned}
F_y^{(4)}(r;z)
 &:=\Gamma\!\left(
   \pi_{\le4}\exp_{\otimes}(\delta_rz)\right)
   \operatorname{Id}(y)\\
 &=y+\sum_{m=1}^4r^m
   \Gamma\!\left(\pi_m\exp_{\otimes}(z)\right)
   \operatorname{Id}(y).
\end{aligned}
\]

If \(z_m=\ell_m(X)\), then the coefficient of \(r^m\) in this polynomial is
the ideal jet \(J_m(y;X)\).  At a nonzero finite scale, however,
\(F_y^{(4)}(r;z)\) is only object B.  The actual response C is
\(Y_T^y(rX)\), and its expansion contains degrees at least five. This
definition does not identify B with C or provide a full-CDE inverse. The
subsequent scalar actual-response theorem summarized in Section 15 supplies
a uniform remainder bound under its additional total-variation hypothesis.

## 6. Finite-evaluation lemma

**Lemma.** Let \(\Omega\) be a nonempty set and let
\(f_1,\ldots,f_N:\Omega\to\mathbb R^d\) be linearly independent as functions.
Then there are \(N\) point-coordinate pairs

\[
(x_a,k_a)\in\Omega\times\{1,\ldots,d\},\qquad 1\le a\le N,
\]

such that the scalar evaluation matrix

\[
A_{aj}=(f_j(x_a))_{k_a}
\]

is nonsingular.  In particular, finitely many scalar point-coordinate
evaluations have full column rank.

**Proof.** Let \(E=\operatorname{span}\{f_1,\ldots,f_N\}\), so
\(\dim E=N\).  For each \((x,k)\), restrict the scalar functional

\[
\varepsilon_{x,k}:f\mapsto(f(x))_k
\]

to \(E\).  The common kernel of all these restricted evaluations is zero:
a function whose every coordinate vanishes at every point is the zero
function.  If their linear span were a proper subspace of \(E^*\), its
annihilator in the finite-dimensional space \(E\) would contain a nonzero
element, contradicting that common-kernel statement.  Hence the evaluations
span \(E^*\).  Select \(N\) of them forming a basis of \(E^*\).  Their matrix in
the basis \(f_1,\ldots,f_N\) is nonsingular.  No continuity assumption is
needed. \(\square\)

The converse implication used in the existing certificate is immediate as
well: a full-column-rank finite evaluation matrix rules out a nonzero
functional linear relation.

## 7. Reconstruction sufficiency

**Theorem (finite ideal-jet reconstruction).** Fix \(1\le L\le4\).  Assume:

1. \(X\) and \(V_1,V_2\) satisfy the bounded-variation and local \(C^4\)
   hypotheses of Section 1.1 at every observation state;
2. the vector fields, their derivatives, and the representation \(\Gamma\) are
   known exactly;
3. for each \(m\le L\), the homogeneous map
   \(\rho_m:\mathcal L_{2,m}\to\mathfrak X(U)\) is injective;
4. the observer has exact access to the normalized derivatives \(J_m(y;X)\)
   at the finitely many initial states selected below.

Then all Hall coefficients of
\(\ell_1(X),\ldots,\ell_L(X)\) are reconstructed uniquely and recursively from
finitely many ideal normalized scale jets.

**Proof.** Let
\(H_{m,1},\ldots,H_{m,n_m}\) be the repository's standard-Lyndon basis of
\(\mathcal L_{2,m}\), and write

\[
\ell_m=\sum_{j=1}^{n_m}\lambda_{m,j}H_{m,j}.
\]

Injectivity of \(\rho_m\) says that the \(n_m\) scalar functions
\(\rho(H_{m,j})\) are linearly independent.  The finite-evaluation lemma
therefore supplies \(n_m\) states \(y_{m,a}\in U\) for which

\[
(M_m)_{aj}:=\rho(H_{m,j})(y_{m,a})
\]

is nonsingular.  These states and matrices depend only on the known
representation, not on the unknown path.

At degree one, \(\Psi_1=0\), so
\(M_1\lambda_1=\mathbf J_1\).  Suppose degrees below \(m\) have been recovered.
They determine the exact vector

\[
(\boldsymbol\Psi_m)_a
 :=\Psi_m(y_{m,a};\ell_1,\ldots,\ell_{m-1}).
\]

Evaluating the triangular identity gives

\[
\mathbf J_m-\boldsymbol\Psi_m=M_m\lambda_m,
\]

and hence

\[
\boxed{
\lambda_m=M_m^{-1}
 (\mathbf J_m-\boldsymbol\Psi_m),\qquad 1\le m\le L.}
\]

Induction reconstructs every homogeneous component through \(L\). \(\square\)

For observation object A, this triangular decoder is global on the whole
formal coordinate space \(\mathfrak g^{(L)}\): the matrices \(M_m\) are fixed
and invertible, and the recursion has no local-neighborhood restriction.  This
does not assert a global inverse for object B or object C.

The theorem concerns exact ideal derivatives.  Access to them is an idealized
identifiability assumption, not a measurement procedure derived here from a
finite family of nonzero-scale full-CDE runs.  In particular, the theorem
assumes neither recovery from one terminal state nor a finite-difference
approximation to the jets.

## 8. Converse: formal invisibility and actual-path realizability

### 8.1 Formal free-Lie/free-nilpotent coordinate model

For every Lie polynomial \(a\), \(\Gamma(a)\) is itself a first-order
derivation and

\[
\Gamma(a)=\mathcal D_{\rho(a)}.
\]

This follows by induction from the generator definition and the commutator
calculation in Section 1.4.  In particular,

\[
h\in\ker\rho_m\quad\Longrightarrow\quad \Gamma(h)=0
\]

as an operator on all smooth test functions, not merely after application to
\(\operatorname{Id}\).

Let \(1\le m\le L\le4\), let \(0\ne h\in\ker\rho_m\), and let
\(z\in\mathfrak g^{(L)}\).  Introduce an independent formal grading variable
\(u\) and work only through \(u\)-degree \(L\).  Expand the finite tensor
polynomial
\(\pi_{\le L}\exp_{\otimes}(\delta_u(z+t h))\).  Every term in its difference
from \(\pi_{\le L}\exp_{\otimes}(\delta_u z)\) contains at least one factor
\(h\).  Since \(\Gamma\) is an algebra homomorphism and \(\Gamma(h)=0\), every
such term maps to zero.  Hence, as a polynomial in \(u\),

\[
\Gamma\!\left(
 \pi_{\le L}\exp_{\otimes}(\delta_u(z+t h))\right)
=\Gamma\!\left(
 \pi_{\le L}\exp_{\otimes}(\delta_u z)\right).
\]

Equality of coefficients of every power of \(u\) shows

\[
\Gamma\!\left(\pi_k\exp_{\otimes}(z+t h)\right)
 =\Gamma\!\left(\pi_k\exp_{\otimes}z\right)
\qquad\text{for every }k\le L.
\]

Thus \(z\) and \(z+t h\) give identical formal graded jets at every state and
at every observed level through \(L\); they also give the same truncated
finite-scale response.  The degree-\(m\) Hall direction \(h\) is genuinely
invisible in the formal model.  Consequently homogeneous injectivity is
necessary for local injectivity on any open subset of the formal Hall-coordinate
space.

### 8.2 What is required for actual bounded-variation paths

To transfer that necessity statement to actual path signatures, it is enough
that the set of realized truncated log signatures contain a nonempty open set
in \(\mathfrak g^{(L)}\).  The standard finite-step realization result is
stronger:

\[
\{S^{\le L}(X):X\text{ piecewise linear}\}
 =G^L(\mathbb R^2)
 =\exp_{\otimes}(\mathfrak g^{(L)}).
\]

Here is the reduction to finite-dimensional controllability.  A straight
segment of increment \(v\in E\) has truncated signature
\(\exp_{\otimes}(v)\), and Chen concatenation makes a piecewise-linear
signature a finite product of such exponentials.  In the connected,
simply-connected free step-\(L\) nilpotent group, the Lie algebra generated by
the horizontal first layer \(E\) is the entire free nilpotent Lie algebra.
The orbit/subgroup theorem (equivalently Rashevsky--Chow accessibility) says
that finite products of the horizontal one-parameter subgroups therefore fill
the whole group.  Since the nilpotent exponential is a global diffeomorphism,
every Hall-coordinate vector is realized by a piecewise-linear, hence
bounded-variation, path.

It follows that the formal kernel argument also proves actual-path necessity
for observation object A through level \(L\), and for object B.  It does **not**
prove that two realizing paths have identical object C: their signature levels
above \(L\) may differ and may affect the actual full CDE response.

Here “actual-path necessity” is a statement about the observation map on the
realized truncated Hall-coordinate domain, which is all of
\(\mathfrak g^{(L)}\).  Global realization alone does not provide a continuous
local right inverse into path space near an arbitrarily prescribed path, so no
stronger claim in the \(1\)-variation topology of path representatives is made.

## 9. Application to the exact tanh certificate

Let

\[
q=\frac{\log3}{2},\qquad V_1(x)=\tanh x,\qquad V_2(x)=\frac12.
\]

The authoritative exact matrices are generated by
`lie_reservoir.certificates.degree_exact_certificate`; their displayed forms
are also recorded in `research_output/certificate.json` and
`research_output/certificate_note.md`.  Rows follow the certified point order
and columns follow the standard-Lyndon order fixed in Section 1.3:

\[
M_1=
\begin{pmatrix}0&1/2\\1/2&1/2\end{pmatrix}
\quad\text{at }(0,q),
\qquad
M_2=\begin{pmatrix}-1/2\end{pmatrix}
\quad\text{at }(0),
\]

\[
M_3=
\begin{pmatrix}15/32&3/16\\15/32&-3/16\end{pmatrix}
\quad\text{at }(-q,q),
\]

and

\[
M_4=
\begin{pmatrix}
-57/128&-3/32&3/64\\
-1/2&0&1/4\\
-57/128&3/32&3/64
\end{pmatrix}
\quad\text{at }(-q,0,q).
\]

Their exact determinants are respectively

\[
-\frac14,\qquad-\frac12,\qquad-\frac{45}{256},\qquad
\frac{135}{8192},
\]

so these are complete-basis evaluation matrices, not numerical rank evidence.
Their product is the authoritative exact value
\(-6075/16777216\ne0\).
Exact Gauss--Jordan elimination applied to the matrices returned by the
authoritative `Fraction` certificate constructor gives

\[
M_1^{-1}=
\begin{pmatrix}-2&2\\2&0\end{pmatrix},
\qquad
M_2^{-1}=\begin{pmatrix}-2\end{pmatrix},
\]

\[
M_3^{-1}=
\begin{pmatrix}
16/15&16/15\\
8/3&-8/3
\end{pmatrix},
\]

and

\[
M_4^{-1}=
\begin{pmatrix}
-64/45&8/15&-64/45\\
-16/3&0&16/3\\
-128/45&76/15&-128/45
\end{pmatrix}.
\]

All four products \(M_mM_m^{-1}\) are the exact identity over
\(\mathbb Q\); no floating-point determinant or inverse is used.

For \(P_m\) equal to the certified ordered point set, define the residual

\[
R_m(y):=J_m(y;X)
 -\Psi_m(y;\ell_1,\ldots,\ell_{m-1}),
\qquad y\in P_m,
\]

with \(\Psi_1=0\), and form the residual column in the listed point order.  If
\(\lambda_m\) is the Hall-coordinate column in the listed basis order, then

\[
\boxed{
\lambda_m
 =M_m^{-1}\!\left(
   \mathbf J_m-\boldsymbol\Psi_m(
   \lambda_1,\ldots,\lambda_{m-1})\right)
 =M_m^{-1}\mathbf R_m.}
\]

Here \(\boldsymbol\Psi_m(\lambda_1,\ldots,\lambda_{m-1})\) means that
\(\ell_k=\sum_j\lambda_{k,j}H_{k,j}\) has been substituted into the explicit
formula from Section 4 and then evaluated on \(P_m\).

Explicitly,

\[
\lambda_1=
\begin{pmatrix}
-2R_1(0)+2R_1(q)\\
2R_1(0)
\end{pmatrix},
\qquad
\lambda_2=-2R_2(0),
\]

\[
\lambda_3=
\begin{pmatrix}
\frac{16}{15}\bigl(R_3(-q)+R_3(q)\bigr)\\
\frac83\bigl(R_3(-q)-R_3(q)\bigr)
\end{pmatrix},
\]

and

\[
\lambda_4=
\begin{pmatrix}
-\frac{64}{45}R_4(-q)+\frac8{15}R_4(0)-\frac{64}{45}R_4(q)\\
-\frac{16}{3}R_4(-q)+\frac{16}{3}R_4(q)\\
-\frac{128}{45}R_4(-q)+\frac{76}{15}R_4(0)-\frac{128}{45}R_4(q)
\end{pmatrix}.
\]

This reconstructs \(2+1+2+3=8\) Hall coordinates from eight exact,
degree-specific ideal scalar jets at the witness.  The existing analytic
determinant-product argument further shows that the same fixed point sets give
invertible, parameter-dependent \(M_m\) for Lebesgue-almost every
\((a_1,b_1,a_2,b_2)\in\mathbb R^4\).  The rational inverses displayed above
apply only to the exact witness.

## 10. The correct dimension statement

The Hall-coordinate dimensions are

\[
\dim\mathcal L_{2,1}=2,\quad
\dim\mathcal L_{2,2}=1,\quad
\dim\mathcal L_{2,3}=2,\quad
\dim\mathcal L_{2,4}=3,
\]

and hence

\[
\dim\mathfrak g^{(4)}=2+1+2+3=8.
\]

**Dimension proposition.** Let \(U_0\subset\mathbb R^8\) be nonempty and
open.  If a continuous map \(G:U_0\to\mathbb R^N\) is injective on some open
neighborhood in \(U_0\), then \(N\ge8\).

Indeed, if \(N<8\), compose the injective restriction with the coordinate
embedding \(\mathbb R^N\hookrightarrow\mathbb R^8\).  Brouwer invariance of
domain would make its image open in \(\mathbb R^8\), while that image lies in
an \(N\)-dimensional coordinate plane with empty \(8\)-dimensional interior, a
contradiction.

This is a lower bound on the scalar codomain dimension of a continuous
locally injective observation map defined on an open subset of this
eight-dimensional coordinate space. It is not a count of minimal reservoir
runs or scale values, or universal measurement minimality. For one
\(D\)-dimensional truncated terminal vector, the separate fixed-state
Hall-rank theorem attains this lower bound at \(D=8\).

## 11. Joint state--scale linear independence

Let \(\mathcal H\) be the eight standard-Lyndon basis elements in Section 1.3,
and use their Hall coefficients as coordinates \(z\in\mathbb R^8\).  If
\(h\in\mathcal H\) has degree \(m\), then

\[
\pi_{\le4}\exp_{\otimes}(\delta_r(\varepsilon h))
 =1+\varepsilon r^m h+O(\varepsilon^2).
\]

Therefore the derivative of object B at the zero signature is

\[
\boxed{
\partial_hF_y^{(4)}(r;0)=r^m\rho(h)(y).}
\]

**Theorem (joint state--scale independence).** Suppose every
\(\rho_m\), \(1\le m\le4\), is injective.  On any nonempty open scale interval
\(I\subset\mathbb R\setminus\{0\}\), the eight scalar functions

\[
(r,y)\longmapsto r^{\deg h}\rho(h)(y),
\qquad h\in\mathcal H,
\]

are linearly independent on \(I\times U\).

**Proof.** Suppose a linear combination vanishes and group its terms by
degree.  With

\[
h_m:=\sum_{\substack{h\in\mathcal H\\\deg h=m}}c_hh,
\]

the relation becomes

\[
\sum_{m=1}^4r^m\rho(h_m)(y)=0
\qquad\text{for every }(r,y)\in I\times U.
\]

For each fixed \(y\), the left side is a polynomial in \(r\) that vanishes on
an interval.  Every coefficient is therefore zero:
\(\rho(h_m)(y)=0\) for all \(m,y\).  Homogeneous injectivity gives \(h_m=0\)
for every \(m\), and basis independence then gives every \(c_h=0\).
\(\square\)

At the exact tanh witness there is nevertheless an exact cross-degree
restriction at any *single* fixed scale.  Write, in the basis order of
Section 1.3,

\[
\begin{gathered}
 H_2=[X_1,X_2],\qquad
 H_{31}=[X_1,[X_1,X_2]],\qquad
 H_{32}=[[X_1,X_2],X_2],\\
 H_{41}=[X_1,[X_1,[X_1,X_2]]],\qquad
 H_{42}=[X_1,[[X_1,X_2],X_2]],\qquad
 H_{43}=[[[X_1,X_2],X_2],X_2].
\end{gathered}
\]

With \(u=\tanh y\) and \(d/dy=(1-u^2)d/du\), direct application of the
certified bracket convention gives the three odd fields

\[
 \rho(X_1)=u,\qquad
 \rho(H_{32})=-\frac{u(1-u^2)}2,\qquad
 \rho(H_{42})=u^3(1-u^2),
\]

and the five even fields

\[
\begin{aligned}
 \rho(X_2)&=\frac12,&
 \rho(H_2)&=-\frac{1-u^2}{2},&
 \rho(H_{31})&=\frac{1-u^4}{2},\\
 \rho(H_{41})&=-\frac{(1-u^2)(1+3u^4)}2,&
 \rho(H_{43})&=\frac{(1-u^2)(1-3u^2)}4.
\end{aligned}
\]

The odd fields are independent by their successive highest powers
\(u,u^3,u^5\).  The first four displayed even fields are independent by their
successive highest powers \(1,u^2,u^4,u^6\), while all five even fields obey
the unique relation

\[
 \boxed{\rho(4H_2+3H_{31}+2H_{43})=0.}
\]

Thus the eight represented fields have exact cross-degree rank seven at the
witness.  If all observations use one fixed \(r_0\ne0\), and \(\zeta_h\) are
the coefficients of the boxed kernel element, then the nonzero column
coefficient vector \(c_h=r_0^{-\deg h}\zeta_h\) satisfies

\[
 \sum_{h\in\mathcal H}c_h r_0^{\deg h}\rho(h)(y)=0
 \qquad\text{for every }y.
\]

Consequently every all-at-one-scale eight-row Jacobian is singular at this
witness: any nonsingular witness tuple must use at least two distinct scale
values.  This does not contradict the theorem, whose polynomial argument uses
an open interval of scales.

Apply the finite-evaluation lemma on the joint domain \(I\times U\).  There
exist eight state--scale pairs \((r_a,y_a)\) such that the observation map

\[
\mathcal O(z)
 :=\bigl(F_{y_a}^{(4)}(r_a;z)\bigr)_{a=1}^8
\]

has a nonsingular \(8\times8\) Jacobian at \(z=0\).  The finite-dimensional
inverse-function theorem gives a local inverse near the zero signature.
By the realization result in Section 8.2, every \(z\) in that coordinate
neighborhood is the step-four log signature of a piecewise-linear path, so
this local inverse has genuine truncated-signature meaning for object B.  It
does not provide a continuous local inverse into a chosen path topology, and
it says nothing about inversion of the full response C.

There is also a precise conditional stability estimate.  Put
\(A=D\mathcal O(0)\).  On any convex neighborhood \(B\) of zero small enough
that

\[
\sup_{z\in B}\left\|A^{-1}
  \bigl(D\mathcal O(z)-A\bigr)\right\|\le\frac12,
\]

the fundamental theorem of calculus gives

\[
\begin{aligned}
A^{-1}\bigl(\mathcal O(z)-\mathcal O(z')\bigr)
 &=(z-z')\\
 &\quad+\int_0^1 A^{-1}
   \bigl(D\mathcal O(z'+t(z-z'))-A\bigr)(z-z')\,dt,
\end{aligned}
\]

and therefore

\[
\|z-z'\|\le2\|A^{-1}\|\,
  \|\mathcal O(z)-\mathcal O(z')\|,
\qquad z,z'\in B.
\]

The argument above proves existence of eight scalar evaluations and qualitative
local stability for object B.  The downstream exact result
`research_output/finite_observation_theorem.md` now selects

\[
\mathcal D_*=((-q,1),(-q,2),(-q,3),(0,2),(0,3),
               (q,1),(q,2),(q,3))
\]

and certifies

\[
\det D\Phi_{\mathcal D_*}(0)
 =\det B_*=-\frac{258733035}{524288}\ne0.
\]

That result also records the exact inverse, gives non-certifying binary64
spectral diagnostics, proves a local analytic inverse, and proves directly
from the fixed-design analytic determinant that this same design has an
invertible zero-signature Jacobian for Lebesgue-almost every shallow-tanh
parameter vector.  The downstream quantitative result in
`research_output/local_stability_theorem.md` verifies the displayed derivative
bound in the infinity norm on

\[
\|z\|_\infty\le R_*:=\frac1{2048},
\]

with

\[
\|B_*^{-1}\|_\infty=\frac{23812}{2151},\qquad
c_*=\frac{21220001406063}{409087045009408},\qquad
C_*=\frac{409087045009408}{21220001406063}.
\]

Thus

\[
\|\Phi_{\mathcal D_*}(z)-\Phi_{\mathcal D_*}(z')\|_\infty
\ge c_*\|z-z'\|_\infty
\]

and the inverse on the certified image has Lipschitz constant \(C_*\).
That result also proves the common-scale inverse-linearization amplification
\(\|B_*(h)^{-1}\|_\infty=\Theta(|h|^{-4})\) as \(h\to0\).  The radius
\(R_*\) is a conservative coefficient-majorant certificate, not a claim of
the largest possible stable radius.  All these statements remain confined to
object B and do not invert the full response C.

## 12. Standard results requiring citation

A formal paper version should cite the following standard inputs at the points
where they are used:

1. For bounded-variation signatures and controlled-ODE expansion background:
   Peter K. Friz and Nicolas B. Victoir, *Multidimensional Stochastic Processes
   as Rough Paths*, Cambridge University Press, 2010, Chapters 1--3 and
   Section 7.1,
   [DOI 10.1017/CBO9780511845079](https://doi.org/10.1017/CBO9780511845079).
   The word/operator order, scale-parameter regularity, and the precise local
   remainder used here were nevertheless derived directly in Section 2.  This
   matters because other Fliess conventions reverse the word and operator
   indices.
2. For the Banach-space implicit-function theorem used to establish the
   classical scale derivatives: Klaus Deimling, *Nonlinear Functional
   Analysis*, Springer, 1985, Section 15.1 (pp. 147--152),
   [DOI 10.1007/978-3-662-00547-7](https://doi.org/10.1007/978-3-662-00547-7).
3. For primitive/group-like series, tensor exponential and logarithm, and the
   fact that a bounded-variation log signature is a free Lie series:
   Christophe Reutenauer, *Free Lie Algebras*, Oxford University Press, 1993,
   Theorems 3.1--3.2 and Corollary 3.5; for the standard bracketed Lyndon basis
   as a Hall basis, Chapter 5, Theorem 5.1,
   [publisher record](https://academic.oup.com/book/53911).
4. For exact finite-step path realizability:
   Friz--Victoir, Theorems 7.28 (printed pp. 142--143) and 7.30
   (printed pp. 144--145) in the linked May 2009 author draft. They
   identify
   \(G^L(\mathbb R^d)=\exp(\mathfrak g^L(\mathbb R^d))\) with the set of
   truncated signatures of continuous bounded-variation paths, with
   piecewise-linear realizations.
5. The Lebesgue-almost-every tanh implication in Section 9 is imported
   unchanged from the repository's verified exact certificate.  Its external
   standard input is the theorem that the zero set of a nontrivial
   real-analytic function on a connected open subset of Euclidean space has
   Lebesgue measure zero; a paper version should cite a real-analytic geometry
   reference for that theorem.
6. For the codomain-dimension obstruction: Allen Hatcher,
   *Algebraic Topology*, Section 2.B, Theorem 2B.3 (Brouwer invariance of
   domain), [official text](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf).
7. For the local inverse: Walter Rudin, *Principles of Mathematical Analysis*,
   third edition, Theorem 9.24 (the finite-dimensional inverse-function
   theorem).

The algebraic formulas, finite-evaluation lemma, recursive reconstruction,
formal kernel argument, and polynomial-in-scale independence argument were
proved directly above rather than inferred from those citations.

## 13. Rough-path extension is separate

Nothing above assumes a rough path.  A later extension would have to specify a
geometric \(p\)-rough path, sufficient \(\operatorname{Lip}^\gamma\) regularity
of the vector fields, continuity of the Itô--Lyons map, and a rough Taylor
theorem that identifies the scale derivatives with the same signature
coefficients.  The universal limit theorem alone gives continuity, not the
required differentiability in \(r\).  No such extension is claimed in this
note.

## 14. Established results of this derivation and exact scope

The proved result is the following.

* Homogeneous injectivity of every \(\rho_m\), \(m\le4\), is sufficient for
  global recursive recovery on \(\mathfrak g^{(L)}\) of the degree-\(m\) Hall
  coefficients from finitely many exact normalized graded scale jets (object
  A).
* For the exact certified tanh witness, the authoritative degree-specific point
  sets give eight scalar ideal-jet evaluations and the exact inverse formulas
  displayed in Section 9.  The existing analytic certificate transfers
  invertibility of those parameter-dependent matrices to Lebesgue-almost every
  tanh parameter vector.
* A nonzero homogeneous kernel direction is invisible in the formal graded
  model.  Standard free-nilpotent horizontal realizability transfers this
  necessity statement to bounded-variation paths for ideal jets through the
  truncation level and for the truncated response.
* Homogeneous injectivity also implies joint state--scale linear independence,
  hence existence of eight scalar evaluations with nonsingular zero-signature
  Jacobian for object B.  The downstream finite-observation theorem selects the
  explicit design \(\mathcal D_*\) displayed above, certifies its exact Jacobian,
  determinant, and inverse, and proves a local analytic inverse.  The downstream
  quantitative theorem certifies inverse-Lipschitz stability on
  \(\|z\|_\infty\le1/2048\) with the exact constants displayed above, together
  with fourth-order common-scale inverse-linearization amplification.  At the
  exact witness every full-rank design needs at least two distinct scales.
  When states are restricted to \(\{-q,0,q\}\), eight distinct scalar rows
  require at least three distinct scales, and \(\mathcal D_*\) uses exactly
  three.
* Any continuous locally injective Euclidean observation map on an open subset
  of the eight-dimensional Hall-coordinate space needs scalar codomain
  dimension at least eight.

These scalar results do not prove ungraded direct-sum injectivity at the
certified scalar witness or recovery of its eight coordinates from one scalar
terminal value. Its mixed-degree obstruction remains unchanged. They also
do not prove a global inverse for object B or C, a finite-difference jet
scheme, a sharp or maximal stability radius, parameter-uniform stability,
equality of the truncated and full responses at finite scale, or a rough-path
extension. The separate fixed-state theorem supplies one-vector local
recovery for object B at \(D\geq\nu_{d,L}\); it does not transfer this
scalar witness's stability constants to that map.

## 15. Subsequent actual-response result and remaining boundary

The [canonical manuscript](../paper/randomized_signature_identifiability.tex),
in “Approximate recovery from actual responses” (Theorem `thm:actual`), now
proves the bounded-variation remainder bridge for the existing scalar witness
and its eight designed experiments. Under a known total-variation bound and
sufficiently small common scale, a uniform fifth-order remainder and clipped
decoder give approximate recovery with explicit truncation, observation-noise,
and iteration-residual bounds. This is an object-C approximation theorem;
it does not identify B with C or give exact recovery at a fixed nonzero scale.

Fixing only the first four log-signature levels still controls neither path
variation nor higher levels, so there is no uniform remainder conclusion here
over all BV realizations of the same \(z\). The actual-response theorem is
specific to the eight scalar experiments. No analogous actual-response bound
or quantitative stability constant is supplied for the new one-vector
fixed-state Hall map.

## Final status

PROVED
