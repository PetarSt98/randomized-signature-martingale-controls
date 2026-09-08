# Mathematical audit for the identifiability manuscript

Audit date: 5 September 2026. Repository HEAD: `48511d9db391c32c689256538828124ad6dbce0e`.

**Decision before manuscript creation: PASS.** No load-bearing defect or disagreement between the exact artifacts, their reproducers, and the corrected theorem statements was found. The appropriate final classification is **PAPER MATHEMATICS VERIFIED WITH EXPLICIT CONDITIONAL STATEMENTS**. General reconstruction is conditional on homogeneous injectivity on the observation domain; the sufficient-width theorem supplies one sufficient architectural condition. Finance experiments remain preliminary. Full-CDE finite-scale reconstruction and a theorem for the implemented discrete recurrence are not established.

This is an audit of the requested theorem chain and its finance motivation. It does not certify every unrelated experiment or untracked rank artifact in the repository. The manuscript is written only after the decision above. Existing mathematical statements and artifacts have not been repaired or generalized in place.

## 1. Source hierarchy and worktree boundary

Exact JSON artifacts and successful exact reproducers determine the matrices, ranks, rational constants, designs, and status strings. The final mathematical notes determine the intended scope. Source and tests provide independent convention and derivative checks. The June 2026 note supplies the reported financial measurements. Primary texts supply prior-work statements.

Principal inspected sources:

- `research_output/technical_note.md`, `certificate_note.md`, `certificate.json`, and `reproduce_certificate.py`;
- `graded_reconstruction_summary.md`, `graded_reconstruction_derivation.md`, `general_graded_identifiability.md`, `graded_reconstruction_certificate.json`, and its reproducer;
- `finite_observation_theorem.md`, `finite_observation_certificate.json`, and its reproducer;
- `local_stability_theorem.md`, `local_stability_certificate.json`, and its reproducer;
- `general_shallow_tanh_width_theorem.md` and `reproduce_tensor_witness.py`;
- the existing `lie_reservoir` free-Lie, bracket, activation, certificate, graded-reconstruction, finite-observation, and tensor-witness implementations, and their focused tests;
- `README.md`, `rsmc/control.py`, `rsmc/hedging.py`, `rsmc/heston.py`, `rsmc/reservoir.py`, `scripts/hedging_demo.py`, and `tmp/research_notes_innovation_project_6.txt`.

Before any edit, `git status --short` and `git diff --stat` were inspected. Four tracked files already differed from HEAD: `AGENTS.md`, `rsmc/__init__.py`, `rsmc/hedging.py`, and `tests/test_hedging.py` (981 insertions, 840 deletions in aggregate). Numerous unrelated untracked experiment files, `.idea/`, and `tmp/` were present. These were left in place. A SHA-256 baseline of every tracked file was retained for the final nonmodification check.

Only the two requested deliverables are created. No new algebra implementation, experiment script, external figure, or bibliography file is introduced.

## 2. Convention audit — VERIFIED

Sources: `free_lie.py`, `brackets.py`, `graded_reconstruction.py`, `finite_observation.py`, `tensor_witness.py`, and the convention fields of the exact artifacts.

The manuscript reserves `d` for input channels, `D` for state dimension, `L` for truncation, `N` exclusively for time steps, `m` for homogeneous degree, `r` for input scale, `h` for a common scale multiplier, and `z` for Hall coordinates. Its intrinsic Hall count is `nu_{d,L}`, avoiding the notes' occasional use of `N_{d,L}`.

Words are chronological: the earliest integration time corresponds to the leftmost letter. Tensor multiplication is concatenation `uv`. With `D_V f = Df V`, `Gamma(i1...im)=D_Vi1 ... D_Vim`. The first letter is the outermost operator; therefore `Gamma(uv)=Gamma(u)Gamma(v)`: this is a homomorphism, not an anti-homomorphism. On linear vector fields `W_i(x)=R_i x`, applying the composed operators to the identity gives the reversed matrix product `R_im ... R_i1 x`. Right concatenation consequently produces the chronological word, without an added reversal.

The bracket is `[V,W]=DW V-DV W`. The Hessian terms cancel in `[D_V,D_W]`, giving `D_[V,W]`. Induction proves `Gamma(H)=D_rho(H)` for every Lie polynomial before application to `Id`. This representation is defined on the full tensor/free-Lie algebra. Finite-degree restriction does not assert that the nonlinear representation factors through the step-L nilpotent quotient.

Lyndon words are degree ordered, then lexicographically ordered; standard Lyndon bracketing splits at the longest proper Lyndon suffix. Internal letters 0,1 are displayed as 1,2. Jets are normalized as `J_m=(1/m!) partial_r^m Y|_0`; their coefficients are `Gamma(S_m)Id`, with no further factorial. One-segment and two-segment exact checks confirm the tensor exponential and chronological Chen product.

## 3. Claim register

The identifiers below also identify the manuscript's labelled statements. “Standard input” means the statement is proved after an explicitly stated classical theorem, not inferred from a numerical check.

### F1. Sample-split pricing (`prop:unbiased`) — VERIFIED

Statement: for the discretized payoff target `p_N=E_Q G_N`, a frozen predictor trained independently of evaluation, predictable prefix features, conditionally centered simulator innovations, and integrable products imply `E[C(Z)|training]=0` and conditional unbiasedness of the evaluation mean for `p_N`.

Sources: `rsmc/control.py`, `README.md`, June note. Dependencies: conditional expectation and independent sample splitting. Training includes preprocessing, reservoir draws, fitting, and validation selection. Simulator innovations are not traded gains. Evaluation controls exclude the fitted intercept and use coefficients transformed back from training standardization. This is not unbiasedness for an undiscretized continuous-time price.

### F2. Finite linear-control projection (`prop:projection`) — VERIFIED

Statement: for a square-integrable centered feature-control vector `Q`, covariance `Sigma`, and `c=E[Q(G_N-p_N)]`, `c` lies in `range(Sigma)` and the minimum residual variance is `Var(G_N)-c^T Sigma^dagger c`; the excess for any coefficient is its `Sigma`-quadratic distance from `Sigma^dagger c`.

Sources: regression setup in `control.py` and `README.md`; elementary covariance calculation supplied in the paper. Handles singular covariance using the Moore–Penrose inverse. This is a population projection identity, not a guarantee for finite-sample ridge fitting. It is classical.

Optional retained-variance perturbation bound relative to finite signatures: **UNSUPPORTED by the inspected project sources; EXCLUDED**. No load-bearing result needs it and the user requested it only if supported.

### F3. Observed-information quadratic hedging (`prop:hedging`) — VERIFIED WITH EXPLICIT STANDARD INPUT

Statement: with the same square-integrable discounted traded martingale under nested information filtrations, the closures in `L^2` of constants plus admissible predictable terminal gains are nested closed subspaces. Orthogonal projection minimizes quadratic risk; nested risks differ by the squared distance between projections.

Sources: `hedging.py`, `heston.py`, June note. Standard input: Hilbert-space projection theorem, whose needed version and proof are supplied. Assumptions: simulator filtration contains all shocks/state/observation noise to date; observed filtration is generated by prices and noisy signals; state filtration additionally observes latent variance. All strategies use only traded-price increments. Closure is explicit; an attained predictable optimizer is not silently asserted. The conditional Gaussian exponential in the log-price update has mean one, yielding the discounted-price martingale, subject to stated integrability. The fitted true-state model is not identified with the population projection or with a nested fitted class.

### F4. Financial measurements — EMPIRICAL / PRELIMINARY

Authoritative source: Petar Stojković's June 2026 research note, available as `tmp/research_notes_innovation_project_6.txt`. No committed detailed run artifact establishing these exact displayed values was located. They were not regenerated.

Context: observation noise `eta=0.02`; 10,000 training / 3,000 validation / 10,000 test paths; all model selection on validation paths; `N=252`; one seed; particle filter only for the one-asset problem.

| Model | One-asset Heston Asian RMSE | Three-asset Heston basket Asian RMSE |
|---|---:|---:|
| Raw | 2.4037 | 2.0981 |
| Reservoir | 1.0167 | 1.0573 |
| Finite signature | 1.1026 | 1.2486 |
| Particle filter, quadratic readout | 2.2579 | unavailable |
| Causal Transformer | 0.8102 | 0.8684 |
| Fitted true-state model | 0.7921 | 0.8496 |

The paper also records the note's descriptive reservoir gap-closure values 92.1% and 89.2%, explicitly as ratios of empirical MSEs, and its qualitative report of similar ordering at `N=52`. It does not invent missing confidence intervals, additional exact measurements, or new runs. Qualitative pricing improvements in the note are motivation, not a quantified theorem. The finite-signature and reservoir improvements over raw features suggest retained history; the later Lie theory does not explain or cause these RMSE values.

### A1. Free-Lie coordinates and realization — VERIFIED WITH EXPLICIT STANDARD INPUT

Statements: the free Lie algebra embeds into the tensor algebra; standard Lyndon brackets form homogeneous bases; the homogeneous dimension is the Witt number. The truncated signature of a continuous BV path has Lie logarithm; conversely each exponential of a truncated free-Lie element is the signature of some piecewise-linear path.

Sources: existing `free_lie.py` plus Reutenauer's Theorems 0.4–0.5, 4.9, Corollary 4.14, and Theorem 5.1; Friz–Victoir author draft Theorems 7.28 and 7.30, checked in the actual text. The realization theorem concerns unrestricted BV inputs in the free step-L group. It gives neither a continuous path-valued local section nor realizability subject to a fixed monotone time channel.

### A2. Evaluation, genericity, and analytic inverse (`lem:evaluation`, `lem:analytic`, `lem:inverse`) — VERIFIED WITH EXPLICIT STANDARD INPUT

Statements: a finite independent family of vector-valued functions has a nonsingular square matrix of scalar point-coordinate evaluations; a nontrivial real-analytic function on a connected Euclidean open set has an open dense nonzero locus with null complement; a square real-analytic map with nonsingular derivative has a local real-analytic inverse.

Sources: corrected general notes; actual Mityagin Proposition 0 for the zero-set theorem. Complete finite-dimensional evaluation and local inverse arguments are supplied in the manuscript. The zero-set argument uses derivative strata; no genericity claim is made before a fixed nonzero determinant is established. Full measure implies almost sure success for absolutely continuous parameter laws, not for every singular randomization.

### S1. Scalar homogeneous theorem (`thm:scalar`) — VERIFIED WITH EXPLICIT STANDARD INPUT

Architecture `V_i(x)=tanh(a_i x+b_i)`, parameter space `R^4`. At `q=log(3)/2=atanh(1/2)` and parameter `(1,0,0,q)`, fields are `tanh(x),1/2`. Exact evaluations give nonzero determinants in degrees 1–4, so each homogeneous map is injective at the witness. Their fixed-point determinant product is nontrivial analytic on `R^4`; hence separate homogeneous injectivity holds on an open dense full-measure certificate set.

Sources: `certificate.json`, `reproduce_certificate.py`, `certificates.py`, and focused tests. All exact matrices and determinants were reproduced. Point sets: degree 1 `(0,q)`; degree 2 `(0)`; degree 3 `(-q,q)`; degree 4 `(-q,0,q)`.

Exact determinants: `-1/4`, `-1/2`, `-45/256`, `135/8192`; product `-6075/16777216`. The scalar witness is not jointly injective through degree four.

### S2. Cross-degree obstruction (`prop:cross`) — VERIFIED

Statement: the eight represented scalar Hall fields are polynomials in `u=tanh(x)` of degree at most six. The coefficient matrix has rank seven, first-seven-column determinant `-3/32`, and complete kernel spanned by `(0,0,4,3,0,0,0,2)` in the fixed Hall order. Thus `rho(4H_3+3H_4+2H_8)=0`.

Sources: finite-observation artifact's `cross_degree_scope_guard`, source and exact reproducer. Independently verified the displayed polynomial expressions against the existing bracket-jet engine through Taylor order six using exact SymPy arithmetic. Substitution by `tanh(x)` is locally invertible, so this is an exact function-rank computation. The mixed-degree relation does not contradict any homogeneous determinant. At one common nonzero scale the full function Jacobian has rank seven; this rules out a nonsingular derivative and a local analytic inverse at zero. It does not, by itself, disprove nonlinear local injectivity.

### G1. Finite Chen–Fliess expansion (`prop:jets`) — VERIFIED WITH EXPLICIT STANDARD INPUT

For smooth known fields on an open state domain, continuous BV input, and each initial state, sufficiently small scales give a local solution and normalized coefficient `J_m=Gamma(S_m)Id`. Proof: local contraction/differentiation of the integral equation and repeated BV chain rule with a bounded order-`L+1` remainder on a compact state neighborhood. A small-scale solution neighborhood is required; a formal infinite series is not used. Tanh is globally bounded and globally Lipschitz for each fixed parameter, but the general smooth-field proof uses local bounds.

Sources: `general_graded_identifiability.md`, `graded_reconstruction_derivation.md`, operator code, exact normalization tests. No state-scaling identity is assumed.

### G2. Graded triangular recovery (`thm:graded`) — CONDITIONAL

The separate algebraic lemma `lem:triangular` (the exact homogeneous tensor-exponential formula and its represented triangular identity) is **VERIFIED**. The conditional status below belongs to reconstruction under injectivity, not to the algebraic identity.

For each `m<=L`, assume the restriction of `rho_m` to vector fields on the observation open set `U` is injective. Then `n_m=dim L_{d,m}` point-coordinate evaluations of ideal normalized degree-m jets recover that Hall block after subtracting the explicitly defined lower-degree correction. The total is `nu_{d,L}=sum n_m` scalar jets. The construction is recursive and valid for every formal Hall vector, not merely near zero.

Sources: general graded note, exact graded artifact/reproducer, `graded_reconstruction.py`. Dependencies: tensor exponential's homogeneous expansion, Lie operator identity, evaluation lemma, and G1 for actual paths. The condition is injectivity on `U`, not injectivity somewhere outside `U` for arbitrary smooth fields. All three prescribed nonzero rational reconstruction examples round-trip exactly.

### G3. Homogeneous kernel and BV interpretation (`prop:kernel`) — CONDITIONAL

If nonzero homogeneous `H` has represented field zero on `U`, then `Gamma(H)` is the zero differential operator there. Replacing `ell` by `ell+tH` changes no formal graded coefficient through L and no degree-L truncated response, for every real `t`. Every expansion term in the difference contains a zero operator factor. Homogeneity ensures the truncation retains or discards the whole operator product.

Dependencies: operator-level bracket identity; Friz–Victoir realization for the existence of actual PL paths with these truncated signatures. The paths may have different higher signatures and full CDE responses. The scalar mixed-degree kernel alone is not an instance of this homogeneous-kernel proposition.

### O1. Finite state-scale observation (`thm:finite`) — CONDITIONAL

Under G2's homogeneous-injectivity assumption, the functions `r^m rho(H)(y)_k` are jointly independent on any nonempty open scale interval avoiding zero, the state domain, and coordinate indices. Grouping powers of `r` first proves this. Consequently `nu_{d,L}` scalar observations of the degree-L truncated polynomial response have nonsingular zero-coordinate Jacobian and a local analytic inverse. The map is polynomial in Hall coordinates even for smooth, nonanalytic state fields.

Sources: general graded note and finite-observation note; dependencies: finite evaluation and analytic inverse. Observations may vary initial states, scales, and output coordinates. This is Object B, not the actual full CDE Object C. Object A consists of ideal derivatives at zero scale.

### O2. Scalar-output lower bound (`prop:dimension`) — VERIFIED WITH EXPLICIT STANDARD INPUT

A continuous map on an open subset of `R^nu` cannot be locally injective into `R^M` when `M<nu`. Pad with zeros and apply invariance of domain to obtain the contradiction of an open subset lying in a proper linear subspace. Source: actual Hatcher Theorem 2B.3. This counts real outputs, not reservoir runs.

### O3. Exact eight-output decoder (`thm:eight`) — VERIFIED WITH EXPLICIT STANDARD INPUT

Ordered design: `(-q,1),(-q,2),(-q,3),(0,2),(0,3),(q,1),(q,2),(q,3)`. Centered nonlinear response is defined by the tensor exponential through degree four. Its actual derivative at zero is the certified 8-by-8 matrix `B`; determinant `-258733035/524288`. Analytic inverse follows. The same fixed design works on an open dense full-measure certificate subset of `R^4` by parameter analyticity.

Sources: exact finite-observation artifact, reproducer, implementation, focused tests. Exact tensor-based and Lie-based Jacobians agree. Differentiation of the nonlinear coordinate polynomials agrees as well; exact quartic interpolation tests are an additional independent check. There are 486 nonsingular eight-row subsets among 495 subsets of the prescribed twelve candidates. Every single-row deletion of the selected matrix has rank seven. A tested single-scale family has exact function rank seven. The literal response retains irrational starting states; the centered response removes only known constants, not terms of the nonlinear polynomial.

### O4. Exact local inverse-Lipschitz certificate (`thm:stability`) — VERIFIED

Sources: `local_stability_certificate.json`, reproducer, existing exact polynomial and matrix code. Domain is the closed coordinate cube `||z||_infinity<=1/2048`, at the fixed scalar witness and fixed design. With `K=||B^-1||_infinity=23812/2151` and the coefficient-sum majorant `Delta(R)`, all sixteen prescribed radii were recomputed. The largest passing listed radius is `R*=1/2048`.

Reproduced values:

- `Delta*=2643019857/68719476736`;
- `K Delta*=1748210800969/4105988734976`;
- `1/2-K Delta*=304783566519/4105988734976`;
- `c*=21220001406063/409087045009408`;
- `C*=409087045009408/21220001406063`;
- simpler inverse bound `2K=47624/2151`.

The proof preconditions by the fixed matrix `B^-1`, integrates the perturbation along a segment in the convex cube, and then uses the reverse triangle inequality around the fixed displacement. It does not use an invalid lower bound for the norm of an integral. The conditional noise corollary requires both compared coordinates to remain in that cube; arbitrary noisy data need not have a preimage there. The certificate is not an optimal radius or a parameter-uniform bound.

### O5. Common-scale laws (`prop:scales`) — VERIFIED

Hall degrees are `(1,1,2,3,3,4,4,4)`. Replacing every design scale by `h` times that scale gives `B(h)=B diag(h^degrees)` and determinant `h^22 det B`. The exact inverse norm is the maximum of `4547/717 |h|^-1`, `23812/2151 |h|^-2`, `824/99 |h|^-3`, and `160/33 |h|^-4`. For `0<|h|<=1`, it lies between `160/33 |h|^-4` and `23812/2151 |h|^-4`. This is amplification by the inverse linearization, not a matrix condition-number identity or an unchanged nonlinear stability cube.

### S3. State-dimension lift (`cor:lift`) — VERIFIED WITH EXPLICIT STANDARD INPUT

For every `D>=1`, `E(f)(x)=f(x_1)e_1` preserves brackets and every word-response first coordinate globally. The scalar witness embeds into the shallow componentwise-tanh architecture using zero inactive rows/biases. Separate homogeneous injectivity through four, the scalar mixed-degree defect, and the fixed witness's scalar-coordinate decoder and constants are preserved. Fixed lifted homogeneous minors give generic separate homogeneous injectivity in the full D-dimensional parameter space. The quantitative constants are asserted only at the embedded witness. This is a corollary, not a width-expressivity or monotonic-improvement theorem.

### W1. Linear-to-tanh transfer (`lem:transfer`) — VERIFIED

Given linear fields on a finite-dimensional state space and a nonzero fixed evaluation minor for a finite family of words, replacing each field by componentwise `tanh(epsilon times the linear field)` retains that word minor for sufficiently small nonzero epsilon after extracting one factor epsilon per word letter.

Dependencies: finite-jet analyticity of `tanh(epsilon v)/epsilon` with its removable value at zero; continuity of a finite determinant; exact operator multilinearity under constant field scaling. The normalized fields are an auxiliary family; the raw fields vanish at epsilon zero. No numerical epsilon or all-epsilon triangular determinant formula is required.

### W2. General sufficient width (`thm:width`) — VERIFIED WITH EXPLICIT STANDARD INPUT

For every finite `d,L>=1` and `D>=D0=sum_{m=0}^L d^m`, there is an open dense full-Lebesgue-measure certificate set in `R^{d(D^2+D)}` for which all `Gamma_theta(w)Id`, `1<=|w|<=L`, are jointly independent. Therefore `rho_theta` is injective on the direct sum of the homogeneous free-Lie spaces through L.

Sources: corrected width note and existing tensor-witness code. Proof: right-concatenation matrices on the truncated tensor-word space, initial state `e_empty`, nonempty-word coordinate rows. The linear word matrix is exactly the identity. W1 yields a nonzero raw determinant at an existential small epsilon. The fixed determinant is analytic in every entry of every weight and bias. Zero-set genericity then applies. For larger D, a zero inactive-coordinate block preserves the minor globally. The tensor construction is classical; the manuscript identifies the architecture-specific transfer separately.

The strongest proved conclusion is joint finite-degree word-response field independence. It is stronger than joint Hall independence and homogeneous injectivity. No statement concerns trees, degrees beyond L, a nilpotent quotient representation of tanh, minimal width, compressed width, numerical conditioning, explicit epsilon, uniform stability, actual full-CDE reconstruction, the discrete recurrence, or finance performance.

### W3. Finite structural harness — VERIFIED computational checks

The exact harness checks `(d,L)=(2,2),(2,3),(2,4),(3,2)`, respectively `(D0,Q,Hall rank)=(7,6,3),(15,14,5),(31,30,8),(13,12,6)`. Linear word matrices are exact identities; words of length L+1 vanish; segment flows and chronological Chen products match; Hall expansions and exact ranks agree. Block and normalized-tanh diagnostics also run in the smaller cases. Tanh-limit floating-point diagnostics are expressly non-certifying. These checks verify implementation instances; the arbitrary-d,L proof is structural and independent of them.

## 4. Primary-source audit and contribution boundary

Actual text, rather than only a search summary, was inspected for the sources below. Numbered locators are restricted to passages actually checked.

| Primary source opened | Checked content and manuscript use |
|---|---|
| [Cuchiero et al., Discrete-time signatures and randomness in reservoir computing](https://arxiv.org/pdf/2010.14615) | Theorems III.7–III.8 and surrounding SigSAS/random-projection argument. Prior discrete-time approximation theory; not an exact shallow-tanh word determinant. |
| [Akyildirim et al., Market anomaly detection](https://arxiv.org/pdf/2201.02441) | Proposition 2.10's tensor system, Definition 2.12, Theorem 2.13 and its proof. Broad recovery sketch does not exhibit the nonzero analytic witness needed here; activation hypotheses do not directly include tanh. No reliance on that recovery claim. |
| [Akyildirim et al., Optimal portfolio selection](https://arxiv.org/pdf/2312.16448) | Introduction and methodology: reservoir drift prediction and empirical portfolio application. Prior motivation/application, not a certificate of this finite-degree theorem. |
| [Glückstad, Muça Cirone, Teichmann, 2025 v1](https://arxiv.org/html/2502.03163v1) | Lemma 3.4, Proposition 3.6, Section 3.3, Remark 3.10, Theorem 4.1 proof, and Section 5. Tree independence implies per-degree word-operator independence; the depth-two nested-exponential construction has a smaller sufficient state dimension for a different architecture and stronger tree condition. No ranking. |
| [Bailleul–Diehl, The inverse problem for rough controlled differential equations](https://arxiv.org/pdf/1407.2768) | Theorem 6 and reconstruction matrix: multiple initial states, fields and brackets, finite small-time flow observations. Prior inverse-problem framework; no rough-path theorem is added here. |
| [Reutenauer, Free Lie Algebras](https://academic.oup.com/book/53911), [actual book transcription/scan](https://djvu.online/file/nbRaButjp7ffB) | Theorems 0.4–0.5, 4.9, Corollary 4.14, Theorem 5.1: free-Lie tensor embedding and Lyndon/Hall bases. Publisher contents and actual book text checked. |
| [Friz–Victoir author draft](https://page.math.tu-berlin.de/~friz/master4_May6th.pdf) | Theorems 7.28 and 7.30, printed pp. 142–144 in the May 2009 draft: PL signature realization and the exponential free nilpotent group. |
| [Burde–Eick–de Graaf](https://arxiv.org/pdf/0807.2345) | Section 2, Theorem 2.1: classical finite-dimensional weighted enveloping-module construction. Orientation in this repository is checked independently. |
| [Mityagin, The zero set of a real analytic function](https://arxiv.org/pdf/1512.07276) | Proposition 0 and proof: nontrivial analytic zero sets are Lebesgue null. Used instead of pretending access to a paywalled numbered Krantz–Parks theorem. |
| [Fouque–Han](https://www.numdam.org/item/10.1051/ps:2007005.pdf) | Published paper, centered stochastic-integral controls and unbiased pricing estimator in the opening derivation. Classical martingale-control antecedent. |
| [Mania–Tevzadze–Toronjadze](https://arxiv.org/pdf/math/0703424) | Restricted-information mean-variance hedging and its continuous-time semimartingale formulation. Related projection problem, not the finite experiment's population solution. |
| [Schweizer, Risk-minimizing hedging strategies under restricted information](https://people.math.ethz.ch/~mschweiz/Files/restrinf.pdf) | Proposition 1.2, Section 2, Theorem 2.5: GKW and restricted-information projection; distinct risk-minimization setup is acknowledged. |
| [Frey–Runggaldier, author-uploaded full article](https://www.researchgate.net/publication/226761883_Risk-minimizing_hedging_strategies_under_restricted_information_The_case_of_stochastic_volatility_models_observable_only_at_discrete_random_times) | Actual article introduction and model: stochastic volatility with discrete random observation times, restricted-information risk minimization and filtering. Close discrete-observation precedent, not identical to the present fixed-grid global risk objective. |
| [Hatcher, Algebraic Topology](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf) | Theorem 2B.3: invariance of domain in equal finite dimensions, applied after padding the target. |

No worldwide priority claim is supported or made. The scalar exact certificates, direct graded Hall recovery, fixed eight-output truncated decoder, exact stability certificate, and normalized shallow-tanh transfer are presented in their exact proved form. In the primary sources checked for this manuscript, no proof of the finite-degree shallow componentwise-tanh transfer theorem in that stated form was found. The existing note's phrase “technically new” is not carried into the paper.

### External statement excluded from the proof

The general same-field state/input-scaling identity used in the displayed proof of Theorem 4.1 of the 2025 v1 paper is **FALSE without an extra homogeneity hypothesis**. In one channel, take `V(x)=tanh(x)`, initial state `a=q`, scale `r=2`, and driving path `X_t=t`. The derivative at time zero of the input-scaled solution is `2 tanh(q)=1`; that of `2Y^{q/2,1}` is `2 tanh(q/2)`, which differs. A bare mth derivative also differs from its normalized Taylor coefficient by `m!`.

This is not a defect in the repository's corrected theorem chain: it defines independently accessible input scaling and normalized jets and does not use that identity. The related-work discussion reports the prior tree/operator construction and stated reconstruction programme, without importing its flow-to-scale step as an established premise. The current note already describes the tree and word conclusions with the correct distinct scope; no load-bearing novelty comparison depends on the excluded identity. Thus this external caveat does not trigger the manuscript stop condition. If reconstruction from unscaled ordinary flows were required instead, the missing bridge would be blocking.

## 5. Translation changes and limitations

These are explicit qualifications in the new manuscript, not edits to the authoritative theorem package:

1. Reserve `N` for time steps and use `nu_{d,L}` for the intrinsic coordinate count.
2. Use “word-response field” for `Gamma(w)Id`; prove the operator identities before evaluating `Id`.
3. Define the representation on the full algebra and restrict its vector space by degree; do not assert quotient factorization.
4. State injectivity on the observation domain for general smooth fields.
5. Supply local BV existence, scale differentiability, and a finite remainder argument; distinguish normalized jets from raw derivatives.
6. Keep ideal jets (A), truncated finite-scale responses (B), and actual CDE responses (C) separate.
7. Cite the PL realization theorem before interpreting formal coordinate counterexamples as path-signature counterexamples.
8. Interpret the mixed-degree rank defect as a singular-linearization obstruction, not a proof of nonlinear noninjectivity.
9. State the inverse-Lipschitz result on a fixed closed convex cube, and make the noise corollary conditional on preimages in that cube.
10. Treat the scale law as inverse-linearization amplification; it does not transfer the same nonlinear domain to rescaled experiments.
11. Keep the normalized tanh limit auxiliary, use existential epsilon only, and retain the strongest proved word-response independence conclusion.
12. Replace unsupported novelty adjectives with a bounded primary-source search statement. Do not use the external scaling identity identified above.
13. Distinguish simulator innovations, traded gains, Hilbert-space closures, population projections, and fitted benchmarks.
14. Quote financial values from the June note, mark them single-seed, and exclude the optional unsupported retained-variance bound.

Unresolved bridges: finite full-CDE observations to truncated coefficients with a proved remainder decoder; transfer to the implemented discrete recurrence; compressed-width conditions; quantitative conditioning of the sufficient-width construction; parameter-uniform stability; and a theorem linking these representation properties to finance risk. These are limitations, not work undertaken in this task.

## 6. Executed validation

All five default reproducers completed successfully, without artifact rewriting:

```text
python research_output/reproduce_certificate.py
python research_output/reproduce_graded_reconstruction.py
python research_output/reproduce_finite_observation.py
python research_output/reproduce_local_stability.py
python research_output/reproduce_tensor_witness.py
```

The scalar source candidate retained status `numerical_candidate_only`; the rational certificate retained status `exact_nonzero`, with method `exact_symbolic_determinant_simplification`. Numerical discovery was not relabelled as proof. The finite design search was reproduced exactly before the selected-minor proof was used.

| Test file | Result |
|---|---:|
| `tests/test_certificates.py` | 7 passed |
| `tests/test_graded_reconstruction.py` | 28 passed |
| `tests/test_finite_observation.py` | 21 passed |
| `tests/test_local_stability.py` | 18 passed |
| `tests/test_tensor_witness.py` | 40 passed |
| Complete suite | **233 passed** |

Tests were invoked with `python -m pytest ... -q -p no:cacheprovider` and `PYTHONDONTWRITEBYTECODE=1`, preserving the requested assertions while avoiding new cache artifacts. The full suite completed in 118.74 seconds. Environment: Python 3.13.3, NumPy 2.3.1, SymPy 1.14.0, pytest 9.1.1, PyTorch 2.12.1.

Global `git diff --check` exits 2 on 804 pre-existing trailing-whitespace reports, confined to `rsmc/__init__.py`, `rsmc/hedging.py`, and `tests/test_hedging.py`. These files are not normalized or repaired. Final checks of the two newly created files and tracked-file hashes are recorded below after manuscript construction.

## 7. Delivery validation

Neither `pdflatex` nor `latexmk` is installed or on PATH; standard local MiKTeX/TeX Live locations were also absent. No installation was attempted. The standalone LaTeX source is therefore delivered without a PDF. Compilation, overfull-box checks, page rendering, and PDF page count are unavailable. Static source checks do not replace compilation or visual verification.

Final source checks found 96 unique labels and 15 bibliography entries, with no missing referenced labels, no missing citation keys, balanced braces, matched environments, no remaining table-generation placeholders, no control characters, and no trailing whitespace in either new file. All 15 bibliography entries are cited. The full eight-output polynomial was checked from its 27-monomial table against the existing implementation at all eight state-scale pairs, with 190 nonzero terms. The manuscript uses only the permitted standalone source structure and an embedded bibliography.

Scoped `git diff --no-index --check -- NUL <new-file>` reported no whitespace errors for either new file. Its exit status is 1 because a new file differs from the empty comparison file; this is not reported as a zero-exit check. Git also warned about its configured future LF-to-CRLF conversion. An independent direct line check found no trailing whitespace and confirmed a final newline in both files. Global whitespace counts remain exactly 53 in `rsmc/__init__.py`, 410 in `rsmc/hedging.py`, and 341 in `tests/test_hedging.py`.

The final SHA-256 comparison covered all 70 tracked files: **zero differences from the pre-edit baseline**. The only added deliverables are this audit and `paper/randomized_signature_identifiability.tex`. No existing source, certificate, test, theorem note, or finance file was modified. No PDF, bibliography file, helper script, or new experiment artifact was created.

### Complete labelled-result classification

| Status | Manuscript results |
|---|---|
| VERIFIED | `prop:unbiased`, `prop:projection`, `prop:cross`, `lem:triangular`, `thm:stability`, `prop:scales`, `lem:transfer` |
| VERIFIED WITH EXPLICIT STANDARD INPUT | `prop:hedging`, `lem:evaluation`, `lem:analytic`, `lem:inverse`, `thm:scalar`, `prop:jets`, `prop:dimension`, `thm:eight`, `cor:lift`, `thm:width` |
| CONDITIONAL | `thm:graded`, `prop:kernel`, `thm:finite` |
| EMPIRICAL / PRELIMINARY | The June-note finance table, descriptive gap closure, and qualitative experiment observations |

Classical free-Lie embedding, Hall bases, Witt dimensions, and BV signature realization are explicitly stated standard inputs in the preliminaries. Definitions, assumptions, and finite computational checks are not relabelled as mathematical theorems.

**Final decision: PAPER MATHEMATICS VERIFIED WITH EXPLICIT CONDITIONAL STATEMENTS.** Compilation and visual layout are unverified, as specified above.

## 8. Exact certificate transcription

The following tables are transcribed mechanically from the authoritative JSON after successful reproduction. They supplement the claim register; the manuscript contains the same exact quantities and the mathematical reason each matters.


### Homogeneous matrices and inverses

#### Degree 1

Points: 0, log(3)/2.

Matrix:

```text
['0', '1/2']
['1/2', '1/2']
```

Inverse:

```text
['-2', '2']
['2', '0']
```

#### Degree 2

Points: 0.

Matrix:

```text
['-1/2']
```

Inverse:

```text
['-2']
```

#### Degree 3

Points: -log(3)/2, log(3)/2.

Matrix:

```text
['15/32', '3/16']
['15/32', '-3/16']
```

Inverse:

```text
['16/15', '16/15']
['8/3', '-8/3']
```

#### Degree 4

Points: -log(3)/2, 0, log(3)/2.

Matrix:

```text
['-57/128', '-3/32', '3/64']
['-1/2', '0', '1/4']
['-57/128', '3/32', '3/64']
```

Inverse:

```text
['-64/45', '8/15', '-64/45']
['-16/3', '0', '16/3']
['-128/45', '76/15', '-128/45']
```

### Eight-output matrix and inverse

```text
B =
['-1/2', '1/2', '-3/8', '15/32', '3/16', '-57/128', '-3/32', '3/64']
['-1', '1', '-3/2', '15/4', '3/2', '-57/8', '-3/2', '3/4']
['-3/2', '3/2', '-27/8', '405/32', '81/16', '-4617/128', '-243/32', '243/64']
['0', '1', '-2', '4', '0', '-8', '0', '4']
['0', '3/2', '-9/2', '27/2', '0', '-81/2', '0', '81/4']
['1/2', '1/2', '-3/8', '15/32', '-3/16', '-57/128', '3/32', '3/64']
['1', '1', '-3/2', '15/4', '-3/2', '-57/8', '3/2', '3/4']
['3/2', '3/2', '-27/8', '405/32', '-81/16', '-4617/128', '243/32', '243/64']

B inverse =
['-18/11', '9/22', '-2/33', '0', '0', '18/11', '-9/22', '2/33']
['360/239', '126/239', '-88/717', '-405/239', '80/239', '360/239', '126/239', '-88/717']
['-76/239', '547/239', '-964/2151', '-990/239', '1760/2151', '-76/239', '547/239', '-964/2151']
['-760/717', '1168/717', '-664/2151', '-432/239', '256/717', '-760/717', '1168/717', '-664/2151']
['-76/33', '52/33', '-28/99', '0', '0', '76/33', '-52/33', '28/99']
['-3056/10755', '476/1195', '-2992/32265', '-1244/3585', '808/10755', '-3056/10755', '476/1195', '-2992/32265']
['-40/33', '32/33', '-8/33', '0', '0', '40/33', '-32/33', '8/33']
['-472/10755', '646/3585', '-2264/32265', '-1018/3585', '1276/10755', '-472/10755', '646/3585', '-2264/32265']
```

Exact LU additionally gives one row swap (second/fourth) and pivots `-1/2, 1, -9/4, -15/16, -3, 4863/1280, -9339/3242, -32265/2264`; the signed product agrees with the artifact determinant. This is an auxiliary exact check of the existing matrix.

Inverse absolute row sums: 139/33, 4547/717, 23812/2151, 17552/2151, 824/99, 63644/32265, 160/33, 31978/32265.

### Jacobian-variation row sums

| Row | Degree 1 | Degree 2 | Degree 3 |
|---|---|---|---|
| 1 | 225/128 | 189/256 | 47/128 |
| 2 | 135/8 | 165/16 | 47/8 |
| 3 | 9369/128 | 12717/256 | 3807/128 |
| 4 | 18 | 8 | 2 |
| 5 | 315/4 | 297/8 | 81/8 |
| 6 | 225/128 | 189/256 | 47/128 |
| 7 | 135/8 | 165/16 | 47/8 |
| 8 | 9369/128 | 12717/256 | 3807/128 |

### Complete exact radius search

| Radius | Delta | K Delta | Pass |
|---|---|---|---|
| 1/2 | 3375/64 | 2232375/3824 | False |
| 1/4 | 11349/512 | 7506733/30592 | False |
| 1/8 | 42777/4096 | 28294609/244736 | False |
| 1/16 | 166113/32768 | 109874521/1957888 | False |
| 1/32 | 654705/262144 | 433050985/15663104 | False |
| 1/64 | 2599569/2097152 | 1719470473/125304832 | False |
| 1/128 | 10360017/16777216 | 6852575689/1002438656 | False |
| 1/256 | 41363793/134217728 | 27359851081/8019509248 | False |
| 1/512 | 165302865/1073741824 | 109338661705/64156073984 | False |
| 1/1024 | 660907089/8589934592 | 437153322313/513248591872 | False |
| 1/2048 | 2643019857/68719476736 | 1748210800969/4105988734976 | True |
| 1/4096 | 10570862673/549755813888 | 6992038388041/32847909879808 | True |
| 1/8192 | 42281017425/4398046511104 | 27966544081225/262783279038464 | True |
| 1/16384 | 169119203409/35184372088832 | 111862957543753/2102266232307712 | True |
| 1/32768 | 676467081297/281474976710656 | 447445392773449/16818129858461696 | True |
| 1/65536 | 2705848860753/2251799813685248 | 1789768696451401/134545038867693568 | True |

### Graded round trips

Check 1:

```json
{
  "input_hall_coordinates": [
    [
      "1",
      "-1"
    ],
    [
      "2"
    ],
    [
      "3",
      "-2"
    ],
    [
      "-1",
      "2",
      "-3"
    ]
  ],
  "observed_normalized_jets": [
    [
      "-1/2",
      "0"
    ],
    [
      "-5/4"
    ],
    [
      "37/32",
      "3/2"
    ],
    [
      "445/512",
      "5/32",
      "207/256"
    ]
  ],
  "recovered_hall_coordinates": [
    [
      "1",
      "-1"
    ],
    [
      "2"
    ],
    [
      "3",
      "-2"
    ],
    [
      "-1",
      "2",
      "-3"
    ]
  ],
  "exact_equality_verified": true
}
```

Check 2:

```json
{
  "input_hall_coordinates": [
    [
      "1/2",
      "-2/3"
    ],
    [
      "5/7"
    ],
    [
      "-3/5",
      "4/9"
    ],
    [
      "7/11",
      "-8/13",
      "9/14"
    ]
  ],
  "observed_normalized_jets": [
    [
      "-1/3",
      "-1/12"
    ],
    [
      "-37/84"
    ],
    [
      "-5239/32256",
      "-13817/32256"
    ],
    [
      "-1599438283/15498362880",
      "-393733/1995840",
      "-6666901981/15498362880"
    ]
  ],
  "recovered_hall_coordinates": [
    [
      "1/2",
      "-2/3"
    ],
    [
      "5/7"
    ],
    [
      "-3/5",
      "4/9"
    ],
    [
      "7/11",
      "-8/13",
      "9/14"
    ]
  ],
  "exact_equality_verified": true
}
```

Check 3:

```json
{
  "input_hall_coordinates": [
    [
      "-5/3",
      "2/7"
    ],
    [
      "-11/4"
    ],
    [
      "13/8",
      "-17/9"
    ],
    [
      "19/10",
      "-23/12",
      "29/15"
    ]
  ],
  "observed_normalized_jets": [
    [
      "1/7",
      "-29/42"
    ],
    [
      "211/168"
    ],
    [
      "27269/84672",
      "63241/84672"
    ],
    [
      "-74384759/142248960",
      "-2116939/4445280",
      "-45565913/17781120"
    ]
  ],
  "recovered_hall_coordinates": [
    [
      "-5/3",
      "2/7"
    ],
    [
      "-11/4"
    ],
    [
      "13/8",
      "-17/9"
    ],
    [
      "19/10",
      "-23/12",
      "29/15"
    ]
  ],
  "exact_equality_verified": true
}
```

### Non-certifying numerical values preserved

Scalar homogeneous condition numbers: `2.618034, 1.0, 2.5, 6.2704558`. Binary64 exact-product conversion: `-0.00036209821701049805`. Source numerical candidate: 42 points; weight/point/extension seeds `1729, 2719, 1000002719`; zero-based rows `18,32,22`, columns `0,1,2`; determinant `-5.041803997872895`, condition number `1.9857803041316449`; status remains `numerical_candidate_only`.

Eight-output binary64 condition number: `468.626825569`; singular values: `72.8678806904, 13.4576527068, 12.4585453832, 2.9110008089, 1.23631176424, 0.365987026873, 0.197225996535, 0.155492337857`. These diagnostics do not certify rank.

The manuscript also contains a full 27-monomial coefficient table. It was generated using the existing `exact_centered_observation_polynomials` function, and substitution of each design scale was checked to reproduce all eight polynomial dictionaries (190 nonzero terms). No new word, tensor, or Lie implementation was used.
