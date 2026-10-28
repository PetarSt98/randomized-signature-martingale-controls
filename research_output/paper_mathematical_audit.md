# Mathematical audit for the identifiability manuscript

Audit date: 5 September 2026. Repository HEAD: `48511d9db391c32c689256538828124ad6dbce0e`.

**Update, 8 September 2026:** The final addendum records the proved approximate-recovery extension for the fixed eight actual CDE responses and its deterministic validation. The original audit and its historical scope below are retained; exact recovery at a fixed nonzero scale is still not asserted.

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
- `README.md`, `rsmc/control.py`, `rsmc/hedging.py`, `rsmc/heston.py`, `rsmc/reservoir.py`, `scripts/hedging_demo.py`, and `research_output/sources/research_notes_innovation_project_6.txt` (originally under `tmp/`).

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

Authoritative source: Petar Stojković's June 2026 research note, preserved as `research_output/sources/research_notes_innovation_project_6.txt` (originally under `tmp/`). No committed detailed run artifact establishing these exact displayed values was located. They were not regenerated.

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

**2026-09-08 manuscript addendum.** Corollary `cor:zero` fixes `h0=0` at the unchanged sufficient width. Translating `c=e_empty` into the biases by `b_i -> b_i+A_i c` preserves exactly the existing `Q_{d,L}`-square nonempty-word minor at a finite nonzero-epsilon witness. Its determinant at zero is nontrivial real analytic in the full weights and biases, giving almost-everywhere independence of the nonempty word responses and the existing Lie-faithfulness consequences. The empty-word column is excluded and `S_0=1` is known separately; the scalar eight-observation design is unchanged.

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

## 2026-09-08: Approximate recovery from eight actual CDE responses

The canonical manuscript's `thm:actual` proves one extension of the existing scalar certificate. The fields, Hall order, eight pairs, three initial states, polynomial map and exact artifacts are unchanged. For a continuous BV path starting at zero with total component variation at most \(M\), use \(0<h\leq1\) and \(c_m(hM)^m\leq R\), where
\[
(c_1,c_2,c_3,c_4)=(1,1,1,13/12),\qquad R=1/2048.
\]
For \(M>0\), \(h\leq\min(1,R/M)\) suffices. Only \(D_hz(X)\), not necessarily \(z(X)\), must lie in the certified cube.

**Proof and constants.** Every response monomial has Hall weighted degree at most four, so the complete polynomial satisfies \(\Phi_h(z)=\Phi(D_hz)\). The tensor coefficient \(\ell^1\) bound and the existing identity Hall pivot matrices give the stated \(c_m\). The outermost-first recursion \(P_{iv}=v_i(1-s^2)P_v'\), \(v_1=s,v_2=1/2\), gives the conservative global bound \(K_5=2^4(1\cdot3\cdot5\cdot7)=1680\). The finite integral remainder and total component-variation measure give \(A=3^5K_5/5!=3402\), with no hidden scale dependence.

The certificate uses the **induced row-sum infinity norm**, not an entrywise maximum:
\[
K=23812/2151,\quad
\Delta(R)=2643019857/68719476736,\quad
\kappa=1748210800969/4105988734976<1/2,\quad
C=K/(1-\kappa)=409087045009408/21220001406063.
\]
The coordinatewise clipped contraction defines \(G(b)\) for every \(b\in\mathbb R^8\), agrees with the inverse on \(\Phi(Q_R)\), and is \(C\)-Lipschitz. With response error \(\|e\|_\infty\leq\delta\), the recovered coordinates obey
\[
\|D_h^{-1}G(b_h(X)+e)-z(X)\|_\infty
\leq C(3402M^5h+\delta h^{-4}).
\]
A computed iterate adds \(h^{-4}r_n/(1-\kappa)\), where \(r_n=\|u_n-T_b(u_n)\|_\infty\) is the exact fixed-point residual. An unscaled iteration tolerance \(\tau\) therefore requires \(r_n\leq(1-\kappa)h^4\tau\). This decoder is neither a global polynomial inverse nor a least-squares minimizer.

**Deterministic validation.** The single script `research_output/validate_actual_response_recovery.py` writes no artifacts. It reads and verifies the existing design, inverse and stability constants; checks exact full-polynomial scaling at \(h=1/2,-2,1/2048\); checks the Hall pivot conversion, tensor logarithm/re-exponentiation and path reversal; and compares the length-five polynomial recursion with existing exact word jets at three states. Its exact coefficient-norm enumeration returns 560, below the proof's deliberately unoptimized 1680.

An axis segment is written (channel, signed increment), with channels numbered 1 and 2 here:

- Alternating: \((1,1/3),(2,1/4),(1,-1/5),(2,1/6)\), total variation \(M=19/20\).
- Rectangle: \((1,1/3),(2,1/4),(1,-1/3),(2,-1/4)\), \(M=7/6\).
- Reversed: reverse the alternating segments and negate every increment; \(M=19/20\).

Reference signatures are exact chronological products of segment tensor exponentials; their tensor logarithms and Hall coordinates use rational arithmetic. Actual responses independently compose \(F_a(y)=\operatorname{asinh}(e^a\sinh y)\) and \(H_b(y)=y+b/2\) in path order, with scaled signed increments. Runs use \(h=R/(M j)\), \(j=1,2,4\). Perturbed runs use \(e_i=(-1)^{i-1}\delta\), \(\delta=(hM)^5/100\); this is response error, unrelated to finance noise.

The table records the 80-digit run. Coordinate errors use the exact rational Hall reference. Each residual column is the maximum over the clean and perturbed runs at that path/scale, and has its own definition: fixed-point residual \(r_n\) versus response-fit residual \(\|\Phi(u_n)-b\|_\infty\).

| Path | \(h\) | Response remainder | Clean coordinate error | Perturbed coordinate error | Max fixed-point residual | Max response-fit residual |
|---|---:|---:|---:|---:|---:|---:|
| Alternating | 5.139803e-4 | 3.812074e-19 | 1.580985e-7 | 1.915026e-5 | 1.248493e-70 | 1.768738e-69 |
| Alternating | 2.569901e-4 | 1.191486e-20 | 7.905598e-8 | 9.574754e-6 | 7.017146e-72 | 9.940581e-71 |
| Alternating | 1.284951e-4 | 3.723727e-22 | 3.952967e-8 | 4.787282e-6 | 1.577444e-72 | 2.234561e-71 |
| Rectangle | 4.185268e-4 | 4.764082e-19 | 3.212782e-6 | 4.064667e-5 | 3.987309e-75 | 2.882609e-74 |
| Rectangle | 2.092634e-4 | 1.488829e-20 | 1.606525e-6 | 2.032320e-5 | 8.661077e-75 | 5.920002e-74 |
| Rectangle | 1.046317e-4 | 4.652676e-22 | 8.032961e-7 | 1.016157e-5 | 1.651962e-80 | 1.129146e-79 |
| Reversed | 5.139803e-4 | 3.187471e-18 | 4.217411e-6 | 1.575957e-5 | 1.248394e-70 | 1.768310e-69 |
| Reversed | 2.569901e-4 | 9.963001e-20 | 2.109350e-6 | 7.880380e-6 | 7.016742e-72 | 9.939202e-71 |
| Reversed | 1.284951e-4 | 3.113774e-21 | 1.054836e-6 | 3.940339e-6 | 1.577385e-72 | 2.234386e-71 |

All 18 clean/perturbed cases satisfy the remainder and combined recovery bounds including the solver term. Iterations number 4–17. The conservative data-plus-truncation bounds range from about 6.52 to 59.33; they prove a uniform estimate and are much larger than the observed errors.

Five polynomial round trips include zero, both constant-sign corners, an alternating-sign corner and a face point. Their maximum coordinate errors are \(3.95609\times10^{-56}\) at 80 digits and \(1.67220\times10^{-97}\) at 120 digits. A deliberately off-image datum \(b=\Phi(R\mathbf1)+B(R\mathbf1)\) returns the upper corner with computed fixed-point residual zero and response-fit residual approximately 0.0124626. All 18 path cases were repeated at 120 digits; maximum coordinate disagreement was \(3.7345235\times10^{-57}\). The stopping target was an unscaled accuracy \(10^{-(p-25)}\) at \(p\) decimal digits, with the required \(h^4(1-\kappa)\) factor.

Reproduce from the repository root (Python 3.13.3, SymPy 1.14.0, mpmath 1.3.0):
```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python research_output/validate_actual_response_recovery.py
```

The exact checks use existing utilities; the flow evaluations, numerical residuals and precision agreement are **numerical evidence, not new exact certificates or interval error enclosures**. Fixed nonzero response noise is amplified as \(h^{-4}\). Approximate recovery uses the designed eight actual responses; exact fixed-scale recovery, one zero-initialized scalar reservoir, discrete/Euler transfer, rough paths and finance consequences are not asserted.

**Checks performed for this extension.** The validation command above passed at its default 80/120-digit settings. With `PYTHONDONTWRITEBYTECODE=1`, `python -m pytest tests/test_finite_observation.py tests/test_local_stability.py -q -p no:cacheprovider` passed **39 tests in 46.54 seconds**; the repository-required `python -m pytest -q -p no:cacheprovider` passed **233 tests in 167.36 seconds**. Static checks found 104 unique LaTeX labels, no missing references or citations, balanced braces and environments, valid Python syntax, and no trailing whitespace in the three changed files. All pre-existing theorem statements, numbered displays, matrix blocks, table bodies and bibliography entries were preserved. Scoped `git diff --check` passed; Git additionally reported its existing LF-to-CRLF conversion warning. No TeX engine was available, so compilation and visual inspection were not performed. No existing script, certificate, test or finance file was edited, and no experiment, PDF, alternate manuscript or result artifact was generated.


## 2026-09-13: Central fixed-zero Hall-rank theorem and document integration

**Verdict: VERIFIED in the stated fixed-state rank and local truncated-response
scope.** The canonical manuscript now leads with generic rank
\(\min(D,\nu_{d,L})\) in the full shallow biased-tanh parameter space, an
explicit finite witness, and local recovery from one zero-initialized
degree-\(L\) truncated-response vector at the sharp width \(D=\nu_{d,L}\).
The preceding audit entries, counts, hashes, environments and the 8 September
actual-response addendum remain historical records; this addendum does not
reinterpret them as checks performed today. Current repository HEAD is
1280c84f27e3a02c5ed1a2c8dba73c8789eadf57.

### Authority, proof and preserved scope

The authoritative sources are [the Hall-rank proof note](intrinsic_hall_rank_theorem.md)
and [its exact finite certificate](intrinsic_hall_rank_certificate.json).
The latter's SHA-256 is
2bf2c8b57057619e98c36ab516d4535cd748aa12cdd3d9b3c5e2e026c53d6388.
The manuscript's self-contained proof follows the note: grading-derived
affine identities with the existing bracket sign, finite tanh jets,
graded ideal quotients including \(D<d\), inactive padding for \(D>\nu\),
and a nontrivial analytic minor in all weights and biases at state zero.

The main theorem has no computational dimension cap. The finite witness
uses actual parameters \(A_i=-\Lambda_{e_i}\), \(b_i=q e_i\), with
\(q=\operatorname{atanh}(1/2)\); no limiting parameter or numerical rank
establishes genericity. At width eight the complete matrix is
\[
 H_\theta=\tfrac12 I_8-\tfrac1{32}(E_{3,6}+E_{3,8})
\]
in Hall order \((1,2,12,112,122,1112,1122,1222)\), with one-based row and
column indices. The two nonlinear entries are preserved. The actual-jet
reproducer recomputes them rather than inserting this fixture.

For \(r\ne0\), the derivative of the finite truncated response is
\(H_\theta\operatorname{diag}(r^{\deg H_a})\). The selected projection has
a local analytic inverse when \(D\geq\nu\). Invariance of domain supplies
the matching scalar-output lower bound for an open Hall-coordinate
domain. Neighborhoods depend on parameters and scale. This is Object B;
it gives no exact full-CDE endpoint inverse, global inverse, trained
linear readout, conditioning guarantee, or genericity for singular
sampling families. The nonlinear action is applied to finite full-tensor
representatives, not assumed to factor through the nilpotent quotient.

The scalar homogeneous and rank-seven **function** certificates, normalized
ideal jets, eight-output scalar truncated inverse/stability, scalar
approximate recovery from eight actual CDE responses, and the larger-width
all-word independence theorem retain their separate proofs and constants.
The scalar exact sections, actual-response theorem and financial section
are textually preserved. Of 108 original displayed-math/table blocks,
106 are retained literally; the only replaced blocks are the notation
table and the explanatory dependency diagram. The existing word proof and
fixed-zero word corollary are preserved, with their architecture definition
moved earlier. No scalar or word certificate value is changed.

### Bounded primary-source comparison

The checked locators and versions used in the integrated discussion are:

- [Burde, author PDF dated 1 March 2011](https://homepage.univie.ac.at/dietrich.burde/papers/burde_24_pre_lie.pdf),
  Section 3.1, Proposition 3.3 and Corollaries 3.4–3.5, printed pages 14–16;
  journal metadata: *Central European Journal of Mathematics* 4(3):323–357
  (2006). The nonsingular-derivation affine/pre-Lie construction is classical.
- [Glückstad–Muça Cirone–Teichmann, arXiv:2502.03163v1](https://arxiv.org/html/2502.03163v1),
  5 February 2025, still the current arXiv version checked: Section 3.3.2,
  Remark 3.10, Theorem 4.1 and Section 5. Projected tree terms remain distinct
  from complete summed fields; no refutation of their complete reconstruction
  theorem or independently verified depth-two width comparison is asserted.
  Scale-derivative access in the reconstruction proof is identified explicitly.
- [Bailleul–Diehl, arXiv:1407.2768v3](https://arxiv.org/pdf/1407.2768v3),
  14 November 2014, Theorem 6, printed page 7. The reconstruction matrix
  includes the one-state case and degree-two brackets, for its small-time
  rough-flow observation model.
- [Biagini–Gonon–Walter, arXiv:2406.10214v2](https://arxiv.org/pdf/2406.10214v2),
  6 September 2024: Sampling Scheme 4, Assumption 6, Lemma 7 and
  Proposition 9 in Section 2.1 (PDF pages 5–6, 6, 6 and 9). The PDF numbering
  is used because the HTML numbering differs. Their structured discrete-time
  model includes an injective finite-datastream memory block and a terminal
  increment linear-readout approximation result; it is not described as
  approximation alone or equated with the full biased-tanh Hall-rank model.

The contribution language distinguishes the classical machinery from the
finite-jet tanh realization and the specific observation theorem. No worldwide
priority claim is added.

### Checks performed in this integration

From the repository root, both commands completed successfully without
writing certificates:

~~~text
python -B research_output/reproduce_intrinsic_hall_rank.py
python -B research_output/reproduce_certificate.py
~~~

The new reproducer passed all 17 persisted exact cases, including the six
full-width references \((d,L,\nu)=(2,2,3),(2,3,5),(2,4,8),(2,5,14),
(3,3,14),(3,4,32)\), quotient and padding cases, and edge cases.
The scalar reproducer confirmed determinants
\(-1/4,-1/2,-45/256,135/8192\) and product \(-6075/16777216\), and also
verified its generated note. That note required one added scope paragraph;
the identical paragraph was added to its existing prose template in
reproduce_certificate.py so the byte-for-byte comparison remains valid.
No computation, validation logic, or certificate data in that script changed.

Static checks found **113 unique manuscript labels**, no duplicate labels,
no unresolved references, **17 bibliography entries** with no missing
citations, and balanced braces and environments. Local Markdown links and
anchors and the displayed reproduction/test paths were checked against the
current repository. Scoped whitespace checks passed. Repository-wide
git diff --check still reports **804 pre-existing** trailing-whitespace
entries, confined to rsmc/__init__.py, rsmc/hedging.py, and
tests/test_hedging.py. Those files and AGENTS.md were not changed.
Content hashes confirm preservation of the existing exact JSON artifacts,
research implementations and tests, apart from the stated generated-prose
template. No experiment, new implementation, full test campaign, staging,
commit, push, alternate manuscript or certificate regeneration was performed.

The completed implementation stage's 38 focused, 135 affected-existing and
271 full-suite passing tests remain documented in the authoritative note;
they are **not new test runs for this prose integration**.

No pdflatex, latexmk, lualatex, xelatex or tectonic executable was
found in the Windows environment or WSL Ubuntu. Thus the canonical source
was **not compiled in this stage**, and PDF page layout, visual overflow,
and compiler diagnostics remain unverified. Static syntax checks do not
substitute for those checks. The manuscript records a twice-run build
command using paper/build when an engine is available. There is no
unresolved mathematical obstruction identified in this integration; the
remaining validation limitation is compilation and visual inspection.

## Final independent Hall-rank audit — 13 September 2026

**Verdict: VERIFIED within the stated scope; upgrade complete.** This entry
records new checks of the current repository, independently of the preceding
reviews. Their dates, counts, hashes and Stage 3 build limitation remain
historical records. No central mathematical or implementation defect was found.

The grading conjugation and cocycle give the affine homomorphism with the
repository bracket sign, hence evaluation \(cH\), not \(c^{\deg H}H\).
The finite-jet weight argument gives zero components above the Hall degree
and diagonal blocks \(I/2\), while permitting lower-degree nonlinear
components. The discarded subspace is a graded ideal, including partial
top grades and \(D<d\); padding and the \(d=1,L=1\) edge cases are valid.
No nonlinear quotient action or vanishing discarded Hall columns is inferred.
The selected minor is analytic and nontrivial at the fixed state zero in
the full weight-and-bias space. Its zero set need not equal the rank-deficient
locus. The derivative of the truncated polynomial response is
\(H_\theta\operatorname{diag}(r^{\deg H_a})\). Its local inverse, or projected
inverse, for each fixed \(r\ne0\) and the open-domain dimensional lower bound
have the stated hypotheses. Neither this proof nor the finite checks give
an exact full-CDE inverse, singular-law genericity, linear-readout theorem,
or parameter-uniform stability.

### Findings and corrections

- **Substantive wording/source scope, fixed:** the scalar actual-response
  estimate guarantees consistency when observation error and exact iteration
  residual are \(o(h^4)\); it does not prove pointwise necessity. The manuscript
  and technical note now restrict Biagini–Gonon–Walter Proposition 9 to
  continuous functions of preterminal history \(x_1,\ldots,x_{T-1}\).
  The word note's remaining unqualified depth-two comparison now distinguishes
  projected tree terms from complete summed fields and does not claim a
  refutation of the complete reconstruction theorem.
- **Cross-references and prose, fixed:** current overviews and the word/Hall
  notes locate the fixed-zero word corollary in the canonical manuscript,
  label `cor:zero`. The relocated finance paragraph says “in this paper.”
  The Bailleul–Diehl link in the Hall note is pinned to the checked version.
- **Layout, fixed:** `eq:dets` is split without changing values; certificate
  paths are breakable and listed separately. A further small overflow in the
  Hall-label paragraph was removed. Inline shell commands retain their spaces.
  All 120 pre-audit displayed-math/table blocks remain, with only `eq:dets`
  reformatted; the other 119 are literally preserved.

Primary PDFs were independently opened and checked: [Burde's author version
dated 1 March 2011](https://homepage.univie.ac.at/dietrich.burde/papers/burde_24_pre_lie.pdf),
Section 3.1, Proposition 3.3 and Corollaries 3.4–3.5, pages 14–16;
[Bailleul–Diehl v3](https://arxiv.org/pdf/1407.2768v3), Theorem 6, page 7;
[Glückstad–Muça Cirone–Teichmann v1](https://arxiv.org/pdf/2502.03163v1),
Lemma 3.4, Proposition 3.6, Section 3.3.2, Remark 3.10 and Theorem 4.1;
and [Biagini–Gonon–Walter v2](https://arxiv.org/pdf/2406.10214v2),
Sampling Scheme 4, Assumption 6, Lemma 7 and Proposition 9 (PDF page 9;
the preterminal restriction is also explicit on page 10). The classical
affine/pre-Lie attribution and bounded observation-model comparisons are
supported. This check does not establish originality or publication acceptance.

### Commands and preservation

The following commands were executed successfully in this audit:

~~~text
python -B research_output/reproduce_intrinsic_hall_rank.py
python -B research_output/reproduce_certificate.py
python -B research_output/reproduce_graded_reconstruction.py
python -B research_output/reproduce_finite_observation.py
python -B research_output/reproduce_local_stability.py
python -B research_output/reproduce_tensor_witness.py
python -B -m pytest -q -p no:cacheprovider
~~~

All **17 full persisted Hall matrices, selected minors and payloads** matched,
including all six requested full-width reference cases, quotients, padding
and edge cases. An independent direct symbolic Jacobian-bracket calculation
from the actual eight-dimensional tanh fields recovered
\(I_8/2-(E_{3,6}+E_{3,8})/32\), determinant \(1/256\), in the recorded Hall
order. A tampered payload with unchanged rank and determinant was rejected
without changing the certificate's bytes or modification time. Independent
tests cover bracket signs, the full small truncated-map Jacobian at scales
\(0,1,1/3,-2\), and the normalized tanh coefficient \(-17/315\).
The full suite passed **271 tests in 127.32 seconds**, using Python 3.13.3,
pytest 9.1.1 and SymPy 1.14.0. All five legacy reproducers passed, including
the scalar generated-note comparison. No certificate was regenerated.

The central Hall argument and width-eight computation were independently
rederived here. Legacy scalar homogeneous/cross-degree, eight-observation,
stability and word certificates were checked for preservation and reproduced;
the existing scalar actual-response estimate was checked for consistent
scope, without rerunning its numerical validation. No finance experiment ran.
Content hashes confirm that all implementation, test and certificate files,
AGENTS.md and unrelated work are unchanged from this audit's starting state.
Edits are confined to the canonical TeX, technical note, reconstruction
summary, word theorem note, Hall theorem note and this audit, plus normal
LaTeX outputs in `paper/build`. No staging, commit or push was performed.

Static checks found 113 unique labels and 17 bibliography entries, with no
missing references or citations; local document links resolve. Scoped
whitespace checks passed. The required global `git diff --check` was run
and still fails on **804 pre-existing trailing-whitespace entries** in
`rsmc/__init__.py` (53), `rsmc/hedging.py` (410), and
`tests/test_hedging.py` (341). Those files were left untouched.

### Canonical PDF verification

No installed Windows or WSL engine was initially available. A portable
Tectonic 0.17.0 executable was obtained from its official release, checked
against the release SHA-256, and kept outside the repository. The final
PowerShell build command used the available font configuration:

~~~powershell
$env:FONTCONFIG_FILE = 'C:/Users/petar/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/etc/fonts/fonts.conf'
& (Join-Path $env:TEMP 'codex-hall-audit-tectonic-0.17.0/tectonic.exe') --keep-logs --keep-intermediates --outdir paper/build paper/randomized_signature_identifiability.tex
~~~

Tectonic reran TeX until references settled; the build was repeated after
layout fixes. The final local build (`paper/build/randomized_signature_identifiability.pdf`,
regenerable from the tracked TeX source using the command above)
has **37 pages**. All pages were rendered and visually inspected, including
the main proof, fraction-heavy appendices, continuation pages and references;
the final changed page was inspected again after rebuilding. There are no
overfull boxes, unresolved references, missing citations or material visible
clipping. The log retains an innocuous UTF-8-engine `inputenc` warning and
one underfull bibliography paragraph (badness 1048), which is readable on
inspection. These are not mathematical or PDF-verification blockers.

## Literature revision — 13 September 2026

The literature findings in the supplied rigorous_paper_audit.pdf were
checked against the current canonical source and the primary texts below.
The earlier audit entries remain historical records. Their literature
comparison omitted three relevant antecedents; mathematical verification
does not certify novelty or publication significance.

- [Monzio Compagnoni et al., arXiv:2201.00384v3](https://arxiv.org/pdf/2201.00384v3):
  Definition II.5 and Theorem IV.1 were checked. The same shallow biased
  architecture and its asserted generic Hörmander argument are now credited,
  without equating unrestricted spanning with maximal Hall rank at a
  prescribed cutoff or adopting the broad analytic-activation claim.
- [Cuchiero–Larsson–Teichmann, arXiv:1908.07838v2](https://arxiv.org/pdf/1908.07838v2):
  Theorem 5.1, Corollary 5.4, Example 5.5 and Lemma 6.1 were checked.
  The manuscript credits witness-and-analytic-minor methodology and
  distinguishes the activation family with polynomial specializations
  from fixed tanh. Its interpolation summary states the dimension,
  generic-data and trained-control/readout qualifications.
- [Grong–Vega-Molino, arXiv:2403.08090v3](https://arxiv.org/html/2403.08090v3):
  Remark 3.2 explicitly contains the scalar constant/tanh pair and
  polynomial/Vandermonde argument. Attribution is added beside the scalar
  witness and in related work. The prescribed-degree certificates and
  state-scale inverse remain separate results.

The abstract, introduction, contribution summary, related work, conclusion
and affected technical/theorem summaries now focus on the explicit finite
tanh realization, exact prescribed-degree minors and local truncated
inverse. No claim rests on absence of a located precedent or a defect in
another paper. No new theorem, architecture, application or research
direction was introduced.

The portfolio citation now gives the publisher-verified Quantitative
Finance 25(2):197–216 (2025), DOI 10.1080/14697688.2025.2458613.
The derivation note's Friz–Victoir draft locators are corrected to printed
pages 142–143 and 144–145 for Theorems 7.28 and 7.30. The conditional
remark after the control-projection proposition now explicitly requires
conditional centering and finite conditional moments, with conditional
means/moments used in the formula. The numbered proposition is unchanged.

All three requested full texts and relevant locators were accessible and
checked. This was a bounded literature revision, not a fresh verification
of every historical source or experiment. Source-specific read-only
reviewers also checked the revised comparisons.

Validation: the portable Tectonic build command recorded above was run
after the edits and again after the final qualifications; TeX reran until
references settled. The canonical build has 38 pages, with no overfull
boxes or unresolved references/citations. The affected front matter,
scalar attribution, conditional remark, related work, conclusion,
appendices and references were visually inspected. Only the previously
recorded inputenc and readable underfull-bibliography warnings remain.
Static checks found 113 unique labels and 20 bibliography entries.
All 120 displayed-math/table blocks and all 52 theorem/proof blocks are
unchanged from this revision's starting source. Relative links and scoped
whitespace checks passed. The global whitespace gate still reports the
same 804 pre-existing entries in the three untouched finance files.
Implementations, tests, certificate data and AGENTS.md were preserved;
no tests or experiments were rerun for these document-only changes.

## Mathematical scope and evidence revision — 13 September 2026

This review independently checked Sections 6, 7.3–7.4, 8.3 and 9 of the
supplied rigorous_paper_audit.pdf against the latest canonical manuscript
and available repository material. The preceding runs, counts, timings
and review conclusions remain historical records.

**Conditional projection.** The previous revision already supplies the
necessary conditional centering, so that remark was retained. Apply the
sample-split unbiasedness argument coordinatewise to the predictable
terminal controls: it gives \(\mathbb E[Q\mid\mathcal T]=0\).
The projection proposition's \(Q,G_N\in L^2\) assumptions give finite
conditional second moments almost surely. Evaluation/training independence
also gives \(\mathbb E[G_N\mid\mathcal T]=p_N\). This does not require
\(Q\) itself to be independent of training. Without conditional centering,
the corresponding identity would have to use conditional covariances.
The audit's Rademacher counterexample to the old unconditional-only reading
is valid, but does not contradict the current corrected remark.

**Affine scaling.** Set \(S_\epsilon(y)=\epsilon y\), \(c=1/2\), and
\(V_i^\epsilon(x)=\tanh(\epsilon ce_i-\Lambda_{e_i}x)\). These are actual
admissible fields at every finite nonzero \(\epsilon\). The pullbacks are
\[
U_i^\epsilon=S_\epsilon^*V_i^\epsilon
=\epsilon^{-1}\tanh(\epsilon W_{e_i}^c).
\]
Writing \(\tanh t=t+t^3g(t)\), with \(g\) analytic near zero, shows
\(U_i^\epsilon=W_{e_i}^c+O(\epsilon^2)\) in every fixed finite derivative
order on compact neighborhoods. This uses a local finite-jet argument,
not an infinite-radius expansion. The chain rule with the repository
bracket sign gives, inductively for every Hall element,
\[
\rho_{V^\epsilon}(H)(\epsilon y)
=\epsilon\,\rho_{U^\epsilon}(H)(y).
\]
Thus the factor is one \(\epsilon\), independent of Hall degree.
Brackets through length \(L\) use jets through order \(L-1\); their
pullback evaluation matrix tends to \(cI_\nu\). Consequently the original
matrix has determinant \(\epsilon^\nu(c^\nu+o(1))\), nonzero for small
finite positive \(\epsilon\). The existing graded quotient gives the
analogous \(D\)-minor below full width, and padding preserves the
\(\nu\)-minor above it. The manuscript adds only a short explanatory
paragraph relating this qualitative observation to the classical action.
It is not attributed to a prior publication and does not replace the
explicit \(q\)-witness, its exact graded blocks or selected determinants.

**Recovery and quantitative scope.** The current claims already separate
ideal jets, the central local truncated inverse in the full biased family,
and the eight-response scalar approximation. No unsupported endpoint,
sampling-law, linear-readout or finance-performance extension was found.
Independent rational arithmetic confirms \(A=3402\),
\(C=409087045009408/21220001406063\), and
\(\kappa=1748210800969/4105988734976\). At \(M=1,h=1/2048\), the noiseless
upper-bound term is approximately \(32.02393349\), and the response-error
coefficient is \(3.39148677\times10^{14}\). The manuscript now uses these
figures to explain the certificate's limits: they neither measure actual
error nor prove numerical failure, and asymptotic consistency alone does
not certify useful finite-precision accuracy. No scalar constant is
transferred to the Hall-width construction. The supplied audit's 70/100-digit
path calculations and the repository's recorded 80/120-digit validation
are distinct evidence sets; neither was rerun in this review.

**Empirical evidence.** All eleven displayed RMSE values match the local
June-note extraction. Exact arithmetic on the printed decimals gives gap
ratios \(183595/199317\) and \(328414032/368020345\), hence
\(92.11206269\%\) and \(89.23800993\%\). This verifies the reported rounding,
not the experiment. Source inspection confirms predictable prefixes,
training-standardization reversal and intercept omission in the current
control implementation. Code, tests, certificates and the June-note
extraction exist locally; the extraction is untracked. No run-specific
data or fitted-model artifact establishing the displayed hedging values
was located. The existing demo log concerns a different pricing run.
The manuscript now states available evidence and the reproduction limit
explicitly. Attachment-only omissions do not establish that historical
execution claims are false, and present arithmetic does not authenticate
their dates or counts.

**Checks in this review.** In-memory Python Fraction calculations checked
the constants and table ratios above. No test suite, certificate
regeneration or experiment ran. The portable Tectonic command recorded
above compiled the canonical source after edits and reran until references
settled. The final build has 39 pages; affected pages and continuations
were rendered and inspected, with no overfull boxes or unresolved
references/citations. The existing readable underfull bibliography and
inputenc warnings remain. All 120 displayed-math/table blocks and all
52 theorem/proof blocks are preserved. Scoped whitespace and reference
checks passed; the global whitespace gate retains the same 804 unrelated
entries. Only the canonical TeX and this audit record were edited, besides
normal build outputs. The other notes already agree with these corrections.

## Final consistency check — 13 September 2026

The latest canonical manuscript, authoritative Hall note, current supporting
notes, supplied rigorous audit and available verification material were
compared. The main theorem's assumptions and proof agree: fixed-zero Hall
rank in the full biased parameter family, with local inversion only for the
truncated polynomial observation. Conditional centering, the affine scaling
argument, the separate scalar actual-response bound and the revised
literature comparisons remain correct. Independent review also derived the
two width-eight nonlinear entries directly from the tanh components.
No theorem, equation, matrix entry or certificate required correction.
One stale directional reference in the relocated finance section was changed
from "studied below" to "studied in this paper."

Checks executed in this final review, distinct from the historical runs:

- `python -B research_output/reproduce_intrinsic_hall_rank.py`: all 17
  exact cases passed full deterministic payload comparison.
- `python -B research_output/reproduce_local_stability.py`: the design,
  Jacobian, inverse, variation bound and common-scale identities passed.
- The supplied `check_intrinsic_width.py` was read from
  `randomized_signature_novelty_verification.zip` and executed in memory;
  all six outputs matched its supplied expected JSON. Its unrelated
  financial-extension script was not run.
- `exact_setup()` from `validate_actual_response_recovery.py` passed its
  existing exact scaling, Hall/tensor and word-polynomial checks. Fresh
  Fraction calculations confirmed the revised bound figures and both
  finance percentages. No high-precision path run, financial experiment
  or full test suite was repeated.
- All eleven matrix blocks in the available older layout-bundle source
  are retained. That source predates the Hall upgrade. The exact Stage-3
  source snapshot and separate reproduction bundle named in the rigorous
  audit were not located, so a complete textual comparison with that
  snapshot is not claimed.
- Static checks found 113 unique labels, 20 bibliography entries, no
  unresolved references/citations, and 53 resolving relative file links
  across the checked notes. Scoped `git diff --check` passed.

The portable Tectonic command recorded above compiled the canonical source
and settled references, including after the final wording correction.
All 39 rendered pages were visually inspected. The changed page was
inspected again after rebuilding; subsequent rendered pages were unchanged.
There are no overfull boxes or material layout defects. The existing
inputenc warning and readable underfull bibliography paragraph remain.
Only the canonical TeX, this appended record and normal build outputs
changed; implementations, tests, certificates and earlier records were
preserved.

**Status:** the scoped consistency check and exact reproductions pass.
The revised primary-source comparisons were checked, but this does not
certify originality; unchanged Reutenauer book locators were not newly
authenticated in this review. Finance arithmetic is verified, while
run-specific data/model provenance and experimental reproduction remain
unverified. Missing audit attachments do not disprove historical execution
records. No unresolved mathematical or layout defect was found in the
revisions checked here.
