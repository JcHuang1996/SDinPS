# Paper-to-code design note: CGLP-based optimality cuts for the integer L-shaped method

## Scope

The repository adapts separation to its outer iteration loop, without a solver callback: the CGLP is
solved after the exact subproblem of a binary master solution, inside the decomposition iteration. With an
inexact recourse solve, cached values are conservative solver lower bounds rather than exact values, so
cuts stay valid but may not be tight at the true recourse value. Section 11 lists the paper's requirements
that the implementation keeps (cumulative points, tight current point, incremental prefix updates).

This note extracts the part of *Improving the integer L-shaped method* needed to implement **Algorithm 4** (optimality cut function with CGLP-based optimality cuts) and **Algorithm 5** (updating CGLP) for a known two-stage MILP.

The focus is **not** the outer branch-and-cut / callback loop itself. The focus is:

- what mathematical objects the CGLP cut needs,
- what state must be maintained across iterations,
- what LP must be built and solved,
- what is updated when a new exact second-stage evaluation is obtained,
- and how the resulting cut is added back to the master problem.

---

## 1. Problem setting from the paper

The paper considers problems of the form

$$
\begin{aligned}
\min_{x,z,\theta}\quad & c x + d z + \theta \\
\text{s.t.}\quad & A x + C z \le b, \\
& Q(x) - \theta \le 0, \\
& x \in \{0,1\}^n, \\
& z \ge 0,\; z \in Z,
\end{aligned}
$$

where:

- $x$ is the **binary first-stage vector**,
- $z$ is any other first-stage variable block already present in the master problem,
- $Q(x)$ is the **exact second-stage value function**, assumed computable for binary $x$,
- $\theta$ is the master variable that underestimates $Q(x)$.

For two-stage stochastic MILP,

$$
Q(x) = \mathbb{E}_\xi[Q_\xi(x)],
$$

with $Q_\xi(x)$ obtained by solving the exact second-stage subproblem for scenario $\xi$.

The paper also uses the continuous relaxation value function $Q_{LP}(x)$ to generate subgradient cuts.

---

## 2. What this component is supposed to do

When the master solver produces a **candidate integer solution** $(x^*, z^*, \theta^*)$:

1. solve the LP relaxation of the second stage at $x^*$ and add a subgradient cut;
2. solve the exact second-stage problem at $x^*$ and obtain $Q(x^*)$;
3. update the internal CGLP state so that the set of all **exactly evaluated** first-stage points is reflected in the cut-generating LP;
4. if $\theta^* < Q(x^*)$, solve the CGLP and generate a cut
   $$
   \alpha x + \theta \ge -\eta,
   $$
   then reject the candidate;
5. otherwise accept the candidate.

So the CGLP routine is **not** the entire algorithm. It is a **cut separation subroutine** that becomes active after an exact second-stage evaluation shows that the candidate is infeasible to the true epigraph $\theta \ge Q(x)$.

---

## 3. Original-model objects vs CGLP-generated objects

### 3.1 Objects coming from the original master problem

These are already known from the two-stage MILP implementation:

- $A, C, b$: first-stage feasibility system $A x + C z \le b$.
- $x \in \{0,1\}^n$: binary first-stage variables.
- $z \ge 0, z \in Z$: other first-stage variables.
- $Q(x)$: exact second-stage evaluator.
- $Q_{LP}(x)$: LP-relaxation evaluator of the second-stage value function.
- affine lower-bound cuts already present in the master problem, collected as
  $$
  \Pi x - \mathbf{1}\theta \le \pi_0.
  $$

Interpretation of $\Pi x - \mathbf{1}\theta \le \pi_0$:

- each row is one currently valid lower bound on $Q(x)$,
- this collection must include at least the trivial bound $\theta \ge L$,
- and may also include Benders / subgradient cuts already present in the master.

A practical row-wise interpretation is:

$$
a_r x - \theta \le b_r \quad \Longleftrightarrow \quad \theta \ge a_r x - b_r.
$$

Then $\Pi$ stacks the row vectors $a_r$ and $\pi_0$ stacks the corresponding right-hand sides $b_r$.

### 3.2 Objects created/maintained by the CGLP algorithm

These are the additional data structures needed specifically for the CGLP-based cut:

- $V = \{x^1, \dots, x^t\}$: set of first-stage solutions for which **exact** values $Q(x^s)$ are known.
- $V^i$: projection of $V$ onto the first $i$ components.
- $\hat V^1 := \{0,1\} \setminus V^1$.
- for $i \ge 2$,
  $$
  \hat V^i := [V^{i-1} \times \{0,1\}] \setminus V^i.
  $$
- $W_i := \hat V^i \times \{0\}^{n-i} \subseteq \{0,1\}^n$.

If
$$
\hat V^i = \{v_1^i, \dots, v_{k_i}^i\},
$$
then
$$
W_i = \{w_1^i, \dots, w_{k_i}^i\}.
$$

These $w_l^i$ are the binary vectors that appear in the CGLP row family
$$
w_l^i \sigma + \rho_i \ge 0.
$$

---

## 4. Geometry behind the cut (what the paper is modeling)

The target epigraph is

$$
S := \{(x,\theta) \in X \times \mathbb{R} : \theta \ge Q(x)\},
$$

where $X$ is the projection of the master feasible set onto the $x$-space.

Given only the exact values $Q(x)$ for points in $V$, the paper builds a relaxation based on

$$
S(V) := \bigcup_{x\in V} \{(x,\theta): \theta \ge Q(x)\}
\;\cup\;
(\{0,1\}^n \setminus V) \times \{\theta : \theta \ge L\}.
$$

The idea is:

- for points already seen exactly, use their exact epigraph rays;
- for unseen binary points, only use the crude lower bound $L$.

The CGLP generates a valid inequality for $\operatorname{conv}(S(V))$, hence also for the true epigraph.

This is why the cut becomes stronger as more exact evaluations are accumulated in $V$.

---

## 5. Extended formulation ingredients used by the CGLP

The paper decomposes $\operatorname{conv}(S(V))$ into two pieces.

### 5.1 Exact-value piece

For $V = \{x^1,\dots,x^t\}$,

$$
PQ(V) := \operatorname{conv}\left(\bigcup_{x\in V}\{(x,\theta): \theta \ge Q(x)\}\right).
$$

An extended formulation is:

$$
\begin{aligned}
-x^Q + \sum_{s=1}^t \phi_s x^s &= 0, \\
-\theta^Q + \sum_{s=1}^t \phi_s Q(x^s) &\le 0, \\
\sum_{s=1}^t \phi_s &= 1, \\
\phi &\ge 0.
\end{aligned}
$$

### 5.2 Unseen-binary-point piece

$$
PL(V) := \operatorname{conv}(\{0,1\}^n \setminus V) \times \{\theta : \theta \ge L\}.
$$

The paper derives an extended formulation for $\operatorname{conv}(\{0,1\}^n \setminus V)$ using the projected sets $V^i$, $\hat V^i$, and $W_i$.

For implementation purposes, the important fact is:

- the formulation width grows with the stored vectors $w_l^i$,
- and when one new point is added to $V$, only **marginal row updates** are needed.

That is exactly what Algorithm 5 exploits.

---

## 6. The CGLP that must be solved

After the paper normalizes the Farkas certificate with $\beta = 1$, the cut-generating LP becomes the following **feasibility LP**.

Let the newly exact-evaluated violating point be renamed $x^t := x^*$. Then solve for

- $\alpha \in \mathbb{R}^n$,
- $\eta \in \mathbb{R}$,
- $\sigma \in \mathbb{R}^n$,
- $\rho_i \in \mathbb{R}$ for $i=1,\dots,n$,
- $\phi_i \ge 0$ for the chain inequalities (paper notation uses $\varphi$; below I keep $\varphi$),
- $\nu \ge 0$,
- $\psi \ge 0$,

such that

$$
\begin{aligned}
\alpha - \sigma + A^\top \nu + \Pi^\top \psi &= 0, \\
\mathbf{1}\psi &= 1, \\
-\rho_n + \eta - b\nu - \pi_0\psi &\ge 0, \\
C^\top \nu &\ge 0, \\
\sigma_i + \varphi_i &\ge 0 && 2 \le i \le n, \\
-\rho_i + \rho_{i+1} + \varphi_{i+1} &\ge 0 && 1 \le i \le n-1, \\
w_l^i \sigma + \rho_i &\ge 0 && 1 \le i \le n,\; 1 \le l \le k_i, \\
x^s \alpha + \eta &\ge -Q(x^s) && 1 \le s < t, \\
x^t \alpha + \eta &= -Q(x^t), \\
\varphi &\ge 0,\; \nu \ge 0,\; \psi \ge 0.
\end{aligned}
$$

Any feasible solution yields the optimality cut

$$
\alpha x + \theta \ge -\eta.
$$

### Important implementation fact

This LP has **no real optimization objective** in the paper's final form. It is a **feasibility problem**. In practice, you may solve it as:

$$
\min 0
$$

subject to the constraints above.

---

## 7. Meaning of each CGLP block

This is the most important translation from paper notation to implementation logic.

### 7.1 Cut coefficients to be returned

- $\alpha$: coefficient vector on the binary first-stage variables $x$.
- $\eta$: cut intercept.

Returned master cut:

$$
\alpha x + \theta \ge -\eta.
$$

### 7.2 Constraints tied to current master feasibility description

$$
\alpha - \sigma + A^\top \nu + \Pi^\top \psi = 0,
$$
$$
\mathbf{1}\psi = 1,
$$
$$
-\rho_n + \eta - b\nu - \pi_0\psi \ge 0,
$$
$$
C^\top \nu \ge 0.
$$

Interpretation:

- these rows connect the cut coefficients to the first-stage master description;
- $\nu$ corresponds to the original first-stage constraints $Ax + Cz \le b$;
- $\psi$ corresponds to the currently available lower-bound cuts $\Pi x - \mathbf{1}\theta \le \pi_0$;
- the normalization $\mathbf{1}\psi = 1$ is the paper's replacement for the original $\beta = 1$ normalization.

### 7.3 Constraints encoding the convex hull of unseen binary points

$$
\sigma_i + \varphi_i \ge 0 \quad (2 \le i \le n),
$$
$$
-\rho_i + \rho_{i+1} + \varphi_{i+1} \ge 0 \quad (1 \le i \le n-1),
$$
$$
w_l^i \sigma + \rho_i \ge 0 \quad (1 \le i \le n,\; 1 \le l \le k_i).
$$

Interpretation:

- this block is the dual-side representation of the forbidden-vertices formulation for $\operatorname{conv}(\{0,1\}^n \setminus V)$;
- the only part of this block that changes with newly seen points is the family indexed by the vectors $w_l^i$.

### 7.4 Constraints enforcing validity on all exact points in $V$

For old exact points:

$$
x^s \alpha + \eta \ge -Q(x^s), \qquad s=1,\dots,t-1,
$$

and for the newest point:

$$
x^t \alpha + \eta = -Q(x^t).
$$

Interpretation:

- the cut must underestimate $Q(x)$ at all previously exact-evaluated points;
- the cut must be **tight at the current point** $x^t = x^*$.

This tightness is the finite-termination condition carried over from the paper.

---

## 8. Required persistent state across iterations

For a paper-faithful implementation of Algorithm 4 + Algorithm 5, the following objects must persist across callback calls / incumbent checks.

### 8.1 Core persistent state

1. **Exact-value cache**
   - $V = \{x^1,\dots,x^t\}$
   - exact values $Q(x^s)$ for each $x^s \in V$

2. **Projection caches**
   - $V^i$ for every $i=1,\dots,n$
   - $\hat V^i$ for every $i=1,\dots,n$

3. **Current CGLP row set based on $W_i$**
   - equivalently, the list of currently active vectors $w_l^i$ for each $i$
   - or the actual LP row handles if your solver interface supports incremental add/remove

4. **Current exact-point rows inside the CGLP**
   - one inequality row for each older exact point:
     $$x^s\alpha + \eta \ge -Q(x^s)$$
   - one equality row for the newest exact point:
     $$x^t\alpha + \eta = -Q(x^t)$$

5. **Current lower-bound system used inside CGLP**
   - the matrix/vector pair $(\Pi, \pi_0)$
   - this should stay synchronized with the lower-bound cuts currently regarded as part of the master approximation

### 8.2 What does *not* need recomputation from scratch each time

The paper's update logic is designed so that you do **not** rebuild the forbidden-vertices block from scratch after every new point.

Instead, for each coordinate index $i$, adding a new exact point changes $\hat V^i$ by **at most one insertion or one deletion**, hence at most one corresponding row

$$
w\sigma + \rho_i \ge 0
$$

is added or removed for that $i$.

---

## 9. Algorithm 4 as an implementation procedure

Below is the paper's logic rewritten as a code-design procedure.

### Inputs at each incumbent/candidate callback

- candidate solution $(x^*, z^*, \theta^*)$ from the master solver,
- second-stage LP-relaxation evaluator $Q_{LP}(x)$,
- exact second-stage evaluator $Q(x)$,
- persistent state described above.

### Output

- `accept = True` if the candidate is feasible to the true problem,
- `accept = False` if a cut is generated and the candidate must be rejected.

### Procedure

1. **Duplicate exact-point check**
   - If $x^* \in V$, return `True` immediately.
   - The paper's logic is: if $x^*$ has been evaluated exactly before and the point reappears, the current master already guarantees $\theta^* \ge Q(x^*)$.

2. **Compute LP-relaxation value and add subgradient cut**
   - Solve the second-stage LP relaxation at $x^*$ to obtain $Q_{LP}(x^*)$ and a subgradient.
   - Add the subgradient cut to the master.

3. **Compute exact second-stage value**
   - Solve the exact second-stage problem at $x^*$ to obtain $Q(x^*)$.

4. **Update the CGLP state**
   - Use Algorithm 5 to update $V^i$, $\hat V^i$, the active $w$-rows, and the exact-point rows in the CGLP.
   - Then add $x^*$ to $V$ as a newly exact-evaluated point.

5. **Violation test**
   - If $\theta^* < Q(x^*)$:
     - solve the CGLP feasibility LP,
     - extract $(\alpha, \eta)$,
     - add the cut $\alpha x + \theta \ge -\eta$ to the master,
     - return `False`.

6. **Otherwise**
   - return `True`.

### What is solved when

For one candidate point under Algorithm 4:

- **always** solve $Q_{LP}(x^*)$,
- **always** solve exact $Q(x^*)$,
- **always** update the maintained CGLP structures after the exact evaluation,
- **only if** $\theta^* < Q(x^*)$ solve the CGLP itself to generate a cut.

---

## 10. Algorithm 5 as a state-update procedure

Suppose the new exact point is

$$
x^{t+1} = (x_1,\dots,x_n).
$$

For each prefix length $i = 1,\dots,n$, define:

- prefix of the new point:
  $$
  \bar x^i := (x_1,\dots,x_{i-1},x_i),
  $$
- same prefix with last bit flipped:
  $$
  \hat x^i := (x_1,\dots,x_{i-1},1-x_i).
  $$

The paper shows:

$$
\hat V_{t+1}^i = (\hat V_t^i \setminus \{\bar x^i\}) \cup (\{\hat x^i\} \setminus V_t^i).
$$

This yields the exact update logic.

### Per-coordinate update rule

For each $i=1,\dots,n$:

1. If $\hat x^i \notin \hat V^i$ and $\hat x^i \notin V^i$:
   - create
     $$
     w = \hat x^i \times \{0\}^{n-i},
     $$
   - add row
     $$
     w\sigma + \rho_i \ge 0
     $$
     to the CGLP,
   - update
     $$
     \hat V^i \leftarrow \hat V^i \cup \{\hat x^i\}.
     $$

2. If $\bar x^i \in \hat V^i$:
   - create
     $$
     w = \bar x^i \times \{0\}^{n-i},
     $$
   - remove row
     $$
     w\sigma + \rho_i \ge 0
     $$
     from the CGLP,
   - update
     $$
     \hat V^i \leftarrow \hat V^i \setminus \{\bar x^i\}.
     $$

3. Update the prefix set:
   $$
   V^i \leftarrow V^i \cup \{\bar x^i\}.
   $$

### Update of exact-point rows

After the $i$-loop finishes:

1. The previous current-point equality becomes an old-point inequality:
   $$
   x^t \alpha + \eta \ge -Q(x^t).
   $$

2. The new point becomes the tight equality row:
   $$
   x^{t+1} \alpha + \eta = -Q(x^{t+1}).
   $$

### Key computational consequence

The paper proves that adding one new exact point changes the $W_i$ block by **at most one row per coordinate**.
Therefore, a single update requires at most:

- up to $n$ row additions/removals in the forbidden-vertices block,
- one old equality converted to inequality,
- one new equality inserted.

So the intended implementation is **incremental**, not full rebuild.

---

## 11. Non-negotiable paper-faithful requirements

To stay faithful to Algorithm 4 + Algorithm 5 from the paper, the implementation must satisfy all of the following.

1. **Exact values are cumulative.**
   - The method relies on the growing set $V$ of exact first-stage evaluations.

2. **The newest exact point must be tight in the CGLP.**
   - Keep the equality row
     $$x^t\alpha + \eta = -Q(x^t).$$

3. **Older exact points must remain valid underestimating anchors.**
   - Keep the inequality rows
     $$x^s\alpha + \eta \ge -Q(x^s).$$

4. **The forbidden-vertices block must be updated incrementally via prefix sets.**
   - Do not treat $W_i$ as a black box that is fully rebuilt every time unless you intentionally choose a slower implementation.

5. **The CGLP must see the master's current lower-bound information.**
   - The block $(\Pi,\pi_0)$ is part of the paper's formulation.

6. **The generated cut is in the $(x,\theta)$-space only.**
   - The cut returned by Proposition 4 is
     $$\alpha x + \theta \ge -\eta,$$
     not a cut in $(x,z,\theta)$.

---

## Source note

This design note is extracted from:

- Proposition 4,
- Section 3.1 “Construction of CGLP”,
- Section 3.2 “Implementation”,
- Algorithm 4,
- Algorithm 5,

of *Improving the integer L-shaped method* by Angulo, Ahmed, and Dey (INFORMS Journal on Computing 28(3),
483–499, 2016; https://doi.org/10.1287/ijoc.2016.0695).
