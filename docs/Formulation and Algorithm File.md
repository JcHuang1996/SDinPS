## Notations

### Set

$S$: the set of all scenarios.

$J$: the set of all nodes.

$L$: the set of all lines.

$T$: the set of all time slots.

$M$: the set of all DG types.

### Parameters / Data

$\overline{N^{G}}$: the upperbound of number of generators.

$P_{s}$: the probability of scenario $s$, $\forall s\in S$.

$C^{Gf}_{j}$: the fixed cost of installing DG at node $j$, $\forall j \in J$.

$R_{m}$: the rated power of DG type $m$, $\forall m \in M$.

$C^{Gv}_{m}$: the unit investment cost of DG type $m$ (cost per MW), $\forall m \in M$.

$E_{m}$: the extra adjustment cost of DG type $m$, $\forall m \in M$.

$C^{Gu}_j$ : the unit cost of generating power at node $j$, $\forall j \in J$.

$A^{G}_{j}$: the coefficient for DG reactive power upperbound at node $j$, $\forall j\in J$.

$C^{H}_{ij}$: the fixed cost of hardening the line $(i,j)$, $\forall (i,j) \in L$.

$R_{ij}, X_{ij}$: resistance and reactance of line $(i,j)$.

$H^{h}_{ijts}$: the healthy status (with harden) of line $(i,j)$ at time $t$ in scenario $s$, $\forall (i,j)\in L, t\in T, s\in S$.

$H^{nh}_{ijts}$: the healthy status (without harden) of line $(i,j)$ at time $t$ in scenario $s$, $\forall (i,j)\in L, t\in T, s\in S$.

$\underline{V}, \overline{V}$: the lowerbound / upperbound of the bus voltage.

$V_{0}$: the slack-bus voltage of the system.

$\overline{S_{ij}}$: thermal limit of line $(i,j)$, $\forall (i,j)\in L$.

$\Theta_{j}$: all child neighbors of node $j$, $\forall j \in J$.

$\Pi_{j}$: all parent neighbors of node $j$, $\forall j \in J$.

> Note: just using one 'all neighbors of node $j$' is sufficient, as it is formulated as an undirected graph.
>
> We just divide the set into 'child' and 'parent' for the coding convenience:
>
> Given two nodes $i$ and $j$, implementing $(i, j) = (j,i)$ is more complex, while only defining $(i,j)$ and securing that $i$ only in either $\Theta_{j}$ or $\Pi_{j}$ is easier.

$D^{p}_{jts}, D^{q}_{jts}$: active / reactive power demand at node $j$ at time $t$ in scenario $s$, $\forall j\in J, t\in T, s\in S$.

$C^{L}$: the cost coefficient of load shedding.

### Variables

###### Main

> Note:
>
> WLOG, we assume that all indexes are properly refined.
>
> e.g. we do not need to check if line $(i,j)$ is possible to be broken when creating $x^{L}_{ij}$. Such variable refining process would be done in code.

$\theta_{s}$: Continuous, non-negative. The value of objective function of sub problem in scenario $s$, $\forall s\in S$.

$x^{G}_j$: Binary. Indicating whether DG is installed at node $j$ (1) or not (0), $\forall j \in J$.

$x^{GT}_{jm}$: Binary. Indicating whether DG type $m$ is installed at node $j$ (1) or not (0), $\forall j\in J, m\in M$.

$x^{L}_{ij}$: Binary. Indicating whether the line $(i,j)$ is hardened (1) or not (0), $\forall (i,j) \in L$.

###### Sub

> Note:
>
> In the implemented sub problem, local copies of the main-stage decision variables are introduced and fixed to the current main-stage solution. For simplicity, the same notations are reused below.

$p^{Grt}_{j}$: Continuous, non-negative. The rated power of DG at node $j$, $\forall j\in J$.

$y^{C}_{ijts}$: Binary. Indicating the connected (1) or disconnected (0) state of line $(i,j)$ at time $t$ in scenario $s$, $\forall (i,j)\in L, t\in T, s\in S$.

$y^{L}_{jts}$: Continuous, $0 \le y^{L}_{jts} \le 1$. The ratio of load to be shed at node $j$ at time $t$ in scenario $s$, $\forall j\in J, t\in T, s\in S$.

$v_{jts}$: Continuous, non-negative. Voltage of bus $j$ at time $t$ in scenario $s$, $\forall j\in J, t\in T, s\in S$.

$p^{G}_{jts}$: Continuous, non-negative. Active power output of DG at node $j$ at time $t$ in scenario $s$, $\forall j\in J, t\in T, s\in S$.

$q^{G}_{jts}$: Continuous, free. Reactive power output of DG at node $j$ at time $t$ in scenario $s$, $\forall j\in J, t\in T, s\in S$.

$p_{ijts}, q_{ijts}$: Continuous, free. Active / reactive power flow of line $(i,j)$ at time $t$ in scenario $s$, $\forall (i,j)\in L, t\in T, s\in S$.

> Note: following variables are only for topology.

$u^{L}_{ijts}$: Continuous, free. Virtual flow of line $(i,j)$ at time $t$ in scenario $s$, $\forall (i,j)\in L, t\in T, s\in S$.

$u^{S}_{jts}$: Binary. Indicating whether node $j$ is a virtual source (1) or not (0) at time $t$ in scenario $s$, $\forall j\in J, t\in T, s\in S$.

$u^{I}_{jts}$: Continuous, non-negative. Virtual power injected from the virtual source at node $j$ at time $t$ in scenario $s$, $\forall j\in J, t\in T, s\in S$.

## Main Problem Model

### Variables

$\theta_{s}$: Continuous, non-negative. The value of objective function of sub problem in scenario $s$, $\forall s\in S$.

$x^{G}_j$: Binary. Indicating whether DG is installed at node $j$ (1) or not (0), $\forall j \in J$.

$x^{GT}_{jm}$: Binary. Indicating whether DG type $m$ is installed at node $j$ (1) or not (0), $\forall j\in J, m\in M$.

$x^{L}_{ij}$: Binary. Indicating whether the line $(i,j)$ is hardened (1) or not (0), $\forall (i,j) \in L$.

### Objective function

$$
\begin{align}
\min \quad z &=
\sum_{j\in J} C^{Gf}_{j} x^{G}_{j}
\\&+ \sum_{j\in J}\sum_{m\in M} (R_m C^{Gv}_{m} + E_m) x^{GT}_{jm}
\\&+ \sum_{(i,j) \in L} C^{H}_{ij} x^{L}_{ij}
\\&+ \sum_{s\in S} P_s \theta_s
\end{align}
$$

### Constraints

#### Generator number upperbound

$$
\sum_{j\in J} x^{G}_{j} \leq \overline{N^{G}}
$$

#### DG type selection

$$
\sum_{m\in M} x^{GT}_{jm} = x^{G}_{j}, \qquad \forall j\in J
$$

#### Optimality cuts

The main problem is iteratively tightened by optimality cuts generated from the sub problems. In the current code base, classical Benders, strengthened Benders, Lagrangian, integer optimality, and CGLP cuts may be added. See the section **Cuts Applied in the Algorithm** below.

#### Pretrained cardinality restrictions (optional)

When enabled, two rows $\sum_{i\in\mathcal{U}} x_i \ge \zeta_1$ and $\sum_{i\in\mathcal{L}} x_i \le \zeta_2$ restrict the binary design to a predicted region. Let $\mathbb{E}[x_i]$ be the probability-weighted average of the stored single-scenario optimal designs (`solutions/*.json`).

- If the stored solution of the combined problem over the same scenario list exists, $\mathcal{U}$ is the set of its components equal to 1.
- Otherwise (only in the phase-driven front end), $\mathcal{U} = \{i : \mathbb{E}[x_i] > 0.3334\}$.
- $\mathcal{L}$ is the complement of $\mathcal{U}$, $\zeta_1 = \sum_{i\in\mathcal{U}}\mathbb{E}[x_i]$, and $\zeta_2 = \sum_{i\in\mathcal{L}}\mathbb{E}[x_i]$.

The pre-training experiment in the Final Report defined $\mathcal{U}$ from the known global optimal design, according to the 2026-02-25 progress notes. A restricted master never reports a global lower bound.

### Collapsed SP block in the main problem (optional)

Controlled by `main_problem_model_type`:

| Type | Kept index | Name used in the Final Report |
|---|---|---|
| `base` | no block | — |
| `collapsed_no_time` (default) | none | remove time and scenario indices |
| `collapsed_by_time` | $t$ | only remove the scenario index |
| `collapsed_by_scenarios` | $s$ | only remove the time index |

The block adds master copies of $p^{Grt}, y^{C}, y^{L}, v, p^{G}, q^{G}, p, q, u^{L}, u^{S}, u^{I}$ over the kept index $a$, with the same constraint families as the sub problem (rated-power link, line connectivity, topology, DG operation, voltage, flow balance, no flow on open line, thermal limit). Data that depends on $t$ or $s$ is replaced by collapsed "hat" data. $P_s$ is normalized over the run's scenario list:

- `collapsed_no_time`, `weighted_average`: $\hat{D}_{j} = \big(\sum_{t}\sum_{s} P_s D_{jts}\big)/|T|$.
- `collapsed_no_time`, `first_time_expectation`: $\hat{D}_{j} = \sum_{s} P_s D_{jt_0 s}$, with $t_0$ the first time slot (no division by $|T|$).
- `collapsed_by_time` (weighted average only): $\hat{D}_{jt} = \sum_{s} P_s D_{jts}$.
- `collapsed_by_scenarios` (weighted average only): $\hat{D}_{js} = \sum_{t} D_{jts}/|T|$.

The same rule applies to $D^{p}$ and $D^{q}$. Line health $\hat{H}^{h}, \hat{H}^{nh}$ is the same collapse rounded to a binary: $\hat{H} = 1$ if the collapsed value is $\ge 0.5$, else $0$.

The block's operating cost enters the objective with a guidance weight $w$ (`added_obj_term_weight`):

$$
\min\; \text{(design cost)} + \sum_{s} P_s\theta_s + w\Big(\sum_{j}\sum_{a}\pi_a C^{Gu}_{j}\hat{p}^{G}_{ja} + \sum_{j}\sum_{a}\pi_a C^{L}\hat{y}^{L}_{ja}\Big),
$$

with $\pi_a = P_s$ for `collapsed_by_scenarios` and $\pi_a = 1$ otherwise.

The block never excludes a design, because full load shedding is always feasible. However, a master solved with $w>0$ minimizes a different function, so its bound is not a valid lower bound. With `bound_mode = resolve_at_zero_weight` the runner re-solves the master at $w=0$ after the iteration's cuts. With `bound_mode = heuristic` no valid bound is reported, and bound-based termination is disabled.

## Sub-problem Model

> Note:
>
> For one sub problem, the scenario index can represent either a single scenario or a scenario subset assigned to that sub model. The formulation below keeps the same notation $s\in S$ for simplicity.

### Variables

$p^{Grt}_{j}$: Continuous, non-negative. The rated power of DG at node $j$, $\forall j\in J$.

$x^{G}_j$: Binary local copy of the main-stage DG installation decision, fixed to the current main-stage solution, $\forall j\in J$.

$x^{GT}_{jm}$: Binary local copy of the main-stage DG type decision, fixed to the current main-stage solution, $\forall j\in J, m\in M$.

$x^{L}_{ij}$: Binary local copy of the main-stage line-hardening decision, fixed to the current main-stage solution, $\forall (i,j)\in L$.

$y^{C}_{ijts}$: Binary. Indicating the connected (1) or disconnected (0) state of line $(i,j)$ at time $t$ in scenario $s$, $\forall (i,j)\in L, t\in T, s\in S$.

$y^{L}_{jts}$: Continuous, $0 \le y^{L}_{jts} \le 1$. Ratio of load to be shed at node $j$ at time $t$ in scenario $s$, $\forall j\in J, t\in T, s\in S$.

$v_{jts}$: Continuous, non-negative. Voltage of bus $j$ at time $t$ in scenario $s$, $\forall j\in J, t\in T, s\in S$.

$p^{G}_{jts}$: Continuous, non-negative. Active power output of DG at node $j$ at time $t$ in scenario $s$, $\forall j\in J, t\in T, s\in S$.

$q^{G}_{jts}$: Continuous, free. Reactive power output of DG at node $j$ at time $t$ in scenario $s$, $\forall j\in J, t\in T, s\in S$.

$p_{ijts}, q_{ijts}$: Continuous, free. Active / reactive power flow of line $(i,j)$ at time $t$ in scenario $s$, $\forall (i,j)\in L, t\in T, s\in S$.

$u^{L}_{ijts}$: Continuous, free. Virtual flow of line $(i,j)$ at time $t$ in scenario $s$, $\forall (i,j)\in L, t\in T, s\in S$.

$u^{S}_{jts}$: Binary. Indicating whether node $j$ is a virtual source (1) or not (0) at time $t$ in scenario $s$, $\forall j\in J, t\in T, s\in S$.

$u^{I}_{jts}$: Continuous, non-negative. Virtual power injected from the virtual source at node $j$ at time $t$ in scenario $s$, $\forall j\in J, t\in T, s\in S$.

### Objective function

$$
\begin{align}
\min \quad z &= \sum_{j\in J}\sum_{t\in T}\sum_{s\in S} C^{Gu}_{j} p^{G}_{jts}
\\&+ \sum_{j\in J}\sum_{t\in T}\sum_{s\in S} C^{L} y^{L}_{jts}
\end{align}
$$

> Note: in the decomposition, the sub problem of a scenario group $G$ minimizes $\sum_{s\in G}P_s\big(\sum_{j,t} C^{Gu}_{j}p^{G}_{jts} + \sum_{j,t} C^{L}y^{L}_{jts}\big)$. $P_s$ is normalized over the run's scenario list, the same weights as in the main objective. Every cut of that group is written on $\sum_{s\in G}P_s\theta_s$, so group cuts are valid for any grouping.

### Constraints

#### Main-stage local-copy constraints

$$
x^{G}_{j} = \hat{x}^{G}_{j}, \qquad \forall j\in J
$$

$$
x^{GT}_{jm} = \hat{x}^{GT}_{jm}, \qquad \forall j\in J, m\in M
$$

$$
x^{L}_{ij} = \hat{x}^{L}_{ij}, \qquad \forall (i,j)\in L
$$

> Note: these equalities are explicitly implemented in the current sub model, and their dual multipliers are collected for cut generation.

#### Generator rated power linking

$$
p^{Grt}_{j} = \sum_{m\in M} R_{m} x^{GT}_{jm}, \qquad \forall j\in J
$$

#### Line connectivity

##### Line connected condition

$$
y^{C}_{ijts} \leq H^{nh}_{ijts}(1 - x^{L}_{ij}) + H^{h}_{ijts}x^{L}_{ij}, \qquad \forall (i,j)\in L, t\in T, s\in S
$$

#### System topology constraints

> Based on Single-commodity Flow (SCF) Method

##### Virtual power flow balance

$$
\left\{
\begin{aligned}
\sum_{k\in \Theta_{j}}u^{L}_{jkts} - \sum_{i\in \Pi_{j}}u^{L}_{ijts} &= u^{I}_{jts}-1
\\ u^{I}_{jts} &\leq |J|u^{S}_{jts}
\end{aligned}
\right.
,\qquad \forall j \in J, t\in T, s\in S
$$

##### No virtual flow on open line

$$
-|J| y^{C}_{ijts}\leq u^{L}_{ijts} \leq |J| y^{C}_{ijts}, \qquad \forall (i,j)\in L, t\in T, s\in S
$$

##### Radiality

$$
\sum_{(i,j)\in L} y^{C}_{ijts} = |J| - \sum_{j\in J}u^{S}_{jts}, \qquad \forall t\in T, s\in S
$$

#### DG operating constraints

##### Amount of generated power

$$
p^{G}_{jts} \leq p^{Grt}_j, \qquad \forall j\in J, t\in T, s\in S
$$

$$
q^{G}_{jts} \leq A^{G}_{j} p^{G}_{jts}, \qquad \forall j\in J, t\in T, s\in S
$$

#### System operating constraints

##### Voltage range & the slack bus of the system

$$
\underline{V} \leq v_{jts} \leq \overline{V}, \qquad \forall j \in J,t\in T,s\in S
$$

$$
V_{\text{slack},ts} = V_{0}, \qquad \forall t\in T, s\in S
$$

##### Voltage - flow relationships

$$
-M(1-y^{C}_{ijts}) \leq v_{its} - v_{jts} - \frac{R_{ij}p_{ijts} + X_{ij}q_{ijts}}{V_{0}} \leq M(1-y^{C}_{ijts}), \quad \forall (i,j)\in L, t\in T, s\in S
$$

##### Flow balance

$$
\sum_{k\in \Theta_{j}} p_{jkts} - \sum_{i\in \Pi_{j}}p_{ijts} = p^{G}_{jts} - (1 - y^{L}_{jts})D^{p}_{jts}, \qquad \forall j\in J, t\in T, s\in S
$$

$$
\sum_{k\in \Theta_{j}} q_{jkts} - \sum_{i\in \Pi_{j}}q_{ijts} = q^{G}_{jts} - (1 - y^{L}_{jts})D^{q}_{jts}, \qquad \forall j\in J, t\in T, s\in S
$$

##### No flow on open line

$$
-M y^{C}_{ijts}\leq p_{ijts} \leq My^{C}_{ijts}, \qquad \forall (i,j)\in L, t\in T, s\in S
$$

$$
-M y^{C}_{ijts}\leq q_{ijts} \leq My^{C}_{ijts}, \qquad \forall (i,j)\in L, t\in T, s\in S
$$

##### Linearized thermal limitation

$$
\left\{
\begin{aligned}
-2\overline{S_{ij}} \leq &\sqrt{3}p_{ijts} + q_{ijts} \leq 2\overline{S_{ij}}
\\ -\overline{S_{ij}} \leq &p_{ijts} \leq \overline{S_{ij}}
\\ -2\overline{S_{ij}} \leq &\sqrt{3}p_{ijts} - q_{ijts} \leq 2\overline{S_{ij}}
\end{aligned}
\right.
,\qquad \forall (i,j)\in L, t\in T, s\in S
$$

The polygon is intentional. Implemented big-M values:
- voltage-flow relation: $M = \overline{V} - \underline{V}$, because an open line carries no flow;
- no flow on open line: $M = \overline{S_{ij}}$ for $p_{ijts}$ and $M = 2\overline{S_{ij}}$ for $q_{ijts}$, both implied by the polygon above.

## Cuts Applied in the Algorithm

#### Classical Benders Cut

Let \(y\) denote the main-stage decision vector, and let \(y^{*}\) be the current main-stage solution at which the subproblem is evaluated. Let \(x\) denote the sub-stage operating variables, and let \(\theta\) denote the estimator of the recourse value in the main problem.

A general form of the classical Benders optimality cut is

$$
\theta \;\ge\; Q^{LP}(y^{*}) + \pi^{T}(y-y^{*}),
$$

where:

- \(Q^{LP}(y^{*})\) is the optimal objective value of the **LP relaxation** of the subproblem when the main-stage variables are fixed at \(y^{*}\),
- \(\pi\) is the vector of dual multipliers associated with the constraints that fix the subproblem-side copy of the main-stage variables to \(y^{*}\).

Equivalently, it can be written in affine form as

$$
\theta \;\ge\; \alpha + \pi^{T}y,
\qquad
\alpha = Q^{LP}(y^{*}) - \pi^{T}y^{*}.
$$

**Models to be solved for generating the cut:**

1. Solve the current main-stage problem and obtain \(y^{*}\).
2. Build the corresponding subproblem with the main-stage decisions fixed at \(y^{*}\).
3. Relax the integer variables in that subproblem and solve the resulting LP.
4. Read the dual multipliers \(\pi\) from the fixing / local-copy constraints, and then formulate the above cut.

In words, the classical Benders cut uses the local linear approximation of the LP-relaxed recourse function around the current main-stage point \(y^{*}\).

---

#### Strengthened Benders Cut

Let \(y^{R}\) denote the solution of the **LP relaxation of the main-stage problem**. Let \(z\) denote a copied main-stage decision vector used inside an auxiliary subproblem, and let \(x\) denote the sub-stage operating variables. Let \(\lambda\) denote the multiplier vector obtained from the relaxed subproblem.

A general form of the strengthened Benders cut is

$$
\theta \;\ge\; c^{T}\bar{x} + \lambda^{T}(y-\bar{z}),
$$

where:

- \(\lambda\) is obtained from the subproblem solved at \(y^{R}\),
- \((\bar{x},\bar{z})\) is obtained from an auxiliary **inner minimization problem**,
- \(c^{T}\bar{x}\) is the recourse-cost term corresponding to the obtained auxiliary solution.

The inner minimization problem can be written in the generic form

$$
(\bar{x},\bar{z}) \in
\arg\min_{x,z}
\left\{
c^{T}x - \lambda^{T}(z-y^{R})
:
(x,z)\in X
\right\},
$$

where \(X\) denotes the feasible region of the copied subproblem, including the original sub-stage constraints and the integrality requirements that are kept in this auxiliary model.

**Models to be solved for generating the cut:**

1. Solve the **LP relaxation of the main-stage problem** and obtain \(y^{R}\).
2. Build the corresponding subproblem with the copied main-stage variables fixed at \(y^{R}\), relax it, and solve it to obtain the multiplier vector \(\lambda\).
3. Using this \(\lambda\), solve the auxiliary inner minimization MILP above, and obtain \((\bar{x},\bar{z})\).
4. Form the strengthened cut
   $$
   \theta \;\ge\; c^{T}\bar{x} + \lambda^{T}(y-\bar{z}).
   $$

   Compared with the classical Benders cut, this strengthened cut still uses multiplier information from a relaxed subproblem, but its intercept is generated from an additional inner optimization model, which usually gives a tighter cut.

> Implemented form: $\sum_{s\in G}P_s\theta_s \ge L_{lb}(\lambda) + \lambda^{T}(y-y^{R})$, where $L(\lambda) = \min_{(x,z)\in X} c^{T}x - \lambda^{T}(z-y^{R})$ and $L_{lb}$ is the solver's dual bound of that MILP. At an exact inner solve this equals the formula above. Under a MIP gap or time limit the cut becomes weaker but stays valid. $X$ omits the main-problem-only rows (DG number limit, DG type selection), so the inner model is a relaxation. This is valid but can be weaker.

### Lagrangian cut

The multiplier search follows the regularized update of Rahmaniani et al. (2020), *The Benders Dual Decomposition Method*, at a fixed anchor $y^{*}$ (the relaxed main solution).

1. **Start.** $\lambda^{1}$ is the dual vector of the relaxed subproblem at $y^{*}$. Solve the inner minimization (as in the strengthened cut) at $\lambda^{1}$. This gives the dual bound $L_{lb}(\lambda^{1})$ and an incumbent $(\bar{x}^{1},\bar{z}^{1})$. Set $L_{best} = L_{lb}(\lambda^{1})$.
2. **Multiplier update** at iteration $t = 1,\dots,\kappa$ ($\kappa$ = `max_lagrangian_iterations`, default 10):
   $$
   \max_{\lambda,\eta}\; \eta - \frac{\delta_t}{2}\lVert\lambda^{t}-\lambda\rVert_2^2
   \quad\text{s.t.}\quad
   \eta \le c^{T}\bar{x}^{v} + \lambda^{T}(y^{*}-\bar{z}^{v}),\;\; v = 1,\dots,t,
   $$
   with $\delta_t = 0.01/(1+t)$. Each bundle row uses an inner incumbent, so it over-estimates $L(\lambda)$.
3. **Inner solve** at the new multiplier $\lambda^{t+1}$. Add its incumbent to the bundle. If $L_{lb}(\lambda^{t+1}) > L_{best}$, update $L_{best}$ and the stored multiplier $\lambda_{best}$.
4. **Stop** when $\eta - L_{best} \le 0.001\,|L_{best}|$ or $t = \kappa$. This rule is heuristic: $\eta$ from the regularized model does not certify the best multiplier. It affects cut strength only, not validity.
5. **Cut:** $\sum_{s\in G}P_s\theta_s \ge L_{best} + \lambda_{best}^{T}(y-y^{*})$.

> Note: only the optimality case is implemented. An infeasible recourse raises an error; no feasibility cut is generated. The former deterministic cut-generation model was removed.

### Integer optimality cut

At a binary anchor $\hat{x}$ (all binary design variables $x^{G}, x^{GT}, x^{L}$), let $q$ be the dual bound of the exact sub problem MILP of group $G$. Let $L$ be a global lower bound on the weighted recourse; $L=0$ because all operating costs are non-negative. The cut is

$$
\sum_{s\in G}P_s\theta_s \;\ge\; q - (q-L)\Big(\sum_{k:\hat{x}_k=1}(1-x_k) + \sum_{k:\hat{x}_k=0}x_k\Big).
$$

It is tight at $\hat{x}$ and valid for every binary design.

### CGLP cut

The CGLP cut follows Algorithms 4 and 5 of Angulo, Ahmed, and Dey (2016), *Improving the integer L-shaped method*, INFORMS Journal on Computing 28(3), 483–499. It is used at a binary anchor $\hat{x}$ when the weighted $\theta$ value is below the certified bound $q(\hat{x})$.

The cut-generating LP is built from:
- the main-problem structural rows (DG number limit, DG type selection);
- one row per evaluated binary point, with equality at the current anchor;
- all valid lower-bound rows already known for the group (every cut added for that group, plus the zero row), combined by convex multipliers;
- the prefix sets $V^{i}, \widehat{V}^{i}$ and their $W$-rows, which cover the binary points not yet evaluated.

It returns $(\alpha,\eta)$ and the cut $\sum_{s\in G}P_s\theta_s \ge -\eta - \alpha^{T}x$. The cut is added only if it is violated at $\hat{x}$.

Point values are dual bounds of the exact sub problem. If that solve is inexact, the cut stays valid but may not be tight at the true recourse value. In this repository, separation runs inside the outer decomposition iterations, not in a solver callback. Details: `cglp_paper_to_code_design_note.md`.

### Separated cuts and aggregated cuts

- The current code can add one cut for each scenario (or scenario group) separately.
- It can also aggregate the cuts generated in the same iteration into one probability-weighted cut before adding it to the main problem. Because every group cut is already written on $\sum_{s\in G}P_s\theta_s$, the aggregated cut is the plain sum of the group cuts.

## General Algorithm

- Read and process the input data for the selected scenarios and time periods.
- Build the main-stage model.
- Pretrained cardinality restrictions are opt-in. When enabled, the restricted master does not certify a global lower bound or global gap convergence.
- For each iteration:
  - solve the main-stage MILP and record its dual bound. It is a valid lower bound only when $w=0$ and no pretrained restriction is active;
  - obtain the current integer main-stage solution;
  - solve the relaxed main-stage model to get a fractional solution for strengthened / Lagrangian cut generation;
  - for each scenario or scenario group, build the copied-variable sub problem;
  - solve the relaxed sub problem to obtain dual information for classical Benders cuts;
  - optionally solve the integer sub problem to update the incumbent objective value;
  - generate classical Benders, strengthened Benders, Lagrangian, integer optimality, and / or CGLP cuts. The choice follows the fixed iteration ranges (`CutSchedule`) or the phase-driven policy (`PhasePolicy`);
  - add cuts either separately or in aggregated probability-weighted form.
- Stop when the iteration limit is reached or when the recorded incumbent and main-model bound satisfy the convergence tolerance used in the driver.

## Scenario data provenance

The line-state data $H^{nh}, H^{h}$ (`s_line_state_w_o_harden.csv`) comes from an external pole-fragility simulation (`scenario_generation.py`). That script and its inputs are not in this repository. Recorded assumptions:

- **Inputs:** hourly wind-speed paths for templates `s_1`–`s_10` with start-hour offsets; the number of poles on each 33-bus line; lognormal fragility parameters $(\mu,\beta)$ by pole class and age.
- **Pole failure probability:** $\Pr[\text{fail by hour } t \mid s] = \Phi\big((\ln v_{t,s}-\ln\mu)/\beta\big)$. Class-age pairs are hardened $(2,0)$, unhardened $(3,25)$, and LIC-aged $(3,50)$.
- **Cases:** the default no-harden case uses the LIC-aged pair $(3,50)$ for all lines. The plain $(3,25)$ case exists in the script but is disabled.
- **Simulation:** each pole draws one $u\sim U(0,1)$, shared by the no-harden, LIC-aged, and harden cases. The pole fails at the first hour with $\Pr \ge u$. A line records its number of failed poles and its earliest failure time.
- **Time expansion:** time step 1 h; repair begins at hour 22 and takes 4 h per failed pole. A line is down at output hour $t$ iff $(t+1)\in[\lceil t_{fail}\rceil, \lceil 22 + 4 n_{fail}\rceil)$. A scenario starts at its earliest failure (over both cases) and lasts $22 + 4\max n_{fail}$ hours.
- **Reproducibility:** the original script passes a seed but never applies it (`np.random.seed` is commented out), so the in-sample line states cannot be regenerated exactly. The out-of-sample file `out_sample_data/out_sample_line_status.csv` has 12 samples (`sample_id`) for each of `s_1`–`s_10`. It comes from a refined version with an explicit seed and command-line options (documented example: `--samples-per-scenario 12 --seed 20260329`). The exact command used for that file was not recorded.
- **Other inputs:** scenario probabilities (`s_probability.csv`) are an external table keyed by `scenario_id`; they do not sum to one and are normalized over the scenarios in use. Demand (`s_load_P3715kW_system.csv`) is given per node and scenario and is constant over time.
