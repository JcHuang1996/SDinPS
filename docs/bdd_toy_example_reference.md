# Reference note: toy example in *The Benders Dual Decomposition Method*

This note summarizes the small example used in Figure 1 (page 5) and Appendix C of the paper. It is intentionally short and is meant to be a **reference solution** for checking code output.

## 1. Test example

The toy problem is

$$
\begin{aligned}
\min_{y\in\{0,1\},\;x\ge 0} \quad & x \\
\text{s.t.}\quad
& x + 15y \ge 8,\\
& 3x + 10y \ge 13,\\
& x + 10y \ge 7,\\
& 2x - 10y \ge -1,\\
& 2x - 70y \ge -49.
\end{aligned}
$$

The optimal **integer** solution is

$$
(y^*,x^*)=(0,8),\qquad \text{optimal objective value }=8.
$$

For the Benders-style procedures, the relaxed master problem is

$$
\min_{0\le y\le 1,\;\theta} \; \theta,
$$

and the subproblem for a current master value $\bar y$ is

$$
\begin{aligned}
\min_{x,z} \quad & x \\
\text{s.t.}\quad
& x + 15z \ge 8,\\
& 3x + 10z \ge 13,\\
& x + 10z \ge 7,\\
& 2x - 10z \ge -1,\\
& 2x - 70z \ge -49,\\
& z = \bar y,\qquad z\in[0,1].
\end{aligned}
$$

There are **no feasibility cuts** in this example; only optimality cuts are needed.

## 2. What cut is added?

### Classical / generalized Benders cut

Solve the LP subproblem at the current master point $\bar y$. If $(\bar x,\bar z)$ is the optimal primal solution and $\lambda$ is the multiplier of $z=\bar y$, add

$$
\theta \ge \bar x + (y-\bar z)\lambda.
$$

### Strengthened Benders cut

At a **fractional** master point, first get $\lambda$ from the LP subproblem, then solve the inner problem with an **integer** copy variable:

$$
\min_{x,z}\; x - \lambda z + \lambda \bar y
\quad\text{s.t. the same five constraints, } z\in\{0,1\}.
$$

If the solution is $(\bar x,\bar z)$, add the lifted cut

$$
\theta \ge \bar x + (y-\bar z)\lambda.
$$

In this toy example, the strengthened cut at the third master point is **parallel** to the classical cut but shifted upward; the upward shift is $6$, and the resulting strengthened-master lower bound becomes $6.125$.

### Lagrangian cut

Instead of fixing $\lambda$ to the LP-subproblem multiplier, optimize the Lagrangian dual over $\lambda$ (the paper uses a subgradient update in Appendix C). The final cut has the same form,

$$
\theta \ge \bar x + (y-\bar z)\lambda,
$$

but with a better multiplier, so it is stronger than the strengthened cut.

## 3. Exact iteration results

### 3.1 Classical Benders cuts (baseline from Figure 1)

The paper states that the classical method converges after **five master solves** to the LP solution $y=0.58$ with objective $2.4$. The exact cut sequence is:

| Master solve | Current master solution | Cut added from subproblem | Notes |
|---|---:|---|---|
| 0 | $y_0=0$ | $\theta \ge 8 - 15y$ | First cut |
| 1 | $y_1=1$ | $\theta \ge -\tfrac{49}{2} + 35y$ | Second cut |
| 2 | $y_2=\tfrac{13}{20}=0.65$ | $\theta \ge -\tfrac12 + 5y$ | Third cut |
| 3 | $y_3=\tfrac{17}{40}=0.425$ | $\theta \ge \tfrac{13}{3} - \tfrac{10}{3}y$ | Fourth cut |
| 4 | $y_4=\tfrac{29}{50}=0.58$ | none; converged | Final LP value $\theta=\tfrac{12}{5}=2.4$ |

So the final classical master is bounded by

$$
\theta \ge \max\left\{8-15y,\; -\frac{49}{2}+35y,\; -\frac12+5y,\; \frac{13}{3}-\frac{10}{3}y\right\},
$$

and its optimum is

$$
y=\frac{29}{50}=0.58,\qquad \theta=\frac{12}{5}=2.4.
$$

> The fourth classical cut is **not written explicitly** in Appendix C, but it follows directly from the toy subproblem: at $y=0.425$, the second original constraint is active, so the cut is $\theta \ge \frac{13}{3}-\frac{10}{3}y$.

### 3.2 Strengthened Benders cuts (Appendix C.1)

For the first two iterations, the master solutions are integer, so the strengthened and classical cuts are the same.

| Master solve | Current master solution | Cut added | Resulting master bound |
|---|---:|---|---:|
| 0 | $y_0=0$ | $\theta \ge 8 - 15y$ | next master picks $y=1$ |
| 1 | $y_1=1$ | $\theta \ge -\tfrac{49}{2} + 35y$ | next master picks $y=0.65$ with lower bound $-1.75$ |
| 2 | $y_2=\tfrac{13}{20}=0.65$ | classical LP cut would be $\theta \ge -\tfrac12 + 5y$; the **strengthened** cut is $\theta \ge \tfrac{11}{2} + 5y$ | new master solution $y=\tfrac18=0.125$, lower bound $\theta=\tfrac{49}{8}=6.125$ |

Key point:

$$
\underbrace{\theta \ge -\frac12 + 5y}_{\text{classical at }y=0.65}
\qquad\Longrightarrow\qquad
\underbrace{\theta \ge \frac{11}{2} + 5y}_{\text{strengthened at }y=0.65}
$$

The two cuts are parallel (same slope $5$), and the strengthened cut is tighter by

$$
\frac{11}{2} - \left(-\frac12\right) = 6.
$$

The updated master after the strengthened cut is

$$
\theta \ge 8-15y,\qquad
\theta \ge -\frac{49}{2}+35y,\qquad
\theta \ge \frac{11}{2}+5y,
$$

whose optimum is

$$
y=\frac18=0.125,\qquad \theta=\frac{49}{8}=6.125.
$$

### 3.3 Lagrangian cuts (Appendix C.2)

Again, the first two master cuts are the same as above:

$$
\theta \ge 8-15y,
\qquad
\theta \ge -\frac{49}{2}+35y.
$$

With only these two cuts, the master gives

$$
y=\frac{13}{20}=0.65,\qquad \theta=-\frac74=-1.75.
$$

Then Appendix C.2 performs the Lagrangian-dual search:

1. **Initialize** with the LP-subproblem multiplier $\lambda=5$ at $y=0.65$.
2. Solve the inner problem with $\lambda=5$:
   $$
   \min_{x,z}\; x-5z
   $$
   subject to the five original inequalities and $z\in\{0,1\}$.
   This gives
   $$
   z=1,\qquad x=\frac{11}{2}=5.5.
   $$
   The corresponding integer incumbent is
   $$
   x=5.5,\; y=z=1 \quad\Rightarrow\quad \text{objective } 10.5.
   $$
3. Update the multiplier by the paper's subgradient step to
   $$
   \lambda=2.5=\frac52.
   $$
4. Solve the inner problem with $\lambda=2.5$:
   $$
   \min_{x,z}\; x-2.5z
   $$
   subject to the same constraints and $z\in\{0,1\}$.
   This gives
   $$
   z=0,\qquad x=8,
   $$
   hence the **Lagrangian cut**
   $$
   \theta \ge 8 + \frac52 y.
   $$

This cut gives the convex-hull representation for the master in the example. The master with

$$
\theta \ge 8-15y,
\qquad
\theta \ge -\frac{49}{2}+35y,
\qquad
\theta \ge 8+\frac52 y
$$

has optimum

$$
y=0,\qquad \theta=8,
$$

which is already the optimal **integer** solution of the original toy problem.

## 4. Final reference outputs

For quick checking, the expected cut sets are:

- **Classical**:
  $$
  \left\{8-15y,\; -\frac{49}{2}+35y,\; -\frac12+5y,\; \frac{13}{3}-\frac{10}{3}y\right\}
  $$
  with final LP master solution
  $$
  y=0.58,\quad \theta=2.4.
  $$

- **Strengthened**:
  $$
  \left\{8-15y,\; -\frac{49}{2}+35y,\; \frac{11}{2}+5y\right\}
  $$
  with final root-master solution
  $$
  y=0.125,\quad \theta=6.125.
  $$

- **Lagrangian**:
  $$
  \left\{8-15y,\; -\frac{49}{2}+35y,\; 8+\frac52 y\right\}
  $$
  with final solution
  $$
  y=0,\quad \theta=8.
  $$

## Source

Rahmaniani, Ahmed, Crainic, Gendreau, and Rei, *The Benders Dual Decomposition Method*, Figure 1 (page 5), Section 3, and Appendix C.
