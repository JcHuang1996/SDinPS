[TOC]

Historical experiment results below predate the 2026-10-28 maintenance and were not recomputed.
Guidance-objective values and bounds from pretrained search restrictions do not certify global gaps.

# Problem Statement

## Introduction

This project studies a resilience-oriented planning problem for a distribution system or microgrid under uncertain disruptive events. The system operator must make some design decisions before the events happen, and then operate the system after the events are realized. The purpose is to build a system that can keep serving load as much as possible during difficult conditions, while still controlling the total cost.

Some parts of the problem do not change after the event is realized, while other parts depend on the realized event. The unchanged part is the long-term planning design. This includes where to install distributed generators, which generator type is selected at each installed location, and which lines are hardened. These decisions are made before uncertainty is revealed, and they are shared by all later operating conditions.

The event-related part is scenario based. Each scenario describes possible system conditions over time, such as line availability and load level. After a scenario is realized, the operator must decide how to run the system under that condition. This includes which lines are connected, how generators are dispatched, how power flows through the network, and how much load must be shed if full service cannot be maintained.

The problem therefore links one set of fixed long-term design decisions with many possible operating conditions. The model evaluates these decisions by their expected total cost across all scenarios.

The below plot give a overall description on how the problem is solved.

![General workflow.drawio](/Users/huangjiacheng/drawio/SDinPS/General workflow.drawio.png)

## Decisions, objecitive, constraints

The model includes both planning decisions and operating decisions. At the planning level, the model decides the following long-term choices: where to install distributed generators, which generator type is selected at each installed location, and which lines should be hardened. They are made once and then shared by all scenarios and all time periods. In operation, the model decides which available lines are actually connected, how much active and reactive power each installed generator should produce, how power should flow through the network, what the bus voltages are, and how much load must be shed when full service is not possible. The model also uses some extra topology variables. These variables are not physical power quantities. They are only used to keep the operated network in a valid radial structure.

The objective is to minimize the total expected cost. This cost has several parts. First, there is a fixed cost for installing generators. Second, there is an investment cost that depends on the selected generator type and its capacity. Third, there is a hardening cost for the lines that are upgraded. After that, the model also counts operating cost in each scenario, mainly the cost of local generation and the penalty for load shedding. By combining all of these terms, the model looks for a design that is both affordable and resilient.

The constraints connect these decisions to the physical and logical limits of the system. The number of installed generators is limited, and a generator type can be chosen only when a generator is installed at that node. A line can be connected in operation only if it is healthy in that scenario and time period, with the health depending on whether the line was hardened. Generator output is limited by installed capacity and reactive power capability. The network must satisfy active and reactive power balance at each node. Bus voltages must stay within allowed bounds, and the slack bus voltage is fixed. Power flow on each line must respect the line status and thermal limit. Finally, the operated network must remain radial, so the model prevents invalid cycles by using the auxiliary topology variables.

## Data Source

The data in this project can be divided into deterministic data and scenario-based data. The deterministic part describes the basic distribution system and the planning cost structure. It includes the set of buses and lines, the electrical parameters of each line, the base location of loads, and the slack-bus setting. It also includes the planning and operating cost data needed by the model, such as the maximum number of distributed generators that can be installed, the available generator sizes, generator investment cost, generator operating cost, line hardening cost, voltage limits, line capacity limit, and the penalty applied to unserved load.

These deterministic inputs describe the physical system that is always present, no matter which disruption happens later. They define what assets can be installed, how power can move through the network, and what operating limits must be respected after a disturbance.

The scenario-based data describes uncertain operating conditions during disruptive events. This part includes the probability of each scenario, the demand level at each node under that scenario, and the time-varying availability of each line. The line availability is described separately for the hardened and non-hardened cases, because the planning model must compare the benefit of strengthening a line before the event happens. Together, these inputs allow the model to evaluate one planning design against many possible damage and recovery paths over time.

![Workflow.drawio](/Users/huangjiacheng/drawio/SDinPS/Workflow.drawio.png)

## Scenario Simulation Process

### Line status

The scenario generation process is designed to represent storm-driven damage and post-event recovery in a simple but realistic way. It starts from wind-speed trajectories over time, together with fragility information that links wind intensity to the failure risk of utility poles. Because line vulnerability depends on the condition of the supporting poles, the model can distinguish between a normal line and a hardened line by assigning them different fragility characteristics.

For each storm path, failure is simulated at the pole level. As the wind changes over time, the cumulative probability of failure increases. A random draw is then used to determine whether and when each pole fails. This produces different damage realizations even under the same general storm pattern, which is important because real outages are not fully deterministic.

The pole-level results are then translated into line-level outage states. If one or more poles on a line fail, the line is treated as unavailable from the failure time until repair is completed. The repair duration depends on the amount of damage, so lines with more failed poles remain out of service for longer. Repeating this process across storm patterns produces a set of scenarios with different probabilities, damage timing, and restoration paths.

As a result, each scenario gives a time-dependent view of system conditions after a disruptive event. This makes it possible to test whether a planning decision can still support the network when line outages, load conditions, and recovery progress differ from one realization to another.

### Demand

Though the demand data is deterministic with respect to location and time in the dataset, the variability on the power demand can be represented simply by adding purterbation on each location's demand data following a given distribution.

# Formulation

## Notations

### Notation naming rules

The notation system follows these rules:

- **Uppercase Roman letters denote data or parameters**, while **lowercase Roman letters and Greek letters denote decision variables**.
- **Subscripts** are used to distinguish **dimensions or indices**. For example, $P_s$ denotes the probability of scenario $s$, where $P$ indicates probability and $s$ identifies the scenario.
- **Superscript**s are used to distinguish **meanings**. Variables with similar roles or close relationships may share the same base letter, while superscripts indicate their specific interpretations. For example, since $X$ is commonly used for main-problem decision variables in two-stage decomposition, our main-problem variables are generally written as $x$, with superscripts such as $G$, $L$, and $GT$ representing generator, line, and generator type, respectively.

### Sets for the integrated model

- $S$: set of scenarios.
- $J$: set of buses.
- $L$: set of lines.
- $T$: set of time periods.
- $M$: set of DG types.

### Parameters for the integrated model

- $\overline{N^{G}}$: maximum number of DG installations.
- $P_s$: probability of scenario $s$, for all $s\in S$.
- $C^{Gf}_{j}$: fixed DG installation cost at bus $j$.
- $R_m$: rated power of DG type $m$.
- $C^{Gv}_{m}$: unit investment cost of DG type $m$.
- $E_m$: extra adjustment cost of DG type $m$.
- $C^{Gu}_{j}$: unit generation cost at bus $j$.
- $A^{G}_{j}$: coefficient for the DG reactive power limit at bus $j$.
- $C^{H}_{ij}$: hardening cost of line $(i,j)$.
- $R_{ij},X_{ij}$: resistance and reactance of line $(i,j)$.
- $H^{nh}_{ijts}$: health status of line $(i,j)$ at time $t$ in scenario $s$ without hardening.
- $H^{h}_{ijts}$: health status of line $(i,j)$ at time $t$ in scenario $s$ with hardening.
- $\underline{V},\overline{V}$: lower and upper voltage bounds.
- $V_0$: slack-bus voltage.
- $\overline{S}_{ij}$: thermal limit of line $(i,j)$.
- $\Theta_j$: buses represented as children of bus $j$ in the chosen line orientation.
- $\Pi_j$: buses represented as parents of bus $j$ in the chosen line orientation.
- $D^{p}_{jts},D^{q}_{jts}$: active and reactive demand at bus $j$ at time $t$ in scenario $s$.
- $C^{L}$: load-shedding penalty coefficient.
- $M$: sufficiently large constant used in voltage and flow linking constraints.

### Decision variables for the integrated model

- $x^{G}_{j}$: 1 if a DG is installed at bus $j$, and 0 otherwise.
- $x^{GT}_{jm}$: 1 if DG type $m$ is selected at bus $j$, and 0 otherwise.
- $p^{Grt}_{j}$: installed DG rated power at bus $j$.
- $x^{L}_{ij}$: 1 if line $(i,j)$ is hardened, and 0 otherwise.
- $y^{C}_{ijts}$: 1 if line $(i,j)$ is connected at time $t$ in scenario $s$, and 0 otherwise.
- $y^{L}_{jts}$: load-shedding ratio at bus $j$ at time $t$ in scenario $s$.
- $v_{jts}$: bus voltage at bus $j$ at time $t$ in scenario $s$.
- $p^{G}_{jts},q^{G}_{jts}$: active and reactive DG output at bus $j$ at time $t$ in scenario $s$.
- $p_{ijts},q_{ijts}$: active and reactive power flow on line $(i,j)$ at time $t$ in scenario $s$.
- $u^{L}_{ijts}$: virtual flow on line $(i,j)$ at time $t$ in scenario $s$.
- $u^{S}_{jts}$: 1 if bus $j$ is selected as a virtual source at time $t$ in scenario $s$, and 0 otherwise.
- $u^{I}_{jts}$: virtual injected flow at bus $j$ at time $t$ in scenario $s$.

### Additional notation for decomposition and cuts

- $x$: vector of first-stage planning decisions.
- $x^{*}$: current first-stage solution at which a cut is generated.
- $x^{R}$: solution of the relaxed first-stage model.
- $y$: vector of operating variables in the recourse model.
- $z$: copied first-stage decision vector used inside auxiliary cut-generation models.
- $\bar{y},\bar{z}$: inner-problem solutions used to build strengthened and Lagrangian cuts.
- $\theta$: estimator of recourse cost in the decomposed main problem.
- $Q^{LP}(x^{*})$: optimal value of the LP-relaxed recourse problem at $x^{*}$.
- $\pi$: dual multiplier vector used in a classical Benders cut.
- $\lambda$: multiplier vector used in strengthened Benders cuts and Lagrangian cuts.
- $c$: coefficient vector of the recourse objective.
- $X$: feasible region of the copied recourse problem.
- $\eta$: lift value used in the regularized multiplier-update step for Lagrangian cuts.
- $\delta$: stabilization parameter in the multiplier update model.
- $\gamma$: minimum lift improvement threshold.
- $\kappa$: maximum number of multiplier-update iterations.

## Comprehensive Formulation

The full problem can be written as one integrated MILP that contains both planning decisions and scenario-based operating decisions.

### Objective

The objective minimizes investment cost, hardening cost, and expected operating cost over all scenarios.

$$
\begin{aligned}
\min \quad z
=& \sum_{j\in J} C^{Gf}_{j} x^{G}_{j}
+ \sum_{j\in J}\sum_{m\in M} \left(R_m C^{Gv}_{m} + E_m\right)x^{GT}_{jm}
+ \sum_{(i,j)\in L} C^{H}_{ij} x^{L}_{ij}
\\
&+ \sum_{s\in S} P_s
\left(
\sum_{j\in J}\sum_{t\in T} C^{Gu}_{j} p^{G}_{jts}
+ \sum_{j\in J}\sum_{t\in T} C^{L} y^{L}_{jts}
\right).
\end{aligned}
$$

### Planning selection constraints

These constraints limit the total number of DG sites and force each installed DG to choose exactly one type.

$$
\sum_{j\in J} x^{G}_{j} \leq \overline{N^{G}}
$$

$$
\sum_{m\in M} x^{GT}_{jm} = x^{G}_{j},
\qquad \forall j\in J
$$

### DG sizing constraint

This constraint links the installed DG size to the selected DG type.

$$
p^{Grt}_{j} = \sum_{m\in M} R_m x^{GT}_{jm},
\qquad \forall j\in J
$$

### Line availability constraint

This constraint allows a line to be connected only when it is healthy under the chosen hardening decision.

$$
y^{C}_{ijts} \leq H^{nh}_{ijts}(1-x^{L}_{ij}) + H^{h}_{ijts}x^{L}_{ij},
\qquad \forall (i,j)\in L,\ t\in T,\ s\in S
$$

### Topology constraints

These constraints work together to keep the operated network radial by using a single-commodity virtual flow model.

$$
\sum_{k\in \Theta_{j}}u^{L}_{jkts} - \sum_{i\in \Pi_{j}}u^{L}_{ijts}
= u^{I}_{jts}-1,
\qquad \forall j\in J,\ t\in T,\ s\in S
$$

$$
u^{I}_{jts} \leq |J|u^{S}_{jts},
\qquad \forall j\in J,\ t\in T,\ s\in S
$$

$$
-|J|y^{C}_{ijts} \leq u^{L}_{ijts} \leq |J|y^{C}_{ijts},
\qquad \forall (i,j)\in L,\ t\in T,\ s\in S
$$

$$
\sum_{(i,j)\in L} y^{C}_{ijts} = |J| - \sum_{j\in J}u^{S}_{jts},
\qquad \forall t\in T,\ s\in S
$$

### DG operating constraints

These constraints limit active output by installed capacity and reactive output by the DG capability rule.

$$
p^{G}_{jts} \leq p^{Grt}_{j},
\qquad \forall j\in J,\ t\in T,\ s\in S
$$

$$
q^{G}_{jts} \leq A^{G}_{j} p^{G}_{jts},
\qquad \forall j\in J,\ t\in T,\ s\in S
$$

### Voltage constraints

These constraints keep bus voltage within the allowed range and fix the slack-bus voltage.

$$
\underline{V} \leq v_{jts} \leq \overline{V},
\qquad \forall j\in J,\ t\in T,\ s\in S
$$

$$
v_{\text{slack},ts} = V_0,
\qquad \forall t\in T,\ s\in S
$$

### Voltage-flow linking constraints

These constraints connect voltage drop and line flow when a line is connected, and relax that relation when the line is open.

$$
-M(1-y^{C}_{ijts})
\leq
v_{its}-v_{jts}-\frac{R_{ij}p_{ijts}+X_{ij}q_{ijts}}{V_0}
\leq
M(1-y^{C}_{ijts}),
\qquad \forall (i,j)\in L,\ t\in T,\ s\in S
$$

### Power balance constraints

These constraints enforce active and reactive balance at each bus after generation, flow, and load shedding are considered.

$$
\sum_{k\in \Theta_{j}} p_{jkts} - \sum_{i\in \Pi_{j}} p_{ijts}
= p^{G}_{jts} - (1-y^{L}_{jts})D^{p}_{jts},
\qquad \forall j\in J,\ t\in T,\ s\in S
$$

$$
\sum_{k\in \Theta_{j}} q_{jkts} - \sum_{i\in \Pi_{j}} q_{ijts}
= q^{G}_{jts} - (1-y^{L}_{jts})D^{q}_{jts},
\qquad \forall j\in J,\ t\in T,\ s\in S
$$

### Flow-switching constraints

These constraints force active and reactive flow to be zero when a line is not connected.

$$
-My^{C}_{ijts} \leq p_{ijts} \leq My^{C}_{ijts},
\qquad \forall (i,j)\in L,\ t\in T,\ s\in S
$$

$$
-My^{C}_{ijts} \leq q_{ijts} \leq My^{C}_{ijts},
\qquad \forall (i,j)\in L,\ t\in T,\ s\in S
$$

### Thermal limit constraints

These constraints use a linear outer approximation to keep each line flow within its thermal capacity.

$$
-2\overline{S}_{ij}
\leq \sqrt{3}p_{ijts} + q_{ijts}
\leq 2\overline{S}_{ij},
\qquad \forall (i,j)\in L,\ t\in T,\ s\in S
$$

$$
-\overline{S}_{ij}
\leq p_{ijts}
\leq \overline{S}_{ij},
\qquad \forall (i,j)\in L,\ t\in T,\ s\in S
$$

$$
-2\overline{S}_{ij}
\leq \sqrt{3}p_{ijts} - q_{ijts}
\leq 2\overline{S}_{ij},
\qquad \forall (i,j)\in L,\ t\in T,\ s\in S
$$

### Variable domains

The variable domains complete the mixed-integer linear model.

$$
x^{G}_{j},\ x^{GT}_{jm},\ x^{L}_{ij},\ y^{C}_{ijts},\ u^{S}_{jts}\in \{0,1\}
$$

$$
p^{Grt}_{j}\geq 0,\quad
0\leq y^{L}_{jts}\leq 1,\quad
v_{jts}\geq 0,\quad
u^{I}_{jts}\geq 0
$$

$$
p^{G}_{jts},\ q^{G}_{jts},\ p_{ijts},\ q_{ijts},\ u^{L}_{ijts}\in \mathbb{R}
$$



# Methodology

## Two Stage Decomposition

The integrated MILP is large because it includes a full network model, many possible scenarios, and binary variables in both planning and scenario-based operation. In abstract form, it can be written as

$$
\min_{x,y}\left\{f^{T}x+c^{T}y:\;Bx\ge b,\;Wy+Tx\ge h,\;(x,y)\in X\right\}.
$$

Here, $x$ denotes the (MP) decision vector and $y$ denotes the (SP) operating vector. Because of this structure, we use a two-stage decomposition method. The idea is close to Benders decomposition. The planning decisions are modeled in the (MP), and each scenario is evaluated in one (SP).

In each iteration, the (MP) proposes a planning solution. The (SP)s then evaluate that solution and return cut information. Since each (SP) is still a MILP, classical LP-based Benders cuts alone are often weak. For that reason, the method also uses strengthened Benders cuts, Lagrangian cuts, and CGLP cuts. The process stops when the (MP) bound and the incumbent value are close enough.

The workflow of the two stage decomposition scheme can be previewed in the following plot. Each component will be introduced later.

![Iteration Structure.drawio](/Users/huangjiacheng/drawio/SDinPS/Iteration Structure.drawio.png)

### Problem Division

The detailed division for (MP) and (SP), and interaction between the (MP) and one (SP) in an iteration, can be described as below:

![MP-SP-Interaction.drawio](/Users/huangjiacheng/drawio/SDinPS/MP-SP-Interaction.drawio.png)

#### Main Problem (MP)

Note that the (MP) may contain two optional parts for enhancing the decomposition method's performance, the pre-training based cuts, and a set of variables, constraints and objective terms from the subproblems (SP Block). 

![image-20260413115552509](/Users/huangjiacheng/Library/Application Support/typora-user-images/image-20260413115552509.png)

##### Initial MP & Iterative cuts

In the abstract notation above, the (MP) variable is $x$. In the exact formulation, it corresponds to the planning variables $(x^{G},x^{GT},x^{L})$. Let $\theta_s$ be the estimator of the (SP) cost for scenario $s$. The (MP) can be written as

$$
\begin{aligned}
\min \quad
& \sum_{j\in J} C^{Gf}_{j} x^{G}_{j}
+ \sum_{j\in J}\sum_{m\in M} \left(R_m C^{Gv}_{m}+A_m\right)x^{GT}_{jm}
+ \sum_{(i,j)\in L} C^{H}_{ij}x^{L}_{ij}
+ \sum_{s\in S} P_s \theta_s
\\
\text{s.t.}\quad
& \sum_{j\in J} x^{G}_{j} \le \overline{N^{G}},
\\
& \sum_{m\in M} x^{GT}_{jm} = x^{G}_{j}, \qquad \forall j\in J,
\\
& \text{all valid cuts generated from the subproblems or pretraining}
\\
& x^{G}_{j},x^{GT}_{jm},x^{L}_{ij}\in\{0,1\},\qquad \theta_s\ge 0.
\end{aligned}
$$

The (MP) contains the long-term design decisions only. It does not model the full post-event operation. Instead, it keeps one recourse estimator for each scenario and raises these estimators through iterative cuts.

##### Optional parts

The optional parts of MP, i.e. the pre-training based cuts, and the SP Block, will be introduced later.

#### Subproblem

In the abstract notation above, the (SP) operating vector is $y$. In the exact formulation, it corresponds to the operating and topology variables such as $(p^{Grt},y^{C},y^{L},v,p^{G},q^{G},p,q,u^{L},u^{S},u^{I})$. The copied (MP) vector is denoted by $\hat{x}$, and in the exact formulation its components are the copied variables $(x^{G},x^{GT},x^{L})$. For a fixed (MP) solution $\hat x$, the (SP) for one scenario $s$ can be written as

$$
\min \quad \theta_s = 
\sum_{j\in J}\sum_{t\in T} C^{Gu}_{j} p^{G}_{jts}
\;+\;
\sum_{j\in J}\sum_{t\in T} C^{L} y^{L}_{jts}
$$

subject to the local-copy constraints

$$
x^{G}_{j}=\hat x^{G}_{j},\qquad
x^{GT}_{jm}=\hat x^{GT}_{jm},\qquad
x^{L}_{ij}=\hat x^{L}_{ij},
$$

and the operating constraints

$$
p^{Grt}_{j} = \sum_{m\in M} R_m x^{GT}_{jm},
$$

$$
y^{C}_{ijts} \le H^{nh}_{ijts}(1-x^{L}_{ij}) + H^{h}_{ijts}x^{L}_{ij},
$$

together with the topology, DG output, voltage, power balance, no-flow, and thermal-limit constraints defined in the integrated formulation.

Note that though the local-copy constraints increase the number of constraints in (SP), constraints in such format add little computational pressure on modern MILP solvers, while benefits both readability and implementation for involving formulations.

The (SP) returns the operating cost under the fixed design. For a scenario group, the same structure is used and the objective is summed over the scenarios in that group. The (SP) also provides dual information from the local-copy constraints when the relaxed model is solved. Since it still contains binary variables for network topology, it remains a MILP. This is the main reason why stronger cuts are needed.

### Iterative Cuts

#### Classical Benders cut 

Let $x^{*}$ be the current (MP) solution. First solve the relaxed (SP) at $x^{*}$. This gives the LP-relaxed value $Q_s^{LP}(x^{*})$ and the multiplier vector $\pi_s$ from the local-copy constraints. The cut is
$$
\theta_s \ge Q_s^{LP}(x^{*}) + \pi_s^{T}(x-x^{*}).
$$

This cut is easy to generate and gives a valid local lower bound. However, because the true (SP) is a MILP, the LP relaxation can be loose, so the cut may be weak.

#### Strengthened Benders cut

Let $x^{R}$ be the solution of the relaxed (MP). First solve the relaxed copied (SP) at $x^{R}$ and obtain the multiplier vector $\lambda_s$. Then solve the **Inner Minimization Problem (INMIP)** MILP (which is similar to (SP))
$$
(\bar y,\bar z)\in
\arg\min_{y,z}
\left\{
c^{T}y-\lambda_s^{T}(z-x^{R}) : (y,z)\in X_s
\right\},
$$

where $X_s$ is the feasible region of the copied (SP) for scenario $s$. After obtaining $(\bar y,\bar z)$, add

$$
\theta_s \ge c^{T}\bar y + \lambda_s^{T}(x-\bar z).
$$

Compared with the classical Benders cut, this cut keeps the same slope idea but uses an integer inner problem to produce a tighter intercept.

#### Lagrangian cut

The Lagrangian cut keeps the same final cut form as the strengthened Benders cut, but it improves the multiplier before the cut is added. Start from the multiplier obtained from the strengthened-cut step, denoted by $\lambda^{1}$. Solve the (INMIP) and obtain $(\bar y^{1},\bar z^{1})$. Then solve the regularized multiplier-update model
$$
\max_{\lambda,\eta}
\left\{
\eta-\frac{\delta}{2}\|\lambda-\lambda^{v-1}\|_2^2
\right\}
$$

subject to

$$
\eta \le c^{T}\bar y^{r} + \lambda^{T}(x^{R}-\bar z^{r}),
\qquad r=1,\dots,v-1.
$$

The updated multiplier is used in a new (INMIP) solve. This process is repeated until the lift improvement becomes small. If the final multiplier and inner solution are $(\lambda,\bar y,\bar z)$, the added cut is

$$
\theta_s \ge c^{T}\bar y + \lambda^{T}(x-\bar z).
$$

This method tries to find a stronger multiplier than the one obtained from one relaxed (SP) solve.

#### CGLP cut

The cut-generating linear program (CGLP) cut is used after an exact (SP) MILP evaluation is available. Let $x^{*}$ be a binary (MP) solution and let $Q_s(x^{*})$ be the exact value obtained from the (SP) MILP. First, solve the exact (SP) and check whether $\theta_s(x^{*}) < Q_s(x^{*})$. If so, store this exact point together with the exact value. Then solve a cut-generating linear program built from all exact points already collected and all lower-bound cuts already known. This model returns coefficients $(\alpha,\eta)$ such that
$$
\alpha^{T}x + \theta_s \ge -\eta.
$$

This cut is valid for the convex hull implied by the accumulated exact points and lower bounds. In that sense, it uses more global information than one local multiplier cut.

The CGLP cut involves multiple complex LP model and algorithms, and the details can be checked from the reference:

> Angulo, G., Ahmed, S., & Dey, S. S. (2016). *Improving the integer L-shaped method.* *INFORMS Journal on Computing, 28*(3), 483-499. https://doi.org/10.1287/ijoc.2016.0695 

The above cuts can be added for each scenario or scenario group separately. They can also be aggregated by scenario probability and added as one weighted cut in one iteration.

### Performance Improvement - Subproblem Block

To give the (MP) more structural information, we also add a reduced (SP) block directly into the (MP). This block does not replace the true (SP). It gives the (MP) an approximate operating view before exact evaluation is made.

This block adds compressed versions of the (SP) variables, constraints, and operating-cost terms to the (MP). Depending on the selected variant, the compression removes both time and scenario indices, removes only scenario indices, or removes only time indices. The added variables describe approximate line connection, DG dispatch, load shedding, voltage, line flow, and topology. The added constraints keep the same physical logic as the (SP), but they use collapsed demand and line-health data.

The block can also add a small weighted approximation of operating cost into the (MP) objective. This extra term is only a guide. The true recourse cost is still enforced by the scenario estimators and the iterative cuts. In practice, this enriched (MP) gives a better starting structure and can reduce the number of iterations needed by the decomposition.

## Pre-training Based Cuts

The pre-training based cut method adds simple cardinality cuts to the (MP) when some binary decisions can be predicted in advance with reasonable reliability. Let $x\in\{0,1\}^{n}$ be the binary (MP) decision vector, and let $\hat{x}$ be a predicted good solution. Define $\mathcal{U}=\{i:\hat{x}_i=1\}$ and $\mathcal{L}=\{i:\hat{x}_i=0\}$. Then choose two parameters $\zeta_1$ and $\zeta_2$, and add

$$
\sum_{i\in \mathcal{U}} x_i \ge \zeta_1,
\qquad
\sum_{i\in \mathcal{L}} x_i \le \zeta_2.
$$

These cuts do not fix individual variables. Instead, they guide the (MP) toward a promising region of the binary decision space. This is useful when some binary choices are easier to predict than the full optimal solution.

An optional extension is to solve several (MP) variants with different sign combinations of these two cardinality cuts, as following. This gives a simple way to reduce the risk that one predicted region is too narrow, while still keeping the benefit of prediction-guided search. Note that those cardinality cuts divide the problem into 4 independent problems and can be handled by parallel computation.
$$
\sum_{i \in \mathcal{U}}y_i \leq \zeta_1, \sum_{i \in \mathcal{L}}y_i \geq \zeta_2;\\ \sum_{i \in \mathcal{U}}y_i \geq \zeta_1,\sum_{i \in \mathcal{L}}y_i \leq \zeta_2;\\ \sum_{i \in \mathcal{U}}y_i \leq \zeta_1, \sum_{i \in \mathcal{L}}y_i \leq \zeta_2
$$
In this project, the predictor is built from exact optimal solutions of single-scenario problems. We compute the expectation of each binary planning decision across scenarios, then use thresholding and rounding to define $\hat{x}$, $\mathcal{U}$, $\mathcal{L}$, $\zeta_1$, and $\zeta_2$. However, this framework is not limited to that source. Any reliable predictor for binary decisions can be used in the same way. In particular, machine-learning based predictors may fit this method well, and may provide better guidance than the current expectation-based rule.

## Metrics

### Optimal gap

During the two-stage decomposition, we track a lower bound and a best incumbent solution value. The lower bound comes from the (MP) objective value. In that model, the recourse terms are represented by cut-based estimators. Since these estimators are valid lower bounds on the true (SP) value, the (MP) objective is a valid lower bound on the true optimal objective value.

The best incumbent solution value is obtained from a feasible design. We fix the current (MP) design, solve the corresponding (SP) MILPs, combine the scenario results by scenario probability, and add the design cost. This gives a feasible objective value for the original problem, so it is an upper bound on the true optimum.

The relative gap between these two values measures how well the current solution is verified. A small gap means that the best feasible solution found so far is close to the best lower bound proved by the decomposition.

### Out of Sample test

The optimal gap is based on the finite scenario set used in the decomposition. It does not fully show how the obtained design performs on new random realizations. For that reason, an out-of-sample test is also useful.

In this test, extra samples are generated from the same disturbance-generation logic. For each sample, the obtained (MP) design is fixed and the corresponding (SP) is solved. The sample objective values are then averaged, with scenario probability used as the outer weight. This gives an estimate of the expected performance of the design under unseen realizations.

This metric is useful even when the decomposition gap has not converged well. A solution may still perform well on many new samples, even if the proof of near-optimality inside the training scenario set is still weak.

# Numerical Experiments

## Experiments & Results

### Experiment basic settings



### Cut Family Comparison

#### Three similar scenarios

The first benchmark uses three similar scenarios. Only classical Benders cuts are added. No warm start or extra cuts are used. This case gives the baseline convergence pattern of the decomposition.

![Benchmark on 3 similar scenarios](</Users/huangjiacheng/Library/Application Support/typora-user-images/image-20260210212443726.png>)

Under the same setting, adding strengthened Benders cuts in every iteration gives a much faster improvement. The lower bound increases earlier, and the incumbent solution also improves faster.

![Strengthened Benders on 3 similar scenarios](</Users/huangjiacheng/Library/Application Support/typora-user-images/image-20260210212535184.png>)

#### Three different scenarios

When the three scenarios are more different, strengthened Benders cuts still give clear improvement. The method remains much stronger than the earlier benchmark setting from the previous project.

![Strengthened Benders on 3 different scenarios](</Users/huangjiacheng/Library/Application Support/typora-user-images/image-20260210212740465.png>)

The figure below is the previous-project reference used for comparison between similar-scenario and different-scenario cases.

![Previous project reference](</Users/huangjiacheng/Library/Application Support/typora-user-images/image-20260210212829928.png>)

Lagrangian cuts were then added only in the last 25 iterations of the three-different-scenario case. The observed effect was small. In many iterations, the generated Lagrangian cut was the same as the strengthened Benders cut, so little extra improvement was obtained.

![Lagrangian cuts in the last 25 iterations](</Users/huangjiacheng/Library/Application Support/typora-user-images/image-20260210213143015.png>)

#### Six scenarios

The same strengthened-Benders setting was also tested on all six scenarios. The method still worked, but the convergence remained much harder than in the smaller cases.

![Strengthened Benders on all 6 scenarios](</Users/huangjiacheng/Library/Application Support/typora-user-images/image-20260211092002558.png>)

#### Symmetry check

Because the original planning data is highly symmetric, an extra check was made by perturbing the installation and hardening costs. The effect on convergence was limited. This suggests that symmetry alone is not the main reason for the slow progress.

Initial symmetric data:

![Classical Benders on initial symmetric data](</Users/huangjiacheng/Library/Application Support/typora-user-images/image-20260210214056046.png>)

Perturbed data:

![Classical Benders on symmetry-broken data](</Users/huangjiacheng/Library/Application Support/typora-user-images/image-20260210214127903.png>)

### Long-Run Behavior and Formulation Change

#### Long-run behavior on three similar scenarios

A longer run on the three-similar-scenario case shows that, even after the incumbent solution is already good, the remaining gap is still hard to close. This confirms that the lower bound is the main difficulty in this problem.

![Long-run behavior on 3 similar scenarios](</Users/huangjiacheng/SDinPS/output/20260221224357_longr_750_ns_correct_sbd/iter_obj_plot.png>)

#### Fixed-rated-power DG formulation

The DG formulation was then revised so that DG size is selected from a fixed set of types in the (MP), rather than being determined separately in each scenario. This change slightly improves the lower-bound growth, but it also increases the total computational burden.

For the three-different-scenario case, the new formulation shows faster early lower-bound improvement than the previous one.

Previous setting:

![Previous setting on different scenarios](</Users/huangjiacheng/Library/Application Support/typora-user-images/image-20260224172412250.png>)

Current setting:

![Current setting on different scenarios](</Users/huangjiacheng/Library/Application Support/typora-user-images/image-20260224172616732.png>)

For the three-similar-scenario case, the same pattern is observed, but the main problem becomes harder to solve.

Previous setting:

![Previous setting on similar scenarios](</Users/huangjiacheng/Library/Application Support/typora-user-images/image-20260224203125973.png>)

Current setting:

![Current setting on similar scenarios](</Users/huangjiacheng/Library/Application Support/typora-user-images/image-20260224202945528.png>)

Overall, the new formulation gives a slightly better lower-bound trend, but the total running time for the same number of iterations increases by about 50% to 200%. In later iterations, the main problem becomes the main bottleneck.

### Separated Cuts and Aggregated Cuts

#### Three similar scenarios

The next test compares two ways of adding cuts. In the separated setting, one cut is added for each scenario. In the aggregated setting, one probability-weighted cut is added for the whole scenario set. For similar scenarios, separated cuts give much better bound and incumbent progress.

Separated cuts:

![Separated cuts on similar scenarios](</Users/huangjiacheng/SDinPS/output/20260223182443_preview_s123_longrun/iter_obj_plot.png>)

Aggregated cuts:

![Aggregated cuts on similar scenarios](</Users/huangjiacheng/Library/Application Support/typora-user-images/image-20260224204537443.png>)

#### Three different scenarios

The same comparison on different scenarios leads to the same conclusion. Separated cuts are stronger, but aggregated cuts reduce running time a lot.

Separated cuts:

![Separated cuts on different scenarios](</Users/huangjiacheng/Library/Application Support/typora-user-images/image-20260224204626145.png>)

Aggregated cuts:

![Aggregated cuts on different scenarios](</Users/huangjiacheng/SDinPS/output/20260224100835_preview_s135_longrun_aggr/iter_obj_plot.png>)

In the three-different-scenario case, aggregated cuts reduced the running time from more than 1.5 hours to about 30 minutes for 200 iterations. The price is a much weaker lower-bound improvement.

### Pre-Training Based Cuts

#### Three different scenarios

The pre-training based cuts were tested on the three-different-scenario case with separated cuts. In this test, the method gave a large improvement in both iteration count and running time.

Improved algorithm:

![Pre-training based cuts on 3 different scenarios](</Users/huangjiacheng/Library/Application Support/typora-user-images/image-20260224214924128.png>)

Benchmark:

![Benchmark on 3 different scenarios](</Users/huangjiacheng/SDinPS/output/20260223194331_preview_s135_longrun/iter_obj_plot.png>)

The improved case finished in about 15 minutes, while the benchmark took about 3 hours and 20 minutes. This is one of the strongest improvements observed in the study.

### Diagnostic Experiments on Cut Quality

#### Cut repetition and redundancy

Further diagnostics were used to understand why the decomposition still slows down in later iterations. Repetition of cuts was common. Repetition happened both within one cut family and across different cut families. One duplicate inequality in the final LP appeared 368 times, including 120 Benders cuts, 173 strengthened Benders cuts, and 75 Lagrangian cuts. In contrast, many CGLP cuts were not repeated exactly within their own family, but most of them were still redundant. Only 13 out of 158 CGLP cuts remained active in the final model.

#### Tightness check near the reference solution

Another diagnostic run compared ordinary iterations with reference iterations built around the known global optimal main-stage solution.

![Tightness and efficiency diagnostic](</Users/huangjiacheng/Library/Application Support/typora-user-images/image-20260324214938547.png>)

When all cuts were generated at the known optimal main-stage solution and the main-stage binary decisions were fixed, the best bound matched the best incumbent value. This shows that the cuts can be tight at that specific point. However, after returning to normal iterations, the bound improved only slightly. This suggests that the global approximation of the (SP) value function is still weak, even if the cuts are tight locally.

### Out-of-Sample Evaluation with the New (MP) Formulation

#### Validity of the out-of-sample test

The out-of-sample test was carried out under the new (MP) formulation. For each scenario, 12 samples were generated. Each sample has 11 time slots. For a fixed (MP) design, the sampled (SP) is solved, and the weighted average over all sampled (SP) objective values is recorded.

To check whether this test is meaningful, four different main solutions were compared in a 50-iteration run on the 6-scenario, 33-bus case. The (MP) in this test uses an (SP) block that aggregates only the scenario dimension.

| Solution | Weighted averaged (SP) optimal value |
| --- | ---: |
| Reference solution from the combined model | 792001 |
| Best solution during decomposition | 791663 |
| Last solution of the run | 1577885 |
| First solution of the run | 1025804 |

The large difference between these solutions shows that the out-of-sample test is informative. The best decomposition solution is very close to the reference solution, while weaker (MP) solutions perform much worse.

#### Effect of the (SP)-block collapsing method

Three (SP)-block collapsing methods were compared by out-of-sample testing.

| Method | Weighted averaged (SP) optimal value of the best solution |
| --- | ---: |
| Only remove scenario index | 791663 |
| Remove time and scenario indices | 1285651 |
| Only remove time index | 1135138 |

Among these three methods, removing only the scenario index gives the best out-of-sample performance. The more aggressive aggregations lose too much information. At the same time, the case that removes only the scenario index is harder to solve, which suggests that the scenario dimension is the main source of complexity in this problem.

#### Effect of the (SP)-objective weight in the (MP)

The weight of the (SP)-block objective term in the (MP) was also tested on a 3-scenario case.

| Weight of (SP) objective term | OOS average (SP) objective of best solution | Gap | Lower bound | Best incumbent |
| --- | ---: | ---: | ---: | ---: |
| 1 | 543299 | 0.44403409247718656 | 2150848 | 3868669 |
| 3 | 539129 | 0.517270059293928 | 2001941 | 4147124 |
| 5 | 536703 | 0.5199338955878305 | 2001941 | 4170136 |
| 7 | 536703 | 0.5199338955878305 | 2001941 | 4170136 |
| Reference solution from the combined model | 402288 | N/A | N/A | 3239346 |

Increasing the (SP)-objective weight improves the out-of-sample value slightly within this experiment, but the gap remains large. This again shows that a good practical design can still be difficult to certify by lower-bound convergence alone.

## Summary on results

### Summary on different families of cuts

Note:

- all cuts below are 'tight' in common sense of optimization, while we are interested in their overall performance on deriving approximation for the (SP), i.e. the ratio of 'improvement of lower bound' to 'numbers of iteration'. We use a intuitive grade scale (1 to 5, 5 is relatively best) to remark the intuitive and relative performance difference.
  - 1: almost no improvement on lower bound
  - 2: improve the lower bound, but very slowly
  - 3: improve the lower bound
  - 4: improve the lower bound significantly faster than other cuts
  - 5: give a good approximation on (SP)
- The exact time of solving involving optimization problem for deriving cuts is data and hardware sensitive. To have a fair but general comparison, we use another intuitive grade scale 1-4 to remark the computational complexity.
  - 1: solving one LP is enough, the easiest to generate.
  - 2: need to solve a MILP which is equivalent or similar to an (SP).
  - 3: need to solve a one LP or MILP, and rely on information that must be obtained by solving a MILP which is equivalent or similar to an (SP).
  - 4: need to solve multiple MILP's where each of them is equivalent or similar to an (SP).

| Cut Family         | Performance | Implementing Difficulty                 | Generating Complexity | Needed Problem to Solve                |
| ------------------ | ----------- | --------------------------------------- | --------------------- | -------------------------------------- |
| Classical Benders  | 2           | Low                                     | 1                     | One Relaxed Subproblem                 |
| Strengthen Benders | 4           | Medium, need a (INMIP), similar to (SP) | 2                     | One (INMIP) (equivalent to Subproblem) |
| Integer Optimal    | 1           | Low                                     | 2                     | One Subproblem                         |
| Lagrangian         | 4           | High, need a heuristic framework        | 4                     | Multiple (INMIP) & LP                  |
| CGLP               | 3           | High, need a new LP and an algorithm    | 3                     | One LP + integer solutions of (SP)     |

### Pretraining-based method

The pretrained cuts added to (MP) at the beginning can reduce the needed number of iterations. The quality of the cut relys on the prediction quality toward the binary decisions in MP. In two-stage stochastic decomposition scheme, the prediction can be obtained by solving single-scenario comprehensive formulation, while the quality of prediction is not always satisfying. 

Pretraining-based cuts can reduce the amount of iterations we need to improve the lower bound before we hit the bottleneck (a level where the lower bound is hard be improved anymore), but cannot help breaking the bottleneck.

### Subproblem block in (MP)

Due to the difficulty on approximating the (SP) in (MP), we add components of (SP) to the (MP) so that (MP) can be solved by more information from (SP). The (SP) block in (MP) has similar effect as pretraining-based method:  reduce the amount of iterations before hitting the bottleneck of lowerbound. Still, it cannot help breaking through the lowerbound.

# Conclusions & Potential Improvement

1. It is hard to obtain a relatively good approximation on the (SP) by any cuts we have implemented.
   - Hundreds of iterations and the generated cuts cannot give a tight lower bound even in small network case.
   - Even after a long-run iteration, adding a strongest cut at our best knowledge (Lagrangian Cut) by the actually global optimal (MP) solution still cannot improve the lower bound.
   - After hitting the bottleneck of lowerbound, the generated cuts are often repetitive, suggesting that more iterations may not generate more effective cuts, thus it is hard to approximate the (SP) by only cuts.
3. Among a problem with a few scenarios (e.g. 6 scenarios) and a known global optimal (MP) solution, the best solution we obtained still has a large optimal gap, and a difference to the known optimal solution. However, it can be close to the global optimal in out of sample (OOS) test, while still dominate other solutions obtained during the iteration.
   - Numerical experiments on the state-of-the-art cuts suggests that:
     - try achieving a good lower bound by optimal cuts is not practical.
     - when using OOS test as evaluation, proper cuts can still improve the quality of solution.
   - We can consider using a hybrid method combining pretraining-based cuts, efficient optimality cuts, heuristic algorithm such as genetic algorithm, and OOS test as evaluating metric.

# Reference

Zou, J., Ahmed, S., & Sun, X. (2019). *Stochastic dual dynamic integer programming.* *Mathematical Programming, 178*(1-2), 329-356. https://doi.org/10.1007/s10107-018-1249-5 

Angulo, G., Ahmed, S., & Dey, S. S. (2016). *Improving the integer L-shaped method.* *INFORMS Journal on Computing, 28*(3), 483-499. https://doi.org/10.1287/ijoc.2016.0695 

Rahmaniani, R., Ahmed, S., Crainic, T. G., Gendreau, M., & Rei, W. (2020).*The Benders Dual Decomposition Method. Operations Research, 68(3)*, 878–895. https://doi-org.prox.lib.ncsu.edu/10.1287/opre.2019.1892

Chen, Y., Gao, W., Ge, D., & Ye, Y. (2023). *Pre-trained mixed integer optimization through multi-variable cardinality branching*. *arXiv preprint arXiv:2305.12352*.

