# Reference note: Example 2 in Section 3.1 of *Improving the integer L-shaped method*

This note summarizes **Example 2** from **Section 3.1 (Construction of CGLP)** of the paper *Improving the integer L-shaped method*. It is intentionally short and is meant to be a **reference solution** for checking code that implements the set-recursion / forbidden-vertices construction used in the CGLP.

## 1. Test example

The example is a **set-construction** example, not a full optimization problem.

The data are:

- Dimension:
  $$
  n = 3.
  $$

- Known binary vectors:
  $$
  V = \left\{
  \begin{pmatrix}0\\1\\0\end{pmatrix},
  \begin{pmatrix}0\\1\\1\end{pmatrix},
  \begin{pmatrix}1\\0\\1\end{pmatrix}
  \right\}.
  $$

For the recursion at **$j=2$**, the paper uses the projection of $V$ onto the first two coordinates:

$$
V^2 = \left\{
\begin{pmatrix}0\\1\end{pmatrix},
\begin{pmatrix}1\\0\end{pmatrix}
\right\}.
$$

Hence,

$$
\{0,1\}^2 \setminus V^2
= \left\{
\begin{pmatrix}0\\0\end{pmatrix},
\begin{pmatrix}1\\1\end{pmatrix}
\right\}.
$$

## 2. What should be computed

The example illustrates the recursion

$$
\{0,1\}^{j+1} \setminus V^{j+1}
=
\left[(\{0,1\}^{j} \setminus V^{j}) \times \{0,1\}\right] \cup \widehat V^{j+1}.
$$

For this example, the target is to verify the case **$j=2$**.

### Step 1: Expand the complement of the projection

From

$$
\{0,1\}^2 \setminus V^2
= \left\{
\begin{pmatrix}0\\0\end{pmatrix},
\begin{pmatrix}1\\1\end{pmatrix}
\right\},
$$

we obtain

$$
\left[(\{0,1\}^2 \setminus V^2) \times \{0,1\}\right]
=
\left\{
\begin{pmatrix}0\\0\\0\end{pmatrix},
\begin{pmatrix}0\\0\\1\end{pmatrix},
\begin{pmatrix}1\\1\\0\end{pmatrix},
\begin{pmatrix}1\\1\\1\end{pmatrix}
\right\}.
$$

### Step 2: Compute $\widehat V^3$

First,

$$
V^2 \times \{0,1\}
=
\left\{
\begin{pmatrix}0\\1\\0\end{pmatrix},
\begin{pmatrix}0\\1\\1\end{pmatrix},
\begin{pmatrix}1\\0\\0\end{pmatrix},
\begin{pmatrix}1\\0\\1\end{pmatrix}
\right\}.
$$

Since

$$
V = \left\{
\begin{pmatrix}0\\1\\0\end{pmatrix},
\begin{pmatrix}0\\1\\1\end{pmatrix},
\begin{pmatrix}1\\0\\1\end{pmatrix}
\right\},
$$

we get

$$
\widehat V^3
= [V^2 \times \{0,1\}] \setminus V
= \left\{
\begin{pmatrix}1\\0\\0\end{pmatrix}
\right\}.
$$

## 3. Exact reference result

The full complement of $V$ in $\{0,1\}^3$ is

$$
\{0,1\}^3 \setminus V
=
\left\{
\begin{pmatrix}0\\0\\0\end{pmatrix},
\begin{pmatrix}0\\0\\1\end{pmatrix},
\begin{pmatrix}1\\0\\0\end{pmatrix},
\begin{pmatrix}1\\1\\0\end{pmatrix},
\begin{pmatrix}1\\1\\1\end{pmatrix}
\right\}.
$$

The recursion identity should therefore hold exactly:

$$
\{0,1\}^3 \setminus V
=
\left[(\{0,1\}^2 \setminus V^2) \times \{0,1\}\right] \cup \widehat V^3.
$$

Substituting the explicit sets,

$$
\left\{
\begin{pmatrix}0\\0\\0\end{pmatrix},
\begin{pmatrix}0\\0\\1\end{pmatrix},
\begin{pmatrix}1\\0\\0\end{pmatrix},
\begin{pmatrix}1\\1\\0\end{pmatrix},
\begin{pmatrix}1\\1\\1\end{pmatrix}
\right\}
=
\left\{
\begin{pmatrix}0\\0\\0\end{pmatrix},
\begin{pmatrix}0\\0\\1\end{pmatrix},
\begin{pmatrix}1\\1\\0\end{pmatrix},
\begin{pmatrix}1\\1\\1\end{pmatrix}
\right\}
\cup
\left\{
\begin{pmatrix}1\\0\\0\end{pmatrix}
\right\}.
$$

## 4. Additional derived sets (useful for implementation checks)

These are not all written explicitly in the paper, but they follow directly from the definitions and are useful for validating code.

### Projections

$$
V^1 = \left\{ \begin{pmatrix}0\end{pmatrix}, \begin{pmatrix}1\end{pmatrix} \right\},
\qquad
V^2 = \left\{ \begin{pmatrix}0\\1\end{pmatrix}, \begin{pmatrix}1\\0\end{pmatrix} \right\},
\qquad
V^3 = V.
$$

### Forbidden-prefix sets

$$
\widehat V^1 = \{0,1\} \setminus V^1 = \varnothing,
$$

$$
\widehat V^2 = [V^1 \times \{0,1\}] \setminus V^2
= \left\{
\begin{pmatrix}0\\0\end{pmatrix},
\begin{pmatrix}1\\1\end{pmatrix}
\right\},
$$

$$
\widehat V^3 = \left\{
\begin{pmatrix}1\\0\\0\end{pmatrix}
\right\}.
$$

### Embedded sets $W_i = \widehat V^i \times \{0\}^{3-i}$

$$
W_1 = \varnothing,
$$

$$
W_2 = \left\{
\begin{pmatrix}0\\0\\0\end{pmatrix},
\begin{pmatrix}1\\1\\0\end{pmatrix}
\right\},
$$

$$
W_3 = \left\{
\begin{pmatrix}1\\0\\0\end{pmatrix}
\right\}.
$$

## 5. Final reference outputs

For quick checking, the expected outputs are:

- **Input set**
  $$
  V = \{(0,1,0), (0,1,1), (1,0,1)\}.
  $$

- **Projection at $j=2$**
  $$
  V^2 = \{(0,1), (1,0)\}.
  $$

- **Complement of the projection**
  $$
  \{0,1\}^2 \setminus V^2 = \{(0,0), (1,1)\}.
  $$

- **Lifted complement term**
  $$
  (\{0,1\}^2 \setminus V^2) \times \{0,1\}
  = \{(0,0,0), (0,0,1), (1,1,0), (1,1,1)\}.
  $$

- **Missing vector inside $V^2 \times \{0,1\}$**
  $$
  \widehat V^3 = \{(1,0,0)\}.
  $$

- **Final complement**
  $$
  \{0,1\}^3 \setminus V
  = \{(0,0,0), (0,0,1), (1,0,0), (1,1,0), (1,1,1)\}.
  $$

- **Reference identity to verify**
  $$
  \{0,1\}^3 \setminus V
  = [ (\{0,1\}^2 \setminus V^2) \times \{0,1\} ] \cup \widehat V^3.
  $$

## Source

Gustavo Angulo, Shabbir Ahmed, and Santanu S. Dey, *Improving the integer L-shaped method*, Section 3.1, Example 2.

# Other examples

## 1. Small exact examples for testing CGLP-based cut generation

This section adds **code-testing examples** to the design note.

### Important testing caveat

The paper's CGLP in Proposition 4 is a **feasibility LP**. Therefore, for a fixed input state 
\((V, V^i, \hat V^i, W_i, x^t, Q(x^t))\), there can be **more than one valid cut** 
\(\alpha x + \theta \ge -\eta\) that satisfies the CGLP.

Because of that, there are two different notions of a "correct" output:

1. **Paper-faithful correctness**: the returned cut is CGLP-feasible, is tight at the current point \(x^t\), and is valid for the relaxation built from the current exact-value set \(V\).
2. **Regression-test exactness**: the code returns one predetermined, reproducible cut.

For exact coefficient-by-coefficient regression tests, the examples below specify a **reference facet cut**. 
That cut is a valid CGLP-based cut and is the one that should be used as the expected answer in a unit test.
If an implementation solves only the paper's feasibility CGLP with no tie-breaking rule, it may return a different but still valid cut.

In all examples below, we use the simplest first-stage master:

$$
\min \theta \qquad \text{s.t. } x \in \{0,1\}^n,
$$

so there are no additional first-stage constraints \((A,C,b)\), no \(z\)-variables, and the only always-present affine lower bound is

$$
\theta \ge L, \qquad L = 0.
$$

Equivalently, in the paper's notation, the lower-bound block \((\Pi,\pi_0)\) consists only of

$$
-\theta \le 0.
$$

---

### 1.1 Example A: 2-variable test with one missing first-coordinate prefix

#### Two-stage MILP

First-stage variables:

$$
x = (x_1,x_2) \in \{0,1\}^2.
$$

Second-stage MILP defining \(Q(x)\):

$$
\begin{aligned}
Q(x) := \min \;& 2y_1 + 2y_2 \\
\text{s.t. } & y_1 \ge x_1, \\
& y_2 \ge x_1 + x_2 - 1, \\
& y_1,y_2 \in \{0,1\}.
\end{aligned}
$$

This gives the exact value table

$$
Q(0,0)=0, \qquad Q(0,1)=0, \qquad Q(1,0)=2, \qquad Q(1,1)=4.
$$

#### Exact-value set and target point

Use

$$
V = \{(1,0),(1,1)\},
\qquad x^t = (1,1).
$$

So the exact values currently known to the algorithm are

$$
Q(1,0)=2,
\qquad
Q(1,1)=4.
$$

#### Derived sets

Projected sets:

$$
V^1 = \{(1)\},
\qquad
V^2 = \{(1,0),(1,1)\}.
$$

Forbidden-prefix sets:

$$
\hat V^1 = \{(0)\},
\qquad
\hat V^2 = [V^1 \times \{0,1\}] \setminus V^2 = \varnothing.
$$

Embedded sets:

$$
W_1 = \{(0,0)\},
\qquad
W_2 = \varnothing.
$$

So the forbidden-vertices block contributes only one row of type (11), namely the row associated with

$$
w = (0,0) \in W_1.
$$

#### Reference facet cut

A valid CGLP-based reference cut is

$$
\boxed{\theta \ge -2 + 4x_1 + 2x_2.}
$$

Equivalently, in Proposition 4 form,

$$
\boxed{-4x_1 - 2x_2 + \theta \ge 2.}
$$

#### Why this is the correct reference cut

The affine function

$$
g(x) = -2 + 4x_1 + 2x_2
$$

satisfies

$$
g(0,1)=0 = L,
\qquad
g(1,0)=2 = Q(1,0),
\qquad
g(1,1)=4 = Q(1,1).
$$

Thus it is tight at:

- the unseen point \((0,1)\) at the lower bound \(L=0\),
- the known exact point \((1,0)\), and
- the current point \((1,1)\).

It is also valid at the remaining unseen point:

$$
g(0,0) = -2 \le 0 = L.
$$

So this cut is valid for the current relaxation and cuts off every candidate with

$$
\theta < -2 + 4x_1 + 2x_2.
$$

#### Minimal regression checks

A correct implementation should reproduce at least the following facts:

- \(V^1=\{(1)\}\), \(\hat V^1=\{(0)\}\), \(W_1=\{(0,0)\}\),

- \(\hat V^2=\varnothing\), \(W_2=\varnothing\),

- the target equality row is built with \(x^t=(1,1)\) and \(Q(x^t)=4\),

- the reference cut is
  $$
  \theta \ge -2 + 4x_1 + 2x_2.
  $$

---

### 1.2 Example B: 2-variable test with a nonempty \(W_2\)

#### Two-stage MILP

First-stage variables:

$$
x = (x_1,x_2) \in \{0,1\}^2.
$$

Second-stage MILP defining \(Q(x)\):

$$
\begin{aligned}
Q(x) := \min \;& 3y_1 + 2y_2 \\
\text{s.t. } & y_1 \ge x_2, \\
& y_2 \ge x_1 + x_2 - 1, \\
& y_1,y_2 \in \{0,1\}.
\end{aligned}
$$

This gives the exact value table

$$
Q(0,0)=0, \qquad Q(1,0)=0, \qquad Q(0,1)=3, \qquad Q(1,1)=5.
$$

#### Exact-value set and target point

Use

$$
V = \{(0,1),(1,1)\},
\qquad x^t = (1,1).
$$

So the exact values currently known are

$$
Q(0,1)=3,
\qquad
Q(1,1)=5.
$$

#### Derived sets

Projected sets:

$$
V^1 = \{(0),(1)\},
\qquad
V^2 = \{(0,1),(1,1)\}.
$$

Forbidden-prefix sets:

$$
\hat V^1 = \varnothing,
$$

$$
\hat V^2 = [V^1 \times \{0,1\}] \setminus V^2 = \{(0,0),(1,0)\}.
$$

Embedded sets:

$$
W_1 = \varnothing,
\qquad
W_2 = \{(0,0),(1,0)\}.
$$

So the CGLP row family (11) contains exactly the two rows induced by

$$
(0,0),\ (1,0) \in W_2.
$$

#### Reference facet cut

A valid CGLP-based reference cut is

$$
\boxed{\theta \ge -2 + 2x_1 + 5x_2.}
$$

Equivalently,

$$
\boxed{-2x_1 - 5x_2 + \theta \ge 2.}
$$

#### Why this is the correct reference cut

Define

$$
g(x) = -2 + 2x_1 + 5x_2.
$$

Then

$$
g(1,0)=0 = L,
\qquad
g(0,1)=3 = Q(0,1),
\qquad
g(1,1)=5 = Q(1,1).
$$

So it is tight at:

- the unseen point \((1,0)\) at the lower bound,
- the known exact point \((0,1)\), and
- the current point \((1,1)\).

It is also valid at the remaining unseen point:

$$
g(0,0) = -2 \le 0 = L.
$$

#### Minimal regression checks

A correct implementation should reproduce at least the following facts:

- \(\hat V^1=\varnothing\),

- \(\hat V^2=\{(0,0),(1,0)\}\),

- \(W_2=\{(0,0),(1,0)\}\),

- the target equality row is built with \(x^t=(1,1)\) and \(Q(x^t)=5\),

- the reference cut is
  $$
  \theta \ge -2 + 2x_1 + 5x_2.
  $$

---

### 1.3 Example C: 3-variable test using the same toy-set structure as the paper note

This example uses **exactly the same set-recursion structure** as the toy note extracted from the paper, namely

$$
V = \{(0,1,0),(0,1,1),(1,0,1)\}.
$$

So it is the best example in this section for checking both:

1. the recursive construction of \(V^i\), \(\hat V^i\), and \(W_i\), and
2. the correctness of the generated CGLP cut.

#### Two-stage MILP

First-stage variables:

$$
x = (x_1,x_2,x_3) \in \{0,1\}^3.
$$

Second-stage MILP defining \(Q(x)\):

$$
\begin{aligned}
Q(x) := \min \;& y_1 + 4y_2 \\
\text{s.t. } & y_1 \ge x_2, \\
& y_2 \ge x_1 + x_3 - 1, \\
& y_1,y_2 \in \{0,1\}.
\end{aligned}
$$

This gives the exact value table

$$
\begin{array}{c|c}
x & Q(x) \\
\hline
(0,0,0) & 0 \\
(0,0,1) & 0 \\
(0,1,0) & 1 \\
(0,1,1) & 1 \\
(1,0,0) & 0 \\
(1,0,1) & 4 \\
(1,1,0) & 1 \\
(1,1,1) & 5
\end{array}
$$

#### Exact-value set and target point

Use

$$
V = \{(0,1,0),(0,1,1),(1,0,1)\},
\qquad x^t = (1,0,1).
$$

So the exact values currently known are

$$
Q(0,1,0)=1,
\qquad
Q(0,1,1)=1,
\qquad
Q(1,0,1)=4.
$$

#### Derived sets

Projected sets:

$$
V^1 = \{(0),(1)\},
$$

$$
V^2 = \{(0,1),(1,0)\},
$$

$$
V^3 = V.
$$

Forbidden-prefix sets:

$$
\hat V^1 = \varnothing,
$$

$$
\hat V^2 = [V^1 \times \{0,1\}] \setminus V^2 = \{(0,0),(1,1)\},
$$

$$
\hat V^3 = [V^2 \times \{0,1\}] \setminus V^3 = \{(1,0,0)\}.
$$

Embedded sets:

$$
W_1 = \varnothing,
$$

$$
W_2 = \{(0,0,0),(1,1,0)\},
$$

$$
W_3 = \{(1,0,0)\}.
$$

These are exactly the same \(V^i\), \(\hat V^i\), and \(W_i\) values as in the paper-based toy note.

#### Reference facet cut

A valid CGLP-based reference cut is

$$
\boxed{\theta \ge -4 + 4x_1 - 4x_2 + 4x_3.}
$$

Equivalently,

$$
\boxed{-4x_1 + 4x_2 - 4x_3 + \theta \ge 4.}
$$

#### Why this is the correct reference cut

Define

$$
g(x) = -4 + 4x_1 - 4x_2 + 4x_3.
$$

Then

$$
g(1,0,1)=4 = Q(1,0,1),
$$

and at the unseen points

$$
g(0,0,1)=0 = L,
\qquad
g(1,0,0)=0 = L,
\qquad
g(1,1,1)=0 = L.
$$

So the cut is tight at the current exact point and at three unseen points that are currently only protected by the lower bound \(L=0\).

It remains valid at all other binary points:

$$
g(0,0,0)=-4 \le 0,
$$

$$
g(0,1,0)=-8 \le Q(0,1,0)=1,
$$

$$
g(0,1,1)=-4 \le Q(0,1,1)=1,
$$

$$
g(1,1,0)=-4 \le 0.
$$

This example is especially useful because the cut is generated from a state where:

- \(W_2\) is nonempty,
- \(W_3\) is nonempty,
- the current target point is in \(V\), and
- some known exact points are **not** active on the final cut.

That last feature is important: a correct implementation should **not** assume that every old exact point becomes tight in the new cut.

#### Minimal regression checks

A correct implementation should reproduce at least the following facts:

- \(V^2=\{(0,1),(1,0)\}\),

- \(\hat V^2=\{(0,0),(1,1)\}\),

- \(\hat V^3=\{(1,0,0)\}\),

- \(W_2=\{(0,0,0),(1,1,0)\}\),

- \(W_3=\{(1,0,0)\}\),

- the target equality row is built with \(x^t=(1,0,1)\) and \(Q(x^t)=4\),

- the reference cut is
  $$
  \theta \ge -4 + 4x_1 - 4x_2 + 4x_3.
  $$

---

### 1.4 Recommended use of these examples in code tests

For practical automated testing, the most robust approach is:

1. **Check the set recursion exactly.**
   - Verify that \(V^i\), \(\hat V^i\), and \(W_i\) match the values listed above.

2. **Check the CGLP state exactly.**
   - Verify that the correct row family \(w_l^i\sigma + \rho_i \ge 0\) is present.
   - Verify that the target equality row uses the current \(x^t\) and \(Q(x^t)\).

3. **Check the generated cut functionally.**
   - It must be tight at the current point \(x^t\).
   - It must be valid for all exact points in \(V\).
   - It must be valid at the unseen binary points under the lower bound \(L=0\).

4. **If you want coefficient-by-coefficient exact regression**, use the reference facet cut given above as the expected answer.

This separates two things that are easy to confuse:

- **paper-faithful CGLP correctness**, and
- **deterministic exact-cut regression testing**.
