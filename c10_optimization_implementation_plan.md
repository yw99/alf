# High-Level Implementation Plan for Optimizing Eq. C.10

This note summarizes a practical implementation plan for the continuous sampling-distribution optimization problem in Eq. C.10.

## 1. Problem

We optimize a probability distribution \(p \in \Delta_N\) over buffer samples:

\[
\min_{p \in \Delta_N} F(p)
=
G(p) + \beta H(p),
\]

where

\[
G(p)
=
\operatorname{tr}\!\left(A_\pi \Sigma_p^{-1} M_p \Sigma_p^{-1}\right),
\qquad
H(p)
=
\operatorname{tr}\!\left(A_\pi \Sigma_p^{-1}\right),
\]

with

\[
\Sigma_p = \sum_{i=1}^N p_i \phi_i \phi_i^\top,
\qquad
M_p = \sum_{i=1}^N p_i^2 \phi_i \phi_i^\top,
\qquad
\beta = \frac{1}{K(1-\gamma)}.
\]

Interpretation:

- \(G(p)\): LSTD-layer variance term.
- \(H(p)\): TD-to-LSTD / coverage term.
- \(\beta\): trade-off knob.
  - Small \(\beta\): use more of the full buffer, close to uniform.
  - Large \(\beta\): approach the L-optimal design.

The optimization is smooth on the interior of the simplex, but **not convex in general**, because \(G(p)\) is nonconvex.

---

## 2. Recommended Solver

Use a **multi-start hybrid projected-gradient + Frank–Wolfe solver**.

The main solver should combine:

1. **Projected gradient descent (PGD)** on the simplex.
2. **Frank–Wolfe vertex-escape steps** when the FW direction gives certified descent.
3. **Warm starts** from:
   - the previous epoch's solution \(p^{(k-1)}\),
   - the uniform distribution,
   - the Fedorov–Wynn solution for the coverage term \(H\),
   - several random simplex initializations.

The Fedorov–Wynn update should be treated only as a warm start or endpoint solver for the coverage-dominated regime. It does **not** optimize the full nonconvex objective \(G+\beta H\).

---

## 3. Quantities to Compute

Let \(X \in \mathbb{R}^{N \times d}\) have rows \(\phi_i^\top\).

Given \(p\), compute:

\[
\Sigma_p = X^\top \operatorname{diag}(p)X,
\]

\[
M_p = X^\top \operatorname{diag}(p^2)X.
\]

For numerical stability, use

\[
\Sigma_p^{(\rho)} = \Sigma_p + \rho I_d
\]

with small ridge \(\rho > 0\), especially when \(p\) is sparse.

Let

\[
B = \left(\Sigma_p + \rho I_d\right)^{-1}.
\]

Then

\[
H(p) = \operatorname{tr}(A_\pi B),
\]

\[
G(p) = \operatorname{tr}(A_\pi B M_p B).
\]

Define leverage-like scores

\[
\ell_i(p)
=
\phi_i^\top B A_\pi B \phi_i,
\]

and

\[
h_i(p)
=
\phi_i^\top B A_\pi B M_p B \phi_i.
\]

Then the exact gradient is

\[
\nabla_i F(p)
=
(2p_i-\beta)\ell_i(p) - 2h_i(p).
\]

---

## 4. Stopping Criterion: Frank–Wolfe / KKT Gap

Use the Frank–Wolfe gap as the main stationarity certificate.

The identity

\[
\langle p, \nabla F(p) \rangle = -\beta H(p)
\]

implies

\[
g_{\mathrm{FW}}(p)
=
-\beta H(p) - \min_j \nabla_j F(p).
\]

Properties:

- \(g_{\mathrm{FW}}(p) \ge 0\).
- \(g_{\mathrm{FW}}(p)=0\) iff \(p\) is a first-order KKT point.
- If \(g_{\mathrm{FW}}(p)>0\), then the vertex direction
  \[
  d = e_{j^\star} - p,
  \qquad
  j^\star = \arg\min_j \nabla_j F(p),
  \]
  is a feasible descent direction.

Stop when

\[
g_{\mathrm{FW}}(p) \le \varepsilon_{\mathrm{KKT}}.
\]

---

## 5. Main Algorithm

```text
Input:
    X: feature matrix, shape (N, d)
    A: target covariance A_pi, shape (d, d)
    beta: trade-off parameter
    rho: ridge parameter
    max_iter
    tol

Initialize candidate starts:
    p_uniform = (1/N, ..., 1/N)
    p_prev    = previous epoch's solution, if available
    p_FW      = Fedorov-Wynn solution for H, optional
    p_random  = several random simplex points

For each initialization p:

    for t = 1, ..., max_iter:

        1. Compute Sigma = X.T @ diag(p) @ X + rho * I
        2. Compute B = inv(Sigma)
        3. Compute M = X.T @ diag(p**2) @ X

        4. Compute objective:
               H = tr(A @ B)
               G = tr(A @ B @ M @ B)
               F = G + beta * H

        5. Compute scores:
               ell_i = phi_i.T @ B @ A @ B @ phi_i
               h_i   = phi_i.T @ B @ A @ B @ M @ B @ phi_i

        6. Compute gradient:
               grad_i = (2*p_i - beta) * ell_i - 2*h_i

        7. Compute FW/KKT gap:
               j_star = argmin_j grad_j
               gap_FW = -beta * H - grad[j_star]

           If gap_FW <= tol:
               break

        8. Candidate PGD step:
               p_pgd = Proj_Delta(p - eta * grad)
           Use backtracking line search to ensure F decreases.

        9. Candidate FW step:
               d_fw = e_{j_star} - p
               p_fw = p + alpha * d_fw
           Use line search over alpha in [0, 1].

       10. Pick whichever candidate gives the smaller objective.

Return:
    best p among all initializations.
```

---

## 6. Projection Onto the Simplex

The simplex projection solves

\[
\operatorname{Proj}_{\Delta_N}(z)
=
\arg\min_{p \in \Delta_N} \|p-z\|_2^2.
\]

Standard implementation:

```python
def project_simplex(z):
    # Projects z onto {p >= 0, sum p = 1}
    u = np.sort(z)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(z)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(z - theta, 0.0)
```

---

## 7. Vectorized Computation

Avoid looping over \(i\) when computing \(\ell_i\) and \(h_i\).

Let

```python
C_ell = B @ A @ B
C_h   = B @ A @ B @ M @ B
```

Then

```python
ell = np.einsum("nd,dk,nk->n", X, C_ell, X)
h   = np.einsum("nd,dk,nk->n", X, C_h, X)
```

The objective can be computed as

```python
H = np.trace(A @ B)
G = np.trace(A @ B @ M @ B)
F = G + beta * H
```

---

## 8. Complexity

Per iteration:

- Forming \(\Sigma_p\): \(O(Nd^2)\).
- Forming \(M_p\): \(O(Nd^2)\).
- Inverting \(\Sigma_p\): \(O(d^3)\).
- Computing all \(\ell_i, h_i\): \(O(Nd^2)\).

Total per iteration:

\[
O(Nd^2 + d^3).
\]

This is feasible when \(d\) is moderate and \(N\) is large.

---

## 9. Numerical Safeguards

Use the following safeguards:

1. **Ridge regularization**
   \[
   \Sigma_p \leftarrow \Sigma_p + \rho I.
   \]

2. **Lower mass floor during early iterations**
   Optionally use
   \[
   p_i \ge p_{\min}
   \]
   for early iterations, then relax the floor.

3. **Line search**
   Always require sufficient decrease in \(F\).

4. **Multiple restarts**
   Since the problem is nonconvex, keep the best solution over several initializations.

5. **Check conditioning**
   Monitor
   \[
   \lambda_{\min}(\Sigma_p)
   \]
   or the condition number of \(\Sigma_p\).

6. **Check FW gap**
   Do not rely only on \(\|p_{t+1}-p_t\|\); use \(g_{\mathrm{FW}}\).

---

## 10. Warm-Start Strategy Across Actor Epochs

In the actor loop, features \(\phi_i\) drift as the reference policy changes.

Use the previous optimizer output as the next initialization:

\[
p^{(k)}_{\mathrm{init}} = p^{(k-1)}.
\]

This makes the continuous solver attractive compared with recomputing a hard subset from scratch.

Recommended initialization set at epoch \(k\):

```text
starts = [
    p_previous_epoch,
    uniform_distribution,
    fedorov_wynn_for_current_features,
    random_dirichlet_start_1,
    random_dirichlet_start_2,
    ...
]
```

---

## 11. Practical Solver Choice

A good default implementation is:

```text
Multi-start hybrid PGD/FW

- Use Fedorov-Wynn only as a warm start.
- Use PGD as the main local optimizer.
- Use FW vertex steps as boundary-escape directions.
- Use FW/KKT gap as the stopping criterion.
- Keep the best objective over restarts.
```

This is more reliable than plain PGD because the nonconvexity often appears near the sparse boundary, where a vertex direction can provide a clear descent certificate.

---

## 12. Minimal Python Skeleton

```python
import numpy as np

def project_simplex(z):
    u = np.sort(z)[::-1]
    cssv = np.cumsum(u)
    idx = np.arange(1, len(z) + 1)
    rho = np.nonzero(u * idx > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(z - theta, 0.0)

def objective_and_grad(p, X, A, beta, ridge=1e-6):
    N, d = X.shape
    Sigma = X.T @ (p[:, None] * X) + ridge * np.eye(d)
    B = np.linalg.inv(Sigma)

    M = X.T @ ((p ** 2)[:, None] * X)

    H = np.trace(A @ B)
    G = np.trace(A @ B @ M @ B)
    F = G + beta * H

    C_ell = B @ A @ B
    C_h = B @ A @ B @ M @ B

    ell = np.einsum("nd,dk,nk->n", X, C_ell, X)
    h = np.einsum("nd,dk,nk->n", X, C_h, X)

    grad = (2 * p - beta) * ell - 2 * h

    j_star = np.argmin(grad)
    gap_fw = -beta * H - grad[j_star]

    return F, grad, gap_fw, j_star

def optimize_c10(
    X,
    A,
    beta,
    starts,
    ridge=1e-6,
    max_iter=1000,
    tol=1e-6,
    eta0=1.0,
):
    best_p = None
    best_F = np.inf

    N = X.shape[0]

    for p0 in starts:
        p = project_simplex(np.asarray(p0, dtype=float))

        for _ in range(max_iter):
            F, grad, gap_fw, j_star = objective_and_grad(p, X, A, beta, ridge)

            if gap_fw <= tol:
                break

            # PGD candidate
            eta = eta0
            p_pgd = None
            F_pgd = np.inf

            for _ in range(30):
                cand = project_simplex(p - eta * grad)
                F_cand, _, _, _ = objective_and_grad(cand, X, A, beta, ridge)
                if F_cand <= F:
                    p_pgd = cand
                    F_pgd = F_cand
                    break
                eta *= 0.5

            # FW candidate
            e = np.zeros(N)
            e[j_star] = 1.0
            d_fw = e - p

            alpha = 1.0
            p_fw = None
            F_fw = np.inf

            for _ in range(30):
                cand = p + alpha * d_fw
                F_cand, _, _, _ = objective_and_grad(cand, X, A, beta, ridge)
                if F_cand <= F:
                    p_fw = cand
                    F_fw = F_cand
                    break
                alpha *= 0.5

            # Choose better candidate
            if F_pgd <= F_fw:
                if p_pgd is None:
                    break
                p = p_pgd
            else:
                if p_fw is None:
                    break
                p = p_fw

        F_final, _, _, _ = objective_and_grad(p, X, A, beta, ridge)
        if F_final < best_F:
            best_F = F_final
            best_p = p.copy()

    return best_p, best_F
```

---

## 13. Paper-Writing Summary

A concise description for the paper:

> We solve the continuous selection problem by a multi-start first-order method on the simplex. Since the objective is smooth but nonconvex, we use the exact gradient from Lemma 14, projected-gradient steps with line search, and Frank–Wolfe vertex-escape steps certified by the gap in Proposition 3. The Fedorov–Wynn update for the convex coverage layer is used as a warm start, rather than as a solver for the full objective. We terminate when the Frank–Wolfe/KKT gap falls below tolerance and keep the best solution across initializations.

---

## 14. Main Takeaway

The optimization in Eq. C.10 should be implemented as a **nonconvex smooth optimization over the simplex**, not as a purely convex design problem.

The practical recipe is:

\[
\boxed{
\text{Fedorov-Wynn warm start}
\;+\;
\text{multi-start PGD}
\;+\;
\text{FW boundary escape}
\;+\;
\text{FW/KKT gap stopping}
}
\]
