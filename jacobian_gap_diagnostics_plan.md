# Jacobian + Gap Diagnostics Harness Plan (JAX)
## JVP/VJP Power Iteration, Subspace Drift, and Davis–Kahan Predictors

**Doc:** `jacobian_gap_diagnostics_plan.md`  
**Status:** v0.2 (implementation plan)  
**Target stack:** JAX (pure), Flax/Equinox model, Optax, W&B optional  
**Core idea:** estimate Jacobian spectral norm and track spectral-gap-stabilized memory subspaces without materializing Jacobians.

---

## 0. Outputs (what the harness produces)

For a batch of prompts / sequences, and selected layers \(\ell\in\mathcal{L}\):

1. \(\widehat{\|J_\ell\|_2}\) — estimated Jacobian spectral norm (per sample and averaged)  
2. Subspace bases \(U_\ell\) — top-\(k\) eigenspaces of chosen operator (covariance, sensitivity proxy, etc.)  
3. Principal-angle drift \(d_\ell=\|\sin\Theta(U_\ell, U_{\ell+1})\|_2\)  
4. Eigengap \(\Delta_\ell = \lambda_k - \lambda_{k+1}\)  
5. Davis–Kahan predictor: compare observed drift to \(\|E\|_2/\Delta_\ell\) proxy  
6. Optional: transient growth proxy via \(\|J_\ell\|_2\) vs \(\rho(J_\ell)\) approximations (non-normality indicator)

All metrics should be loggable to W&B and saved as JSON/NPZ.

---

## 1. Harness structure

### 1.1 Modules
- `probe_runner.py` / `probe_runner.py`-equivalent: runs model, collects intermediate activations at probe layers.
- `jacobian_norm.py`: estimates \(\|J_\ell\|_2\) using power iteration with JVP/VJP.
- `subspace.py`: computes eigenpairs of covariance-like operators; principal angles.
- `gap.py`: eigengap computation and gap dashboards.
- `dk_predictor.py`: Davis–Kahan predictor metrics and plots.
- `bench_write_keep_read.py`: synthetic benchmark and perturbation suite.

### 1.2 Probe layers
Avoid probing every layer initially. Choose:
- early / mid / late layers,
- plus any “designated corridor” layers.

Allow a config like:
```yaml
probe_layers: [0, 2, 4, 8, 12, 16, 20, 24]
k_eigs: 16
power_iters: 20
num_probe_samples: 32
```

---

## 2. Jacobian spectral norm estimation (JAX)

We estimate \(\|J\|_2 = \sqrt{\lambda_{\max}(J^\top J)}\) via power iteration.

### 2.1 Setup
Let `f(x)` be the layer map (or block map) at fixed parameters and fixed conditioning inputs (e.g., fixed attention mask and positions).

We need:
- JVP: \(Jv\) using `jax.jvp(f, (x,), (v,))`
- VJP: \(J^\top w\) using `jax.vjp(f, x)` then `vjp_fun(w)`

### 2.2 Algorithm
Initialize random \(v_0\) with \(\|v_0\|=1\). For \(t=0..T-1\):
1. \(w_t = J v_t\)  (JVP)
2. \(u_t = J^\top w_t\) (VJP)
3. \(v_{t+1} = u_t/\|u_t\|\)
Estimate:
\[
\widehat{\|J\|_2} \approx \sqrt{v_t^\top u_t}.
\]

### 2.3 Practical cautions
- Use float32 for speed; consider float64 for validation runs.
- Normalize vectors with epsilon to avoid division issues.
- Ensure `f` is `jit`-compiled; keep shapes static.
- For large `nd`, operate on flattened vectors and reshape inside `f` only when needed.

### 2.4 Batched estimation
Use `vmap` over samples:
- sample a batch of inputs \(x\),
- run power iteration per sample, possibly with shared initial seeds.

---

## 3. Choosing the operator for subspaces and gaps

### Option A: Covariance subspaces (cheap)
For per-layer token states \(H\in\mathbb{R}^{B\times n\times d}\), form:
\[
C = \mathbb{E}_{b,t}\left[(h_{b,t}-\mu)(h_{b,t}-\mu)^\top\right].
\]
Compute top-\(k\) eigenpairs of \(C\) (symmetric PSD).

Pros: fast, stable.  
Cons: not directly sensitivity.

### Option B: Sensitivity subspaces via Jacobian sketches (medium)
Approximate \(A=J^\top J\) implicitly by applying it to vectors (using the same JVP/VJP), then use Lanczos / randomized SVD to recover top eigenspace.

Pros: sensitivity-faithful.  
Cons: heavier.

### Option C: Routing/influence operator (token graph)
Derive an influence matrix \(P\) from attention weights; compute its Laplacian eigenspaces (diffusion modes).

Pros: directly routing.  
Cons: may be head-specific; need aggregation choices.

**Recommendation:** start with A, then add C, then add B.

---

## 4. Principal angles and drift

Given orthonormal bases \(U, \tilde U \in \mathbb{R}^{d\times k}\), compute:
- \(M=U^\top \tilde U\)
- singular values \(\sigma_i(M)=\cos\theta_i\)
- drift metric: \(d=\sin(\theta_{\max})=\sqrt{1-\sigma_{\min}(M)^2}\)

Implement via `jax.numpy.linalg.svd` on small \(k\times k\) matrix.

Track:
- drift layer-to-layer,
- drift under perturbations (prompt edits, distractor insertions),
- drift vs eigengap.

---

## 5. Eigengap computation

Given eigenvalues \(\lambda_1\ge \dots\ge \lambda_d\), define:
\[
\Delta_k = \lambda_k - \lambda_{k+1}.
\]
Use this gap for Davis–Kahan scaling.

Log per layer:
- \(\Delta_k\),
- ratio \(\|E\|/\Delta_k\) proxies (below).

---

## 6. Davis–Kahan predictor harness

We cannot directly observe \(\|E\|\) in most settings, so we use proxies:

### 6.1 Prompt perturbation proxy
For operator \(A\) computed from baseline prompt and perturbed prompt:
\[
E = \tilde A - A.
\]
Compute \(\|E\|_2\) approximately:
- exact for small \(d\) via SVD,
- or randomized power iteration on \(E\).

Then predict:
\[
\text{predicted drift} \sim \min(1,\|E\|_2/\Delta_k).
\]
Compare with observed drift \(d=\|\sin\Theta(U,\tilde U)\|_2\).

### 6.2 Fine-tuning perturbation proxy
If parameters shift \(\theta\to\theta+\delta\theta\), estimate changes in \(A\) by recomputing \(A\) at checkpoints.

---

## 7. Synthetic “write-keep-read” benchmark (required)

### 7.1 Task definition
Sequence contains:
- key token(s) \(K\),
- value token(s) \(V\),
- distractor spans,
- query token(s) \(Q\) at the end.

Model must output \(V\) given \(K\) at query time.

Vary:
- distance between write and read,
- number/strength of distractors,
- number of simultaneous memories.

### 7.2 What to measure
- accuracy vs distance,
- drift of memory subspace across distance,
- whether “write updates” project onto the memory subspace,
- \(\log\|J\|\) in corridor layers.

---

## 8. Config interface (suggested)

Single config (yaml/json) to control:
- probe layers, k, power iters
- operator choice (covariance/sensitivity/routing)
- perturbation suite strength
- batch sizes and seeds
- output directory

---

## 9. Deliverables (implementation order)

1) Covariance subspace drift + gap (fast win)  
2) Jacobian norm estimator for selected layers  
3) DK predictor (compute E and compare)  
4) Synthetic benchmark with logs tied to diagnostics  
5) Sensitivity subspace via implicit \(J^\top J\) eigenspace sketch (advanced)

---

## 10. Success criteria

A harness run is “successful” when:
- metrics are stable and reproducible across seeds,
- Jacobian norms align with intuitive stability (explode/contract cases),
- drift increases when gaps shrink,
- DK predictor correlates with observed drift under perturbations,
- memory benchmark correlates with corridor stability metrics.
