# MODULUS / RUNT Engineering Spec
## Memory Channels via Jacobian-Gap Control (JAX-first)

**Doc:** `modulus_memory_channels_spec.md`  
**Status:** v0.2 (engineering-ready)  
**Target stack:** JAX (+ Flax or Equinox), Optax, W&B (optional)  
**Primary objectives:** (1) long-context memory stability, (2) controlled mixing, (3) well-conditioned operators

---

## 0. Executive summary

We treat a transformer (or MODULUS-family block stack) as a depth-indexed dynamical system:
\[
x^{(\ell+1)} = F_\ell(x^{(\ell)}),
\]
and operationalize *memory* as a **stable, approximately invariant subspace** \(S\) (a “memory corridor”) with:

1. **Write:** updates inject information into \(S\) (\(P_S \Delta x \neq 0\))  
2. **Keep:** restricted gain is near-marginal (\(\rho(J_{\ell|S}) \approx 1\), \(\|J_{\ell|S}\|_2\approx 1\))  
3. **No leakage:** \( \|(I-P_S)J_\ell P_S\| \ll 1\)  
4. **Erase elsewhere:** \( \|J_{\ell|S^\perp}\|_2 < 1\) (or at least controlled)  
5. **Read:** stable alignment between readout/probing directions and \(S\)

The **spectral gap** criterion is used to ensure \(S\) is *identifiable and stable* under perturbations (prompt shifts, fine-tuning, noise) via Davis–Kahan bounds.

This spec defines **what to build**, **what to measure**, and **what knobs to tune** in a JAX implementation.

---

## 1. Definitions and notations

### 1.1 State
- Sequence length \(n\), model width \(d\).
- Token states at layer \(\ell\): \(X^{(\ell)} \in \mathbb{R}^{n\times d}\).
- Flatten when needed: \(x^{(\ell)}=\mathrm{vec}(X^{(\ell)}) \in \mathbb{R}^{nd}\).

### 1.2 Layer map and Jacobian
- Layer map: \(F_\ell: \mathbb{R}^{nd} \to \mathbb{R}^{nd}\).
- Jacobian: \(J_\ell(x)=\partial F_\ell(x)/\partial x\).

**Practical metric:** spectral norm (largest singular value) \(\|J_\ell\|_2\), estimated via JVP/VJP power iteration (see harness plan).

### 1.3 Memory subspace
- Memory subspace \(S \subseteq \mathbb{R}^{d}\) (per token) or \(S\subseteq \mathbb{R}^{nd}\) (global).
- Projector: \(P_S\).

**Operational choice:** define \(S\) as a top-\(k\) eigenspace of one of:
- representation covariance \(C_\ell = \mathbb{E}[h_\ell h_\ell^\top]\) (feature memory),
- sensitivity operator \(A_\ell = J_\ell^\top J_\ell\) (dynamical memory),
- routing/influence operator over tokens (routing memory).

### 1.4 Spectral gap and subspace stability
Let \(A\) be a symmetric operator defining the memory subspace. Let \(U\) be the top-\(k\) eigenspace. If \(\tilde A = A+E\) under perturbations, Davis–Kahan gives:
\[
\|\sin\Theta(U,\tilde U)\|_2 \le \frac{\|E\|_2}{\Delta},
\]
where \(\Delta\) is the relevant eigengap (e.g., \(\lambda_k-\lambda_{k+1}\)).

**Design intent:** *engineer and maintain gaps* for stable memory directions.

---

## 2. Architecture requirements (MODULUS / RUNT / nGPT-aligned)

### 2.1 Corridor–scatterer decomposition (per block)
Each block should decompose into:
- **Corridor:** identity-like transport (residual path)
- **Scatterer:** controlled mixing (attention + MLP “kicks”)

Recommended block skeleton (pre-norm):
\[
X \leftarrow X + \alpha_\ell\,\mathrm{Attn}(\mathrm{Norm}(X))
\]
\[
X \leftarrow X + \beta_\ell\,\mathrm{MLP}(\mathrm{Norm}(X))
\]
with explicit gain control \(\alpha_\ell,\beta_\ell\) (learned or scheduled) and well-conditioned parametrizations.

### 2.2 Conditioning targets
- Keep \(\log\|J_\ell\|_2\) near 0 in memory-preserving depth regions.
- Avoid persistent \(\log\|J_\ell\|_2 \gg 0\) (chaotic brittleness) or \(\ll 0\) (forgetting).
- Encourage *controlled transient amplification* (non-normal bursts) only where mixing is beneficial.

### 2.3 Memory channel mechanisms (must be explicit)
A memory channel is realized when the model reliably implements:

**(Write)** For a “writer” token \(i\) attending to content token(s) \(j\):
- attention weights \(P_{ij}\) address content,
- values \(V_j\) are projected into memory directions,
- residual adds into token state.

Concretely, enforce or encourage:
- a subset of heads dedicated to memory writing,
- value projection sub-block \(W_V\) (or a slice of it) aligned with \(S\),
- optional gating \(g\in[0,1]\) controlling write magnitude.

**(Keep)** Ensure restricted dynamics preserves \(S\):
- norm control (unit-sphere / RMS constraints) for stability,
- bounded gains \(\alpha_\ell,\beta_\ell\),
- avoid rotations of \(S\) across depth (gap + low drift).

**(No-leak)** Reduce leakage into \(S^\perp\) and vice versa:
- head decoupling (reduce destructive \(W_O\) interference),
- orthogonal/near-orthogonal parametrizations for key linear maps,
- optional penalty term on leakage (see §4).

**(Erase elsewhere)** Encourage contraction or mixing outside memory:
- regularize \(\|J_{\ell|S^\perp}\|_2\),
- optional “mixing layers” separated from memory corridors.

**(Read)** Ensure downstream readouts (or probes) remain aligned with \(S\):
- stable eigenspaces via gaps,
- avoid degenerate spectra.

### 2.4 Symmetry breaking (salient, not decorative)
Symmetry breaking is used to **create spectral separation**:
- positional encodings (break token permutation symmetry),
- head specialization (break head symmetry),
- heterogeneity across depth (break layer homogeneity),
- routing sparsity / selective gates (break uniform mixing).

The goal is not turbulence; the goal is **mode separation** (gap engineering).

---

## 3. Measurement requirements (what must be logged)

Minimum required (per run):
1. **Layerwise Jacobian gain:** \(\widehat{\|J_\ell\|_2}\) (approx)  
2. **Subspace drift:** principal angle distance \(d_\ell=\|\sin\Theta(U_\ell,U_{\ell+1})\|_2\) for chosen operator  
3. **Eigengap:** \(\Delta_\ell = \lambda_k - \lambda_{k+1}\) (same operator)  
4. **Leakage proxy:** \(\|(I-P_S)\Delta X\| / \|\Delta X\|\) for write updates  
5. **Memory retention task metric:** synthetic and/or downstream (see §5)

---

## 4. Optional regularizers and constraints (ablation-friendly)

All are optional and should be implemented behind flags.

### 4.1 Gain regularizer
Penalize deviation from marginal gain in the “memory corridor” layers:
\[
\mathcal{L}_{\text{gain}}=\sum_{\ell\in\mathcal{M}} \big(\log\widehat{\|J_\ell\|_2}\big)^2
\]
(or hinge to keep within a band).

### 4.2 Leakage regularizer
Given a projector \(P_S\) and update \(\Delta x_\ell = F_\ell(x)-x\):
\[
\mathcal{L}_{\text{leak}}=\sum_{\ell\in\mathcal{M}} \frac{\|(I-P_S)\Delta x_\ell\|^2}{\|\Delta x_\ell\|^2+\epsilon}
\]
This encourages writes to land in \(S\) (and stay there).

### 4.3 Gap encouragement (careful)
Softly encourage eigengaps:
\[
\mathcal{L}_{\text{gap}}=\sum_{\ell\in\mathcal{M}} \mathrm{softplus}(\tau-\Delta_\ell)
\]
Use cautiously; can distort representation geometry.

### 4.4 Head decoupling / interference control
Penalize cross-head covariance at the head outputs prior to \(W_O\).

---

## 5. Required evaluation tasks (to validate “memory channels”)

### 5.1 Synthetic “write-keep-read” task (must implement)
Create a controlled benchmark where:
- at time \(t_w\), a key-value pair is presented,
- later at time \(t_r\), a query requires retrieving the value.

Measure:
- accuracy vs distance \(t_r-t_w\),
- robustness to distractors inserted between write and read,
- sensitivity to prompt perturbations.

### 5.2 Long-context retrieval (optional)
Plug into existing retrieval tasks, but keep synthetic tasks as ground truth.

### 5.3 Perturbation suite (for Davis–Kahan relevance)
Run:
- prompt edits (synonym swaps),
- insertion of irrelevant spans,
- small parameter perturbations (e.g., LoRA micro-update),
and quantify subspace drift vs \(\|E\|/\Delta\) proxies.

---

## 6. Implementation constraints (JAX)

- All Jacobian computations must use **JVP/VJP**, never explicit Jacobian materialization.
- Use `jax.jvp` for \(Jv\), and `jax.vjp` for \(J^\top w\).
- Use `jit` and `vmap` to batch across layers and samples.
- Avoid storing activations for all layers unless checkpointed; prefer a small set of “probe layers.”

---

## 7. Deliverables checklist

- [ ] JAX model wrapper exposing per-layer `F_ell` (or block function)  
- [ ] Diagnostics module to estimate \(\|J_\ell\|_2\) via power iteration  
- [ ] Subspace module: compute covariance eigenspaces and principal angles  
- [ ] Gap tracker and Davis–Kahan predictor plot (drift vs \(\|E\|/\Delta\))  
- [ ] Synthetic write-keep-read benchmark  
- [ ] W&B logging hooks (optional but recommended)  

---

## 8. Success criteria

A build meets spec if:
- memory benchmark retention scales with distance better than baseline,
- layerwise \(\log\|J_\ell\|_2\) stays near 0 in the designated corridor,
- subspace drift remains small under perturbations when gaps are large,
- the empirical drift aligns with Davis–Kahan-style scaling.

---

## 9. Notes on choosing the operator defining \(S\)

Start with covariance \(C_\ell\) (cheap, stable), then add sensitivity operator \(A_\ell\) (more faithful but heavier).

- Covariance \(C_\ell\): “what directions are used”
- Sensitivity \(A_\ell = J^\top J\): “what directions amplify perturbations”
- Influence operator: “what routes move information between tokens”

Use all three in mature studies; begin with \(C_\ell\).
