# The Plaquette Amplitude

## 1. The Cross Region

Fix a conformal boundary condition $|B\rangle$ of a 2d CFT. The domain is the **cross-shaped region**

$$\mathcal{C}_\ell = S_H \cup S_V, \qquad S_H = \mathbb{R} \times \bigl[-\tfrac{1}{2}, \tfrac{1}{2}\bigr], \quad S_V = \bigl[-\tfrac{\ell}{2}, \tfrac{\ell}{2}\bigr] \times \mathbb{R},$$

the union of a horizontal infinite strip of width $1$ and a vertical infinite strip of width $\ell$. The boundary condition $|B\rangle$ is imposed on the entire boundary $\partial\mathcal{C}_\ell$.

The complement of the central rectangle consists of four **half-infinite strips** (arms) $A_R, A_L, A_T, A_B$, each carrying the open-string state space $\mathcal{H}_{BB}$ at its cross-section. A state $|\alpha_i\rangle \in \mathcal{H}_{BB}$ prepared at the infinity of arm $A_i$ enters the path integral, defining an $n$-point amplitude depending on how many arms are open versus capped.

### Open Cardy state: capping a channel

An **open Cardy state** on arm $A_i$ means we cap off the arm by imposing $|B\rangle$ on its terminal cross-section (in the limit, replacing the half-strip with a boundary segment). No propagating degrees of freedom enter from that channel. In the open-channel decomposition, the Cardy state is a specific boundary state $|B\rangle\!\rangle$ — a weighted sum over states in $\mathcal{H}_{BB}$ determined by the boundary condition. It should not be confused with $|0\rangle$, the $SL(2)$-invariant vacuum in $\mathcal{H}_{BB}$ (the state corresponding to the identity local operator $\mathbb{1}$ under the standard state-operator map); the two are distinct objects.

| Arms open | Geometry | Amplitude |
|---|---|---|
| 4 | Cross | Boundary 4-point function |
| 3 | T-shape | Boundary 3-point function |
| 2 (opposite) | Strip | Propagator $= e^{-\pi\ell(L_0 - c/24)}$, trivial |
| 2 (adjacent) | L-shape | Boundary 2-point function |
| 1 | Capped T | Tadpole |
| 0 | Rectangle | Partition function on the rectangle |

The two-opposite-Cardy case is just the strip propagator — the exponential of $L_0$ — and requires no computation.

### The 3-point amplitude as the primary object

The **3-point amplitude** (T-shape, one arm capped) is the central object for the lattice construction. Capping the bottom arm $A_B$, the domain becomes:

$$\mathcal{T}_\ell = S_H \cup \bigl([-\tfrac{\ell}{2}, \tfrac{\ell}{2}] \times [\tfrac{1}{2}, \infty)\bigr),$$

a horizontal strip with one vertical arm extending upward. The amplitude

$$V_\ell(\alpha_L, \alpha_R, \alpha_T)$$

is the **interaction vertex** of the lattice model: it is the tensor contracted when neighboring plaquettes are glued together. The 4-point amplitude serves as a consistency check (associativity of gluing) but is not needed for the basic construction.

---

## 2. Conformal Map and Local Coordinates

### 2.1 The map to the upper half-plane

The domain $\mathcal{T}_\ell$ (or $\mathcal{C}_\ell$ for the 4-point case) is mapped conformally to $\mathbb{H}$ via a Schwarz–Christoffel map $f: \mathbb{H} \to \mathcal{T}_\ell$. The arm infinities map to marked points $x_i \in \mathbb{R}$. The detailed construction and parameter determination are described in the companion document [conformal_map_cross.md](conformal_map_cross.md).

### 2.2 Local coordinate maps $f_i$ and $g_i$

For each open arm $A_i$, define:

$$f_i : (\text{neighborhood of } x_i \text{ in } \mathbb{H}) \;\longrightarrow\; D^+$$

as the composition of $f$ (restricted near $x_i$, landing in arm $A_i$) with the exponential map $\xi = e^{-\pi w / w_i}$ that sends arm $A_i$ to the **upper unit semidisk** $D^+ = \{|\xi| < 1,\; \text{Im}\,\xi > 0\}$. The infinity of the arm maps to $\xi_i = 0$ (the origin of $D^+$).

Near $x_i$, the map $f_i$ has a simple zero:

$$f_i(z) = \alpha_i(z - x_i) + \beta_i(z - x_i)^2 + \cdots, \qquad \alpha_i \neq 0.$$

The local coordinate $\xi_i = f_i(z)$ is the natural coordinate for the state-operator correspondence on arm $i$: a state $|\alpha_i\rangle \in \mathcal{H}_{BB}$ at the arm's infinity corresponds to a boundary operator $\psi_{\alpha_i}(\xi_i = 0)$ in the local frame.

Define the inverse:

$$g_i = f_i^{-1}: D^+ \to \mathbb{H}, \qquad g_i(\xi) = x_i + \frac{1}{\alpha_i}\xi + \cdots$$

This maps the semidisk local patch back to the UHP global patch.

### 2.3 Mode expansion in local coordinates

In the local coordinate $\xi_i$ on arm $A_i$, the current algebra modes are:

$$J^a_n = \oint_0 \frac{d\xi_i}{2\pi i}\; \xi_i^n\, J^a(\xi_i), \qquad n \in \mathbb{Z}.$$

A primary state $|h; \lambda\rangle$ at the origin satisfies $J^a_n|h;\lambda\rangle = 0$ for $n > 0$ and $J^a_0|h;\lambda\rangle = t^a_\lambda|h;\lambda\rangle$.

---

## 3. The Functions $F_m^{(i)}$ and the Recursion

### 3.1 Definition

For each open arm $i$ and each positive integer $m \geq 1$, define:

$$F_m^{(i)}(z) := \bigl(f_i(z)\bigr)^{-m}\Big|_{\text{singular part in } z \text{ around } x_i}.$$

That is: expand $(f_i(z))^{-m}$ in a Laurent series in $(z - x_i)$ around $x_i$, and keep only the terms with negative powers of $(z - x_i)$.

### 3.2 Properties

**(a) Regularity.** $F_m^{(i)}(z)$ is holomorphic in $\mathbb{H}$ except at $z = x_i$, where it has a pole of order $m$. It is a polynomial in $(z - x_i)^{-1}$ of degree $m$, with coefficients determined by the Taylor data of $f_i$.

**(b) Local behavior at $x_i$.** Composing with $g_i$:

$$F_m^{(i)}(g_i(\xi)) = \xi^{-m} + \sum_{k=0}^{\infty} \mathcal{N}^{(i \to i)}_{m,k}\;\xi^k.$$

This follows because $(f_i(g_i(\xi)))^{-m} = \xi^{-m}$, and the regular part of $(f_i(z))^{-m}$ composed with $g_i$ contributes only non-negative powers. The intermediate singular orders $\xi^{-m+1}, \ldots, \xi^{-1}$ are absent by construction.

**(c) Behavior at other arms.** At each $x_j$ with $j \neq i$, $F_m^{(i)}$ is regular. Its expansion in the local coordinate $\xi_j = f_j(z)$:

$$F_m^{(i)}(g_j(\xi_j)) = \sum_{k=0}^{\infty} \mathcal{N}^{(i \to j)}_{m, k}\; \xi_j^k.$$

**(d) Behavior at infinity.** $F_m^{(i)}(z) \to 0$ as $z \to \infty$ in $\mathbb{H}$.

The coefficients $\mathcal{N}^{(i \to j)}_{m,k}$ (for all $j$, including $j = i$) depend on $\ell$ but not on the CFT data. We call them **Neumann coefficients** by analogy with the Neumann matrices $N^{rs}_{nm}$ in open string field theory, where the diagonal matrices $N^{rr}_{nm}$ play the same role as our $\mathcal{N}^{(i \to i)}_{m,k}$.

### 3.3 The Ward identity

Since $J^a(z)$ is holomorphic in $\mathbb{H}$ (by the doubling trick) and $F_m^{(i)}(z)$ is meromorphic with its only singularity at $x_i$, the product $F_m^{(i)}(z)\,J^a(z)$ inside the correlator is meromorphic with singularities at each $x_j$. By Cauchy's theorem (the large-contour integral vanishes by property (d)):

$$0 = \sum_{j} \oint_{x_j} \frac{dz}{2\pi i}\; F_m^{(i)}(z)\, \bigl\langle J^a(z)\, \prod_k \psi_{\alpha_k}(x_k) \bigr\rangle.$$

Changing to local coordinates $\xi_j = f_j(z)$ in each contour integral (using $J^a(z)\,dz = J^a(\xi_j)\,d\xi_j$ for weight 1), the contribution from $x_j$ is:

$$\oint_0 \frac{d\xi_j}{2\pi i}\; F_m^{(i)}(g_j(\xi_j))\; J^a(\xi_j) = \begin{cases} J^a_{-m} + \sum_{k \geq 0}\mathcal{N}^{(i\to i)}_{m,k}\,J^a_k & (j = i), \\ \sum_{k \geq 0}\mathcal{N}^{(i\to j)}_{m,k}\,J^a_k & (j \neq i). \end{cases}$$

### 3.4 The recursion relation

$$\boxed{J^a_{-m}\bigl|\alpha_i\bigr\rangle = -\sum_{j}\;\sum_{k \geq 0} \mathcal{N}^{(i \to j)}_{m, k}\; J^a_k\bigl|\alpha_j\bigr\rangle,}$$

understood as an identity inside the amplitude $V_\ell(\ldots)$, with $j$ running over all open arms including $i$. For a primary state $|\alpha_j\rangle = |h;\lambda\rangle$, only the $k = 0$ term from each $j$ survives ($J^a_0 = t^a_\lambda$; higher $J^a_k$ annihilate the primary).

Each application of the recursion strips one creation mode $J^a_{-m}$ from arm $i$ and replaces it with annihilation/zero modes $J^a_k$ ($k \geq 0$) distributed across all arms. Applied repeatedly, it reduces any descendant amplitude to a linear combination of primary amplitudes, with coefficients polynomial in $\mathcal{N}^{(i\to j)}_{m,k}$ and $t^a_\lambda$.

The recursion can be used in two ways:

**(i) Single-index computation.** Given a specific triple of states, apply the recursion repeatedly to reduce to primaries. Each step triggers further recursion on the resulting terms until all states are primary.

**(ii) Building the full vertex tensor.** To construct the entire tensor $V_\ell^{(h_{\max})}$ up to truncation level $h_{\max}$, order the triples of states by **total level** $N_{\text{tot}} = N_L + N_R + N_T$ (where $N_i$ is the descendant level on arm $i$). At total level $0$, all states are primary and $V_\ell$ is evaluated directly (Section 4.1). For total level $N_{\text{tot}} \geq 1$, pick any arm $i$ carrying a creation mode $J^a_{-m}$ and apply the recursion once: the right-hand side involves only states at strictly lower total level (since $J^a_k$ with $k \geq 0$ does not increase the level on arm $j$, and the mode $J^a_{-m}$ has been removed from arm $i$). These are already computed. So each entry of the vertex tensor is obtained by a **single recursion step plus table lookup** — no iterated recursion is needed.

Mode (ii) is the natural one for building the lattice model. Within a given total level, the entries can be computed in any order.

### 3.5 Virasoro descendants

The same argument applies with $T(z)$ (weight 2) in place of $J^a(z)$, using the same functions $F_m^{(i)}$. The transformation law for weight 2 introduces additional terms (Schwarzian derivative). When the CFT has a current algebra $\hat{\mathfrak{g}}_k$, the Sugawara construction handles all Virasoro descendants within the current-algebra tower via iterated $J^a$ recursion; only coset descendants require the independent $T(z)$ recursion.

---

## 4. Primary Amplitude

### 4.1 The 3-point vertex

For three current-algebra primaries $|h_i; \lambda_i\rangle$ on arms $L, R, T$ (with $B$ capped), the vertex is a **boundary three-point function on $\mathbb{H}$**:

$$V_\ell^{(\text{prim})}(h_L, h_R, h_T) = \prod_{i \in \{L,R,T\}} |\alpha_i|^{2h_i} \;\cdot\; \bigl\langle \psi_{h_L}(x_L)\, \psi_{h_R}(x_R)\, \psi_{h_T}(x_T) \bigr\rangle_{\mathbb{H}},$$

where $|\alpha_i|^{2h_i}$ is the Jacobian factor from the local-to-UHP coordinate change (with $\alpha_i = f_i^{\prime}(x_i)$). The boundary three-point function is fixed by conformal symmetry up to an OPE coefficient:

$$\bigl\langle \psi_{h_L}(x_L)\, \psi_{h_R}(x_R)\, \psi_{h_T}(x_T) \bigr\rangle_{\mathbb{H}} = \frac{C_{LRT}}{|x_L - x_R|^{h_L + h_R - h_T}\,|x_R - x_T|^{h_R + h_T - h_L}\,|x_L - x_T|^{h_L + h_T - h_R}}.$$

So the primary 3-point vertex is proportional to the boundary OPE coefficient $C_{LRT}$, dressed by known powers of the geometric data ($x_i$, $\alpha_i$).

### 4.2 Remark: the 4-point amplitude

When all four arms are open, the primary amplitude becomes a **boundary four-point function**, which is no longer fixed by conformal symmetry:

$$T_\ell^{(\text{prim})} \propto \sum_p C_{LR}^{\;\;p}\,C_p^{\;\;TB}\;\mathcal{F}_p(\eta),$$

with $\eta$ the cross-ratio of the four marked points and $\mathcal{F}_p$ the boundary conformal blocks. This amplitude provides a nontrivial check: the tensor $T_\ell^{(h_{\max})}$ must be consistent with the vertex $V_\ell^{(h_{\max})}$ under contraction (associativity of the lattice model). However, the 4-point amplitude is not needed for the construction of the lattice model itself.

---

## 5. Small-$\ell$ Behavior

### 5.1 Degeneration of the SC map

As $\ell \to 0$: $p \to 0$, $C \to 2/\pi$, and the two branch points $\pm p$ collide with the pole at $x_T = 0$. The SC derivative degenerates to

$$f^{\prime}(z) \;\xrightarrow{\;\ell \to 0\;}\; \frac{2}{\pi(z^2 - 1)},$$

(since $\sqrt{z^2}/z = 1$ in $\mathbb{H}$). The limiting map is $f(z) = \frac{1}{\pi}\log\frac{z-1}{z+1} - \frac{i}{2}$, the standard strip map, with $z = 0$ becoming a regular interior point of the boundary (mapping to $(0, 1/2)$ on the top edge of the strip).

### 5.2 Horizontal Jacobians

At $\ell = 0$, the strip map gives $|\alpha_{L,R}| = \frac{1}{2}$. Corrections are even in $\ell$ by the $\mathbb{Z}_2$ symmetry:

$$|\alpha_{L,R}(\ell)| = \frac{1}{2} + O(\ell^2).$$

### 5.3 Vertical Jacobian

The exact relation $f(p) = \ell/2 + i/2$ (the concave corner) gives, using $\rho_T(p) = \rho_T^{(0)} + O(p^2)$ (odd Taylor coefficients vanish since $f^{\prime}$ is odd around $x_T = 0$) and $p = \ell/2 + O(\ell^3)$:

$$\rho_T^{(0)} = \frac{\ell}{2} + \frac{i}{2} + \frac{i\ell}{\pi}\log\frac{\ell}{2} + O(\ell^2).$$

The Jacobian $\alpha_T = e^{i\pi\rho_T^{(0)}/\ell}$, and computing the exponent:

$$\frac{i\pi\rho_T^{(0)}}{\ell} = -\frac{\pi}{2\ell} + \log\frac{2}{\ell} + \frac{i\pi}{2} + O(\ell),$$

giving:

$$|\alpha_T(\ell)| = \frac{2}{\ell}\,e^{-\pi/(2\ell)}\,(1 + O(\ell)).$$

The dominant behavior is exponential suppression $e^{-\pi/(2\ell)}$, with a power-law prefactor $2/\ell$.

### 5.4 Structure of the vertex at small $\ell$

For a primary $|h_T\rangle$ on the vertical arm, the Jacobian factor is:

$$|\alpha_T|^{2h_T} = \left(\frac{2}{\ell}\right)^{2h_T} e^{-\pi h_T/\ell}\,(1 + O(\ell)).$$

**The $h_T = 0$ state ($|0\rangle$, the identity boundary operator).** The Jacobian $|\alpha_T|^0 = 1$ regardless of $\ell$, and the insertion of the identity operator $\mathbb{1}$ at $x_T = 0$ does not affect the UHP correlator. So:

$$V_\ell(\psi_L, \psi_R, |0\rangle) = \frac{|\alpha_L(\ell)|^{2h_L}\,|\alpha_R(\ell)|^{2h_R}}{|x_L - x_R|^{2h_L}}\;\delta_{LR^{\ast}} = \frac{1}{2^{6h_L}}\bigl(1 + O(\ell^2\, h_L)\bigr)\;\delta_{LR^{\ast}}.$$

This is diagonal (forces $\psi_R = \psi_L^{\ast}$), and the $\ell$-correction starts at $O(\ell^2)$. However, the correction is **$h_L$-dependent**: different conformal weights receive different $O(\ell^2)$ corrections, so as a map $\mathcal{H}_L \to \mathcal{H}_R$, the $|0\rangle$ insertion is a rescaling that varies across weight sectors. It approaches the identity (up to an overall normalization) only at $\ell = 0$.

**States with $h_T > 0$.** Exponentially suppressed by $e^{-\pi h_T/\ell}$, with power-law prefactor $\ell^{-2h_T}$. These contribute off-diagonal terms (when $\psi_T$ carries charge) or subleading diagonal corrections (when $\psi_T$ is neutral).

**The open Cardy state.** In the open-channel decomposition, the Cardy state is $|B\rangle\!\rangle = \sum_h \langle h|B\rangle\!\rangle\, |h\rangle$. Its effect on the vertex is:

$$V_\ell(\psi_L, \psi_R, |B\rangle\!\rangle) = \sum_h \langle h|B\rangle\!\rangle\; V_\ell(\psi_L, \psi_R, |h\rangle).$$

At small $\ell$, the $h > 0$ terms are exponentially suppressed, and the $h = 0$ term dominates — but the Cardy state coefficient $\langle 0|B\rangle\!\rangle$ and the specific $O(\ell^2)$ corrections determine the quantitative behavior, which differs from simply inserting $|0\rangle$.

---

## 6. Summary of the Algorithm

Given three states $|\alpha_i\rangle = J^{a_1}_{-n_1}\cdots J^{a_r}_{-n_r}|h_i;\lambda_i\rangle$ on the arms of the T-shaped domain:

1. **SC parameters** (once per $\ell$): For the T-shape, the map parameters are in closed form: $p(\ell) = \ell/\sqrt{4+\ell^2}$. See [conformal_map_cross.md](conformal_map_cross.md).
2. **Local coordinate maps** (once per $\ell$): Compute the Taylor series of $f_i$ and its inverse $g_i$ at each marked point, to the desired order. See [conformal_map_cross.md](conformal_map_cross.md), Section 3.
3. **$F_m^{(i)}$ and Neumann coefficients** (once per $\ell$): Compute $(f_i(z))^{-m}|_{\text{sing}}$ and expand $F_m^{(i)}(g_j(\xi_j))$ to extract $\mathcal{N}^{(i\to j)}_{m,k}$ for all $j$ (including $j = i$). See [conformal_map_cross.md](conformal_map_cross.md), Section 5.
4. **Build the vertex tensor** (once per $\ell$ and $h_{\max}$): Sweep over triples of states in order of increasing total level $N_{\text{tot}} = N_L + N_R + N_T$. At $N_{\text{tot}} = 0$, evaluate the primary vertex (Section 4.1). At each higher level, apply the Ward identity (Section 3.4) once per entry to reduce to already-computed entries at lower total level.

Steps 1–3 are geometry (CFT-independent). Step 4 uses the CFT data (boundary OPE coefficients and representation matrices) and produces the finite tensor $V_\ell^{(h_{\max})}$, the elementary interaction vertex of the lattice model.
