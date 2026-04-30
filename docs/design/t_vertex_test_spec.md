# Test Spec: T-Vertex for the Compact Boson

Tests are organized in implementation order. Each section corresponds to one TDD cycle: write the tests first, then the code to pass them.

---

## 1. `TruncLaurent` — Truncated Laurent Series

### 1.1 Construction and access

```julia
# Series: 2ζ⁻¹ + 3 + ζ, truncated at O(ζ²)
s = TruncLaurent(-1, [2.0, 3.0, 1.0], 2)
@test valuation(s) == -1
@test precision(s) == 2
@test s[-1] == 2.0    # coefficient of ζ⁻¹
@test s[0] == 3.0
@test s[1] == 1.0
```

### 1.2 Multiplication

$$(\zeta + 2)(3\zeta^{-1} + 1) = 3 + \zeta + 6\zeta^{-1} + 2 = 6\zeta^{-1} + 5 + \zeta$$

```julia
a = TruncLaurent(1, [1.0, 2.0], 3)       # ζ + 2ζ² (but we only need ζ+2 for this test)
```

Actually, cleaner:

$$(1 + 2\zeta)(3 + \zeta) = 3 + \zeta + 6\zeta + 2\zeta^2 = 3 + 7\zeta + 2\zeta^2$$

```julia
a = TruncLaurent(0, [1.0, 2.0], 3)       # 1 + 2ζ + O(ζ³)
b = TruncLaurent(0, [3.0, 1.0], 3)       # 3 + ζ + O(ζ³)
c = a * b
@test c[0] ≈ 3.0
@test c[1] ≈ 7.0
@test c[2] ≈ 2.0
```

Laurent times Laurent:

$$(\zeta^{-1} + 1)(2\zeta^{-1} - 1) = 2\zeta^{-2} - \zeta^{-1} + 2\zeta^{-1} - 1 = 2\zeta^{-2} + \zeta^{-1} - 1$$

```julia
a = TruncLaurent(-1, [1.0, 1.0], 2)      # ζ⁻¹ + 1 + O(ζ²)
b = TruncLaurent(-1, [2.0, -1.0], 2)     # 2ζ⁻¹ - 1 + O(ζ²)
c = a * b
@test c[-2] ≈ 2.0
@test c[-1] ≈ 1.0
@test c[0] ≈ -1.0
```

### 1.3 Inversion

$(1 + \zeta)^{-1} = 1 - \zeta + \zeta^2 - \zeta^3 + \cdots$

```julia
a = TruncLaurent(0, [1.0, 1.0, 0.0, 0.0, 0.0], 5)   # 1 + ζ + O(ζ⁵)
b = inv(a)
@test b[0] ≈ 1.0
@test b[1] ≈ -1.0
@test b[2] ≈ 1.0
@test b[3] ≈ -1.0
# round-trip
c = a * b
@test c[0] ≈ 1.0
for k in 1:4
    @test abs(c[k]) < 1e-14
end
```

$(2 + 3\zeta)^{-1} = \frac{1}{2}(1 + \frac{3}{2}\zeta)^{-1} = \frac{1}{2}(1 - \frac{3}{2}\zeta + \frac{9}{4}\zeta^2 - \cdots)$

```julia
a = TruncLaurent(0, [2.0, 3.0, 0.0, 0.0], 4)
b = inv(a)
@test b[0] ≈ 1/2
@test b[1] ≈ -3/4
@test b[2] ≈ 9/8
```

### 1.4 Exponentiation (zero constant term)

$\exp(\zeta) = 1 + \zeta + \zeta^2/2 + \zeta^3/6 + \cdots$

```julia
a = TruncLaurent(1, [1.0], 6)   # ζ + O(ζ⁶)
b = exp_series(a)
@test b[0] ≈ 1.0
@test b[1] ≈ 1.0
@test b[2] ≈ 1/2
@test b[3] ≈ 1/6
@test b[4] ≈ 1/24
```

$\exp(2\zeta + \zeta^2) = 1 + 2\zeta + \zeta^2(1 + 2) + \cdots$

More precisely: let $f = 2\zeta + \zeta^2$.  $e^f = 1 + f + f^2/2 + \cdots$.  $f^2 = 4\zeta^2 + O(\zeta^3)$.

$$e^f = 1 + 2\zeta + (1 + 2)\zeta^2 + \cdots = 1 + 2\zeta + 3\zeta^2 + \cdots$$

Check: $e^f|_{\zeta^2} = f|_{\zeta^2} + f^2/2|_{\zeta^2} = 1 + 4/2 = 3$.  ✓

```julia
a = TruncLaurent(1, [2.0, 1.0], 5)   # 2ζ + ζ² + O(ζ⁵)
b = exp_series(a)
@test b[0] ≈ 1.0
@test b[1] ≈ 2.0
@test b[2] ≈ 3.0
```

### 1.5 Composition

$f(g(\xi))$ where $g(\xi) = 2\xi + \xi^2$, $f(\zeta) = 1 + 3\zeta + \zeta^2$.

$f(g(\xi)) = 1 + 3(2\xi + \xi^2) + (2\xi + \xi^2)^2 = 1 + 6\xi + 3\xi^2 + 4\xi^2 + \cdots = 1 + 6\xi + 7\xi^2 + \cdots$

```julia
f = TruncLaurent(0, [1.0, 3.0, 1.0], 4)    # 1 + 3ζ + ζ²
g = TruncLaurent(1, [2.0, 1.0], 4)           # 2ξ + ξ² (valuation 1, required for compose)
h = compose(f, g)
@test h[0] ≈ 1.0
@test h[1] ≈ 6.0
@test h[2] ≈ 7.0
```

### 1.6 Series reversion

If $f(\zeta) = 2\zeta + \zeta^2$, then $g = f^{-1}$ satisfies $f(g(\xi)) = \xi$.

$g(\xi) = \frac{1}{2}\xi - \frac{1}{8}\xi^2 + \cdots$ (by Lagrange inversion or iteration).

Check: $f(g) = 2(\frac{1}{2}\xi - \frac{1}{8}\xi^2) + (\frac{1}{2}\xi - \frac{1}{8}\xi^2)^2 = \xi - \frac{1}{4}\xi^2 + \frac{1}{4}\xi^2 + O(\xi^3) = \xi + O(\xi^3)$.  ✓

```julia
f = TruncLaurent(1, [2.0, 1.0, 0.0, 0.0], 5)
g = series_revert(f)
@test g[1] ≈ 1/2        # leading coeff is 1/f'(0) = 1/2
@test g[2] ≈ -1/8

# round-trip: f(g(ξ)) = ξ
h = compose(f, g)
@test h[1] ≈ 1.0
for k in 2:4
    @test abs(h[k]) < 1e-13
end
```

### 1.7 Singular / regular part

```julia
s = TruncLaurent(-2, [1.0, 3.0, 5.0, 2.0, 7.0], 3)
# = ζ⁻² + 3ζ⁻¹ + 5 + 2ζ + 7ζ²
sp = singular_part(s)
@test sp[-2] ≈ 1.0
@test sp[-1] ≈ 3.0
@test valuation(sp) == -2
@test precision(sp) == 0    # or however "up to ζ⁻¹" is encoded

sr = regular_part(s)
@test sr[0] ≈ 5.0
@test sr[1] ≈ 2.0
@test sr[2] ≈ 7.0
```

### 1.8 BigFloat cross-check

```julia
# Same tests as above but with T = BigFloat.
# Verify that results agree to Float64 precision.
a = TruncLaurent(0, BigFloat[1, 1, 0, 0, 0], 5)
b = inv(a)
@test Float64(b[3]) ≈ -1.0
```

---

## 2. `SCMap` — Schwarz–Christoffel Parameters

### 2.1 Closed-form values

At $\ell = 1$:

$$p = \frac{1}{\sqrt{5}} \approx 0.4472135955, \qquad C = \frac{\sqrt{5}}{\pi} \approx 0.7117625434$$

```julia
sc = compute_sc_params(1.0)
@test sc.p ≈ 1/√5
@test sc.C ≈ √5/π
```

At $\ell = 2$:

$$p = \frac{2}{\sqrt{8}} = \frac{1}{\sqrt{2}} \approx 0.7071, \qquad C = \frac{\sqrt{8}}{\pi} = \frac{2\sqrt{2}}{\pi} \approx 0.9003$$

```julia
sc = compute_sc_params(2.0)
@test sc.p ≈ 1/√2
@test sc.C ≈ 2√2/π
```

### 2.2 Limits

```julia
# ℓ → 0: vertical arm closes, p → 0
sc = compute_sc_params(1e-10)
@test sc.p < 1e-10

# ℓ → ∞: p → 1
sc = compute_sc_params(1e6)
@test sc.p > 1 - 1e-6

# Identity: Cp = ℓ/π always
for ℓ in [0.1, 0.5, 1.0, 2.0, 5.0]
    sc = compute_sc_params(ℓ)
    @test sc.C * sc.p ≈ ℓ/π
end
```

### 2.3 f' evaluation (for downstream tests)

Provide a reference implementation of $f^{\prime}(z) = C\sqrt{z^2 - p^2}/(z(z^2 - 1))$ evaluated at a specific point, to cross-check the series expansion.

At $\ell = 1$, $z = 0.5$ (a regular point in $\mathbb{H}$, away from poles and branch points):

```julia
function fprime_exact(z, sc)
    return sc.C * sqrt(z^2 - sc.p^2) / (z * (z^2 - 1))
end
# The branch of sqrt must be chosen consistently (see §8.1 of main spec).
```

---

## 3. `LocalCoordinates` — $f_i$, $g_i$ Series

### 3.1 Residue check

The Laurent expansion of $f^{\prime}$ around each $x_i$ should have the correct residue (coefficient of $\zeta^{-1}$):

```julia
# At ℓ = 1:
geom, _ = compute_geometry(1.0, 20)

# The residue at x_R = 1 should be w_R σ_R / π.
# We don't hardcode the sign here; instead check consistency with
# the SC formula: Res_{z=1} f' = C√(1-p²)/2
sc = geom.sc
expected_res_R = sc.C * sqrt(1 - sc.p^2) / 2
@test expected_res_R ≈ 1/π   # ≈ 0.31830988...

# Check that the computed expansion agrees:
# geom.arms.R stores the Laurent data of f'(z) around x_R = 1.
# The residue is the coefficient of (z-1)⁻¹.
# This is encoded in the arm data as w_R * σ_R / π.
@test abs(geom.arms.R.w * geom.arms.R.σ / π) ≈ abs(expected_res_R)
```

Similarly for the T arm:
$$|\text{Res}_{z=0}\,f^{\prime}| = C\,p = \ell/\pi$$

```julia
@test sc.C * sc.p ≈ 1/π   # at ℓ = 1
```

### 3.2 Round-trip: $f_i(g_i(\xi)) = \xi$

The defining property of the local coordinate maps.  This is the single most important geometry test.

```julia
geom, _ = compute_geometry(1.0, 30)

for arm in (:L, :R, :T)
    f = geom.arms[arm].f_series   # f_i(x_i + ζ) as TruncLaurent, valuation 1
    g = geom.arms[arm].g_series   # g_i(ξ) - x_i as TruncLaurent, valuation 1

    # f(g(ξ)) should equal ξ (i.e., identity series with coeff 1 at ξ¹, 0 elsewhere)
    h = compose(f, g)
    @test h[1] ≈ 1.0  atol=1e-12
    for k in 2:20
        @test abs(h[k]) < 1e-10
    end
end
```

### 3.3 Leading coefficient (Jacobian)

$\alpha_i = f_i^{\prime}(x_i)$ is the leading Taylor coefficient.  Check that $f_i(\zeta) = \alpha_i \zeta + O(\zeta^2)$.

```julia
for arm in (:L, :R, :T)
    a = geom.arms[arm]
    @test a.f_series[1] ≈ a.α   # leading coeff of f_i matches stored α
    @test abs(a.α) > 0           # non-degenerate
end
```

### 3.4 $\mathbb{Z}_2$ symmetry: L vs R

Under $z \mapsto -z$, the L and R arms are swapped.  The local coordinate series should be related:

```julia
α_L = geom.arms.L.α
α_R = geom.arms.R.α
@test abs(α_L) ≈ abs(α_R)

# The f_series coefficients at even/odd orders should be related by ±1 signs.
# (Exact relation depends on the σ convention.)
```

### 3.5 Series vs direct evaluation

Evaluate $f^{\prime}(x_i + \zeta)$ for a small $\zeta$ both by summing the Taylor series and by direct evaluation of the closed-form SC derivative.  They should agree.

```julia
ζ_test = 0.01 + 0.02im    # a small displacement into H
for arm in (:L, :R, :T)
    x_i = geom.arms[arm].x
    z = x_i + ζ_test

    # Direct evaluation
    fp_direct = fprime_exact(z, sc)

    # From series: f'(z) = residue/ζ + a₀ + a₁ζ + ...
    # Sum the Laurent series of f' at ζ_test.
    # (This tests that the series coefficients are correct.)
    fp_series = evaluate(fprime_series_at(geom, arm), ζ_test)

    @test fp_direct ≈ fp_series  rtol=1e-8
end
```

---

## 4. `NeumannCoefficients`

### 4.1 Diagonal self-consistency

By construction, $F_m^{(i)}(g_i(\xi)) = \xi^{-m} + O(\xi^0)$.  The intermediate powers $\xi^{-m+1}, \ldots, \xi^{-1}$ must be **exactly zero**.

```julia
_, neumann = compute_geometry(1.0, 15)

for arm in (:L, :R, :T)
    for m in 1:10
        # 𝒩^{(i→i)}_{m,k} for k < 0 should be zero (by the definition of F_m).
        # The stored data starts at k=0, and the ξ^{-m} term is separated out.
        # Check that the composition indeed starts at ξ^0:
        # This is verified if we reconstruct F_m^{(i)}(g_i(ξ)) and check
        # that coefficients of ξ^{-m+1}, ..., ξ^{-1} are zero.
        #
        # In terms of stored Neumann coefficients, the self-action block
        # 𝒩^{(i→i)}_{m,k} represents the k ≥ 0 part.  The test is that
        # the composition procedure didn't produce spurious negative powers.
        # This is a structural test on the F_m computation.
    end
end
```

**Practical test:** Recompute $F_m^{(i)}(g_i(\xi))$ by composing the stored singular-part polynomial with $g_i$, and check that the coefficients of $\xi^{-m+1}, \ldots, \xi^{-1}$ vanish:

```julia
for arm in (:L, :R, :T)
    F_m_polys = compute_F_m_polys(geom, arm, 10)  # expose this for testing
    g = geom.arms[arm].g_series

    for m in 1:10
        # F_m(g(ξ)): compose polynomial in (z-x_i)⁻¹ with g(ξ)
        composed = compose_Fm_with_g(F_m_polys[m], g, geom.arms[arm].x)
        for k in (-m+1):(-1)
            @test abs(composed[k]) < 1e-10
        end
        # The ξ^{-m} coefficient should be 1 (by construction)
        @test composed[-m] ≈ 1.0  atol=1e-10
    end
end
```

### 4.2 $\mathbb{Z}_2$ symmetry

```julia
_, neumann = compute_geometry(1.0, 10)

# 𝒩^{(L→L)} = 𝒩^{(R→R)}
@test neumann.𝒩.LL ≈ neumann.𝒩.RR

# 𝒩^{(L→R)} = 𝒩^{(R→L)}
@test neumann.𝒩.LR ≈ neumann.𝒩.RL
```

### 4.3 Decay of off-diagonal Neumann coefficients

For large $m$ or $k$, the off-diagonal coefficients $\mathcal{N}^{(i \to j)}_{m,k}$ ($j \neq i$) should decay geometrically.  Check that the magnitude decreases:

```julia
for m in 1:10
    @test abs(neumann.𝒩.LR[m, 11]) < abs(neumann.𝒩.LR[m, 1])
    @test abs(neumann.𝒩.LT[m, 11]) < abs(neumann.𝒩.LT[m, 1])
end
```

### 4.4 Convergence with series order

Compute Neumann coefficients at order $N = 10$ and $N = 20$.  The coefficients $\mathcal{N}^{(i \to j)}_{m,k}$ for small $m, k$ should agree to high precision:

```julia
_, neum_10 = compute_geometry(1.0, 10)
_, neum_20 = compute_geometry(1.0, 20)

for m in 1:5, k in 0:5
    @test neum_10.𝒩.LT[m, k+1] ≈ neum_20.𝒩.LT[m, k+1]  rtol=1e-8
end
```

---

## 5. `FockSpace` — Basis Enumeration

### 5.1 State counting

At $R = 1$, $h_{\max} = 2$:

| Sector $n$ | $h_n$ | Max level | Partitions | Dim |
|-----------|-------|-----------|------------|-----|
| $0$       | $0$   | $2$       | $\varnothing, [1], [2], [1{,}1]$ | $4$ |
| $\pm 1$   | $1/2$ | $1$       | $\varnothing, [1]$ | $2$ |
| $\pm 2$   | $2$   | $0$       | $\varnothing$ | $1$ |

Total dimension: $1 + 2 + 4 + 2 + 1 = 10$.  Five sectors.

```julia
basis = build_fock_basis(1.0, 2.0)
@test length(keys(basis.states)) == 5
@test Set(keys(basis.states)) == Set([-2, -1, 0, 1, 2])
@test length(basis.states[0]) == 4
@test length(basis.states[1]) == 2
@test length(basis.states[-1]) == 2
@test length(basis.states[2]) == 1
@test length(basis.states[-2]) == 1
@test dim(basis.V) == 10
```

At $R = 1$, $h_{\max} = 3$:  additionally $n = 0$ gains level-3 partitions ($[3], [2{,}1], [1{,}1{,}1]$, dim $+3 = 7$), $n = \pm 1$ gains level-2 partitions ($[2], [1{,}1]$, dim $+2 = 4$), $n = \pm 2$ gains level-1 ($[1]$, dim $= 2$).

```julia
basis = build_fock_basis(1.0, 3.0)
@test length(basis.states[0]) == 7
@test length(basis.states[1]) == 4
@test length(basis.states[2]) == 2
@test dim(basis.V) == 7 + 4 + 4 + 2 + 2   # = 19
```

### 5.2 Partition ordering

States within each sector are ordered by level, then reverse-lexicographic:

```julia
basis = build_fock_basis(1.0, 3.0)
@test basis.states[0][1] == Int[]       # primary, level 0
@test basis.states[0][2] == [1]         # level 1
@test basis.states[0][3] == [2]         # level 2, revlex: [2] before [1,1]
@test basis.states[0][4] == [1, 1]      # level 2
@test basis.states[0][5] == [3]         # level 3
@test basis.states[0][6] == [2, 1]      # level 3
@test basis.states[0][7] == [1, 1, 1]   # level 3
```

### 5.3 Levels

```julia
basis = build_fock_basis(1.0, 3.0)
@test basis.levels[0] == [0, 1, 2, 2, 3, 3, 3]
@test basis.levels[1] == [0, 1, 2, 2]
```

### 5.4 Normalisation factors $z_\lambda$

$$z_\lambda = \prod_{j \ge 1} j^{m_j} \, m_j!$$

The unit-norm factors $z_\lambda$ are **not** stored on `FockBasis` — they
are absorbed into the $J_k$ matrix coefficients $\sqrt{k\,m_k}$ in
`JMatrices.jl`. The standalone helper `_compute_z_lambda(λ)` exists for
diagnostic/conversion code that needs the raw factor (e.g. converting
$J_{-\lambda}|0\rangle$ to the unit-normalised basis vector).

```julia
z = CFTTruncation._compute_z_lambda
@test z(Int[])         ≈ 1.0      # z_{∅} = 1
@test z([1])           ≈ 1.0      # z_{[1]} = 1¹·1! = 1
@test z([2])           ≈ 2.0      # z_{[2]} = 2¹·1! = 2
@test z([1, 1])        ≈ 2.0      # z_{[1,1]} = 1²·2! = 2
@test z([3])           ≈ 3.0      # z_{[3]} = 3¹·1! = 3
@test z([2, 1])        ≈ 2.0      # z_{[2,1]} = 2¹·1!·1¹·1! = 2
@test z([1, 1, 1])     ≈ 6.0      # z_{[1,1,1]} = 1³·3! = 6
```

### 5.5 Graded space

```julia
basis = build_fock_basis(1.0, 2.0)
V = basis.V
@test V isa GradedSpace
@test dim(V, U1Irrep(0)) == 4
@test dim(V, U1Irrep(1)) == 2
@test dim(V, U1Irrep(3)) == 0    # not present
```

---

## 6. `JMatrices` — Mode Action

All tests use the **normalised** basis $|\hat\lambda; n\rangle = J_{-\lambda}|n\rangle / \sqrt{z_\lambda}$.

### 6.1 $J_0$ is diagonal with eigenvalue $n/R$

```julia
basis = build_fock_basis(1.0, 3.0)   # R = 1
J = build_J_matrices(basis, 3)

# In sector n, J_0 = (n/R)·I
for n in keys(basis.states)
    d = length(basis.states[n])
    J0 = J.J_action[n][1]       # k=0 stored at index 1
    @test J0 ≈ (n / 1.0) * I(d)   # R = 1
end
```

### 6.2 $J_1$ matrix elements at $n = 0$

Basis at $n = 0$, $h_{\max} = 3$: $|\varnothing\rangle, |[1]\rangle, |[2]\rangle, |[1{,}1]\rangle, |[3]\rangle, |[2{,}1]\rangle, |[1{,}1{,}1]\rangle$.

$J_1$ removes a part equal to 1.  In the normalised basis, $J_1|\hat\lambda\rangle = \sqrt{m_1(\lambda)} \, |\widehat{\lambda \setminus 1}\rangle$:

| Source | $m_1(\lambda)$ | Target | Coefficient |
|--------|----------------|--------|-------------|
| $[1]$  | 1              | $\varnothing$ | $1$ |
| $[1{,}1]$ | 2          | $[1]$  | $\sqrt{2}$ |
| $[2{,}1]$ | 1          | $[2]$  | $1$ |
| $[1{,}1{,}1]$ | 3      | $[1{,}1]$ | $\sqrt{3}$ |

All others are zero.

```julia
J1 = J.J_action[0][2]   # k=1 at index 2, sector n=0
@test size(J1) == (7, 7)

# Check as sparse: only 4 nonzero entries
@test J1[1, 2] ≈ 1.0             # [1] → ∅
@test J1[2, 4] ≈ √2              # [1,1] → [1]
@test J1[3, 6] ≈ 1.0             # [2,1] → [2]
@test J1[4, 7] ≈ √3              # [1,1,1] → [1,1]
@test nnz(sparse(J1)) == 4
```

### 6.3 $J_2$ matrix elements at $n = 0$

$J_2$ removes a part equal to 2.  $J_2|\hat\lambda\rangle = \sqrt{2\,m_2(\lambda)} \, |\widehat{\lambda \setminus 2}\rangle$:

| Source | $m_2(\lambda)$ | Target | Coefficient |
|--------|----------------|--------|-------------|
| $[2]$  | 1              | $\varnothing$ | $\sqrt{2}$ |
| $[2{,}1]$ | 1          | $[1]$  | $\sqrt{2}$ |

```julia
J2 = J.J_action[0][3]   # k=2 at index 3
@test J2[1, 3] ≈ √2              # [2] → ∅
@test J2[2, 6] ≈ √2              # [2,1] → [1]
```

### 6.4 Commutation relation $[J_1, J_{-1}] = I$ (within truncation)

This holds exactly only when $J_{-1}$ doesn't push states outside the truncated space.  At $h_{\max} = 4$, $n = 0$, the max level is 4; states up to level 3 are safe (since $J_{-1}$ adds level 1).

```julia
basis = build_fock_basis(1.0, 4.0)
J = build_J_matrices(basis, 4)

# Build J_{-1} in sector n=0 (creation: adds a part 1)
# In the normalised basis: J_{-1}|λ̂⟩ = √(m_1(λ)+1) |λ∪{1}⟩_hat
J_minus1 = build_creation_matrix(basis, 0, 1)  # helper needed for testing

J1 = J.J_action[0][2]   # J_1
comm = J1 * J_minus1 - J_minus1 * J1

# Should be identity on the full truncated space at n=0
d = length(basis.states[0])
@test comm ≈ I(d)  atol=1e-12
```

**Note:** This test requires exposing `J_{-1}` (creation) matrices for testing, even though the recursion only uses $J_k$ with $k \ge 0$.

### 6.5 Commutation at $n \neq 0$

```julia
# [J_1, J_{-1}] = I on sector n=1 (where J_0 = 1/R, but this is a different check)
basis = build_fock_basis(1.0, 4.0)
J = build_J_matrices(basis, 4)
J_minus1 = build_creation_matrix(basis, 1, 1)
J1 = J.J_action[1][2]
comm = J1 * J_minus1 - J_minus1 * J1
d = length(basis.states[1])
@test comm ≈ I(d)  atol=1e-12
```

---

## 7. BPZ Bilinear Form and Conjugation Map

### 7.0 BPZ bilinear form $\eta_{\text{form}} : V_{\text{bond}} \otimes V_{\text{bond}} \to \mathbb{C}$

The BPZ bilinear form is a TensorMap with trivial codomain.

```julia
basis = build_fock_basis(1.0, 3.0)
η_form = build_bpz_form(basis)

# Codomain is trivial (ℂ), domain is V ⊗ V
@test codomain(η_form) == one(basis.V)
@test domain(η_form) == basis.V ⊗ basis.V

# Evaluate on two basis vectors: ⟨e_α, η, e_β⟩ = (-1)^level · δ_{αβ}
u = ket(basis, 0, 1)    # primary |∅; 0⟩, level 0
v = ket(basis, 0, 2)    # |[1]; 0⟩, level 1

@tensor val_uu[] := η_form[; a, b] * u[a] * u[b]
@test scalar(val_uu) ≈ 1.0     # (-1)^0 = +1

@tensor val_vv[] := η_form[; a, b] * v[a] * v[b]
@test scalar(val_vv) ≈ -1.0    # (-1)^1 = -1

@tensor val_uv[] := η_form[; a, b] * u[a] * v[b]
@test abs(scalar(val_uv)) < 1e-15   # orthogonal sectors
```

### 7.1 Conjugation map: diagonal with $(-1)^N$ entries

```julia
basis = build_fock_basis(1.0, 3.0)
η = build_bpz_map(basis)

for (f₁, f₂) in fusiontrees(η)
    n = Int(f₂.uncoupled[1].charge)
    blk = η[f₁, f₂]
    d = size(blk, 1)
    for α in 1:d
        @test blk[α, α] ≈ (-1.0)^basis.levels[n][α]
    end
    # Off-diagonal entries are zero
    for α in 1:d, β in 1:d
        α == β && continue
        @test abs(blk[α, β]) < 1e-15
    end
end
```

### 7.2 Specific values at $n = 0$

At $h_{\max} = 3$, sector $n = 0$, levels are $[0, 1, 2, 2, 3, 3, 3]$:

```julia
basis = build_fock_basis(1.0, 3.0)
η = build_bpz_map(basis)
blk = block(η, U1Irrep(0))   # 7×7 diagonal matrix

@test blk[1,1] ≈  1.0   # level 0: (-1)^0 = +1
@test blk[2,2] ≈ -1.0   # level 1: (-1)^1 = -1
@test blk[3,3] ≈  1.0   # level 2: (-1)^2 = +1
@test blk[4,4] ≈  1.0   # level 2: +1
@test blk[5,5] ≈ -1.0   # level 3: (-1)^3 = -1
@test blk[6,6] ≈ -1.0   # level 3: -1
@test blk[7,7] ≈ -1.0   # level 3: -1
```

### 7.3 Involution: $\eta^2 = \text{id}$

$\eta: V \to V^{\prime}$, and we can check $\eta^{\prime} \circ \eta = \text{id}_V$ (where $\eta^{\prime}: V^{\prime} \to V^{\prime\prime} \cong V$).  In practice, compose with the canonical isomorphism:

```julia
basis = build_fock_basis(1.0, 3.0)
η = build_bpz_map(basis)

# η' ∘ η should be identity on V
# In TensorKit: η is V' ← V, so η' (adjoint) is V ← V'.
# η' * η is V ← V, should be identity.
@tensor id_check[a; b] := conj(η[c; a]) * η[c; b]
# Or more simply: since η is real and diagonal with ±1,
# η† η = I (because (±1)² = 1).
@test id_check ≈ id(basis.V)
```

### 7.4 BPZ inner product of basis states

```julia
basis = build_fock_basis(1.0, 3.0)
η = build_bpz_map(basis)

u = ket(basis, 0, 1)    # primary |∅; 0⟩, level 0
@tensor val[] := u'[a] * η[a, b] * u[b]
@test scalar(val) ≈ 1.0     # (-1)^0 = +1

v = ket(basis, 0, 2)    # |[1]; 0⟩, level 1
@tensor val[] := v'[a] * η[a, b] * v[b]
@test scalar(val) ≈ -1.0    # (-1)^1 = -1

w = ket(basis, 0, 3)    # |[2]; 0⟩, level 2
@tensor val[] := w'[a] * η[a, b] * w[b]
@test scalar(val) ≈ 1.0     # (-1)^2 = +1
```

---

## 8. `PrimaryVertex`

### 8.1 Selection rule

```julia
geom, _ = compute_geometry(1.0, 5)
R = 1.0

# Momentum conservation: n_L + n_R + n_T = 0
@test primary_vertex(1, -1, 0, geom, R) != 0
@test primary_vertex(0, 0, 0, geom, R) != 0
@test primary_vertex(1, 0, 0, geom, R) == 0   # violates conservation
@test primary_vertex(1, 1, 1, geom, R) == 0
```

### 8.2 All-zero momenta

$$V_\ell^{(\text{prim})}(0, 0, 0) = |\alpha_L|^0 \cdot |\alpha_R|^0 \cdot |\alpha_T|^0 \cdot 2^{-0} = 1$$

```julia
@test primary_vertex(0, 0, 0, geom, R) ≈ 1.0
```

### 8.3 Non-trivial primary vertex

$$V_\ell^{(\text{prim})}(1, -1, 0) = |\alpha_L|^{2 \cdot 1/2} \cdot |\alpha_R|^{2 \cdot 1/2} \cdot |\alpha_T|^0 \cdot 2^{-(1/2 + 1/2 - 0)} = |\alpha_L| \cdot |\alpha_R| / 2$$

```julia
α_L = geom.arms.L.α
α_R = geom.arms.R.α
@test primary_vertex(1, -1, 0, geom, R) ≈ abs(α_L) * abs(α_R) / 2
```

### 8.4 Symmetry under L ↔ R

By the $\mathbb{Z}_2$ of the T-shape:

```julia
@test primary_vertex(1, -1, 0, geom, R) ≈ primary_vertex(-1, 1, 0, geom, R)
```

---

## 9. `Recursion` — Ward Identity

### 9.1 Level 0 matches primary vertex

The recursion at $N_{\text{tot}} = 0$ should reproduce the primary vertex exactly.

```julia
vd = compute_vertex(R=1.0, ℓ=1.0, trunc=TruncationSpec(2.0))
V = vd.vertex

# Extract the (n_L=0, n_R=0, n_T=0) block, primary entry:
B = charge_block(vd, 0, 0, 0)    # d_L × d_R × d_T array
@test B[1, 1, 1] ≈ 1.0            # V(|0⟩, |0⟩, |0⟩) = 1

# Another primary triple:
B2 = charge_block(vd, 1, -1, 0)
@test B2[1, 1, 1] ≈ primary_vertex(1, -1, 0, vd.geom, 1.0)
```

### 9.2 $V(J_{-1}|0\rangle, |0\rangle, |0\rangle) = 0$

All $J_k$ acting on $|0; 0\rangle$ give zero (since $n = 0$ and the primary is annihilated by $J_k$ for $k > 0$).  The Ward identity sum vanishes.

```julia
B = charge_block(vd, 0, 0, 0)
# State 2 in sector 0 is [1], i.e. J_{-1}|0⟩ (normalised).
@test abs(B[2, 1, 1]) < 1e-12   # V(J_{-1}|0⟩, |0⟩, |0⟩) = 0
```

### 9.3 $V(J_{-1}|0\rangle, J_{-1}|0\rangle, |0\rangle) = -\mathcal{N}^{(LR)}_{1,1}$

Derivation (all momenta zero, so $J_0$ contributions vanish):

Peel $J_{-1}$ from L arm.  Residual on L: $|\varnothing; 0\rangle$.  Prefactor = 1.

Ward identity sum over arms $j$ and modes $k$:
- $j = L$: residual is $|\varnothing; 0\rangle$.  $J_k|\varnothing; 0\rangle = 0$ for all $k \ge 0$ (since $n = 0$ and primary).  Zero.
- $j = R$: state is $|[1]; 0\rangle$.  $J_0 = 0$ (since $n = 0$).  $J_1|[1]; 0\rangle = |\varnothing; 0\rangle$ with coefficient 1.  $J_k = 0$ for $k \ge 2$.  Contribution: $-\mathcal{N}^{(LR)}_{1,1} \cdot 1 \cdot V(\varnothing, \varnothing, \varnothing) = -\mathcal{N}^{(LR)}_{1,1}$.
- $j = T$: state is $|\varnothing; 0\rangle$.  All zero.

Result: $V(J_{-1}|0\rangle, J_{-1}|0\rangle, |0\rangle) = -\mathcal{N}^{(LR)}_{1,1} \cdot 1 = -\mathcal{N}^{(LR)}_{1,1}$.

```julia
B = charge_block(vd, 0, 0, 0)
N_LR_11 = vd.neumann.𝒩.LR[1, 2]   # m=1, k=1 (stored at column k+1=2)
@test B[2, 2, 1] ≈ -N_LR_11       # indices: α_L=2 ([1]), α_R=2 ([1]), α_T=1 (∅)
```

### 9.4 $V(J_{-1}|1\rangle, |-1\rangle, |0\rangle)$ — nonzero $J_0$ contribution

Peel $J_{-1}$ from L.  Residual on L: $|1\rangle$ (primary).  Prefactor = 1.

- $j = L$: $J_0|1\rangle = (1/R)|1\rangle$.  Contribution: $-\mathcal{N}^{(LL)}_{1,0} \cdot (1/R) \cdot V_{\text{prim}}(1, -1, 0)$.
- $j = R$: $J_0|-1\rangle = (-1/R)|-1\rangle$.  Contribution: $-\mathcal{N}^{(LR)}_{1,0} \cdot (-1/R) \cdot V_{\text{prim}}(1, -1, 0)$.
- $j = T$: $J_0|0\rangle = 0$.  Zero.

Result:

$$V(J_{-1}|1\rangle, |-1\rangle, |0\rangle) = -\left[\frac{\mathcal{N}^{(LL)}_{1,0}}{R} - \frac{\mathcal{N}^{(LR)}_{1,0}}{R}\right] V_{\text{prim}}(1, -1, 0)$$

```julia
R = 1.0
V_prim = primary_vertex(1, -1, 0, vd.geom, R)
N_LL_10 = vd.neumann.𝒩.LL[1, 1]   # m=1, k=0
N_LR_10 = vd.neumann.𝒩.LR[1, 1]   # m=1, k=0

expected = -(N_LL_10 / R - N_LR_10 / R) * V_prim

B = charge_block(vd, 1, -1, 0)
@test B[2, 1, 1] ≈ expected       # α_L=2 (=[1] in sector 1), α_R=1 (=∅), α_T=1 (=∅)
```

### 9.5 Tensor structure of the vertex

The vertex is a trilinear form $V_{\text{phys}} \otimes V_{\text{bond}} \otimes V_{\text{bond}} \to \mathbb{C}$.  The derived operator form $V_{\text{phys}} \otimes V_{\text{bond}} \to V_{\text{bond}}$ is obtained by adjointing one bond leg and composing with the BPZ map.

```julia
vd = compute_vertex(R=1.0, ℓ=1.0, trunc=TruncationSpec(2.0))

# Trilinear form: codomain is ℂ, domain is V_phys ⊗ V_bond ⊗ V_bond
@test codomain(vd.vertex) == one(vd.basis_phys.V)
@test domain(vd.vertex) == vd.basis_phys.V ⊗ vd.basis_bond.V ⊗ vd.basis_bond.V

# Operator form: codomain is V_bond, domain is V_phys ⊗ V_bond
@test codomain(vd.vertex_op) == vd.basis_bond.V
@test domain(vd.vertex_op) == vd.basis_phys.V ⊗ vd.basis_bond.V
```

### 9.6 Block structure: zero outside conservation

The `TensorMap` should have no block for charge triples that violate $n_L + n_R + n_T = 0$.  This is structural (enforced by TensorKit), but verify:

```julia
V = vd.vertex
@test sectortype(V) == U1Irrep
# The number of fusion tree pairs equals the number of allowed (n_L, n_R, n_T) triples
n_blocks = length(collect(fusiontrees(V)))
n_expected = count((nL, nR) -> haskey(vd.basis_bond.states, nL) &&
                                haskey(vd.basis_bond.states, nR) &&
                                haskey(vd.basis_phys.states, -(nL+nR)),
                   Iterators.product(bond_sectors(vd), bond_sectors(vd)))
@test n_blocks == n_expected
```

### 9.7 Frobenius norm grows with truncation

A basic sanity check: the vertex should have nonzero norm, and adding more states should change it.

```julia
geom, neumann = compute_geometry(1.0, 10)
vd1 = compute_vertex(R=1.0, ℓ=1.0, trunc=TruncationSpec(1.0); geom, neumann)
vd2 = compute_vertex(R=1.0, ℓ=1.0, trunc=TruncationSpec(2.0); geom, neumann)

@test norm(vd1.vertex) > 0
@test norm(vd2.vertex) > norm(vd1.vertex)
```

### 9.8 Different bond and physical truncations

```julia
vd_asym = compute_vertex(R=1.0, ℓ=1.0, trunc=TruncationSpec(h_bond=2.0, h_phys=3.0))

@test bond_dim(vd_asym) < phys_dim(vd_asym)
# Operator form: codomain is V_bond
@test dim(codomain(vd_asym.vertex_op)) == bond_dim(vd_asym)

# Primary vertex should still be correct
B = charge_block(vd_asym, 0, 0, 0)
@test B[1, 1, 1] ≈ 1.0
```

---

## 10. Integration Tests

### 10.1 $\ell$-independence of primary vertex structure

The primary vertex has the form $V = G(\ell) \cdot C_{LRT}$ where $G$ depends on geometry and $C$ is the OPE coefficient.  For the free boson $C = 1$.  Check that the geometric prefactor varies smoothly:

```julia
vals = Float64[]
for ℓ in [0.5, 1.0, 1.5, 2.0]
    vd = compute_vertex(R=1.0, ℓ=ℓ, trunc=TruncationSpec(2.0))
    B = charge_block(vd, 1, -1, 0)
    push!(vals, real(B[1, 1, 1]))
end
# Should be smoothly varying, nonzero, and finite
@test all(isfinite, vals)
@test all(v -> v != 0, vals)
```

### 10.2 $\mathbb{Z}_2$ symmetry (L $\leftrightarrow$ R) at generic $\ell$

The T-shape has $\mathbb{Z}_2$ symmetry $z \mapsto -z$ swapping L and R for all $\ell$.  The vertex should satisfy:

$$V(n_L, n_R, n_T)[\alpha_L, \alpha_R, \alpha_T] = V(n_R, n_L, n_T)[\alpha_R, \alpha_L, \alpha_T]$$

```julia
for ℓ in [0.5, 1.0, 2.0]
    vd = compute_vertex(R=1.0, ℓ=ℓ, trunc=TruncationSpec(3.0))
    B1 = charge_block(vd, 1, -1, 0)
    B2 = charge_block(vd, -1, 1, 0)
    @test norm(B1) ≈ norm(B2)
    # More precisely, B1[α_L, α_R, α_T] = B2[α_R, α_L, α_T]
    # (with matching basis ordering across sectors ±1)
end
```
