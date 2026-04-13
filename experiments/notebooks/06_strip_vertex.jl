### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ a0000001-0001-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ a0000001-0002-0000-0000-000000000001
using CFTTruncation

# ╔═╡ a0000001-0003-0000-0000-000000000001
using TensorKit

# ╔═╡ a0000001-0004-0000-0000-000000000001
using Printf

# ╔═╡ a0000001-0011-0000-0000-000000000001
begin
    R_val = 1.0
    h_max_val = 6.0
    series_order = 20
end

# ╔═╡ a0000001-0012-0000-0000-000000000001
basis = CFTTruncation.build_fock_basis(R_val, h_max_val)

# ╔═╡ a0000001-0022-0000-0000-000000000001
"""
Laurent-expand f'(z) = (2/π)/(z²-1) at z = x₀ + ζ.

At x₀ = 1:  f' = (1/π) ζ⁻¹ · 1/(1+ζ/2) = (1/π) Σ (-1/2)ⁿ ζⁿ⁻¹
At x₀ = -1: f' = -(1/π) ζ⁻¹ · 1/(1-ζ/2) = -(1/π) Σ (1/2)ⁿ ζⁿ⁻¹
"""
function expand_strip_fprime(x₀::Float64, order::Int)
    prec = order + 1
    coeffs = zeros(ComplexF64, prec)
    if x₀ == 1.0
        # Residue = +1/π, geometric ratio = -1/2
        for n in 0:prec-1
            coeffs[n+1] = (1/π) * (-1/2)^n
        end
    elseif x₀ == -1.0
        # Residue = -1/π, geometric ratio = +1/2
        for n in 0:prec-1
            coeffs[n+1] = -(1/π) * (1/2)^n
        end
    end
    CFTTruncation.TruncLaurent(-1, coeffs, prec - 1)
end

# ╔═╡ a0000001-0024-0000-0000-000000000001
struct StripArmData
    label::Symbol
    x::Float64
    σ::ComplexF64
    α::ComplexF64
    f_series::CFTTruncation.TruncLaurent{ComplexF64}
    g_series::CFTTruncation.TruncLaurent{ComplexF64}
end

# ╔═╡ a0000001-0025-0000-0000-000000000001
function build_strip_arm(label::Symbol, order::Int)
    x = label == :L ? 1.0 : -1.0
    # σ is determined by residue of f' at x_i: w·σ/π = Res
    # At z=+1: Res = +1/π → σ_L = +1
    # At z=-1: Res = -1/π → σ_R = -1
    σ = label == :L ? 1.0 + 0.0im : -1.0 + 0.0im
    w = 1.0
    σ_star = -1.0 / σ

    prec = order + 1
    fprime = expand_strip_fprime(x, order)

    # Extract regular part: f'(x+ζ) = Res/ζ + regular
    # ρ'(ζ) = regular part of f'
    # ρ(ζ) = ρ₀ + Σ_{n≥1} a_{n-1}/n · ζⁿ
    rho_coeffs = zeros(ComplexF64, prec)
    rho_coeffs[1] = zero(ComplexF64)  # ρ₀ = 0 (integration constant)
    for n in 1:prec-1
        # Regular part coefficient at ζⁿ⁻¹ is fprime.coeffs[n+1] (the n-th regular coeff)
        if n + 1 <= length(fprime.coeffs)
            rho_coeffs[n+1] = fprime.coeffs[n+1] / n
        end
    end

    # Build f_i(ζ) = ζ · exp(coeff_factor · ρ(ζ))
    coeff_factor = -π * σ_star / w  # L: π, R: -π

    exp_input_coeffs = zeros(ComplexF64, prec - 1)
    for n in 1:prec-1
        exp_input_coeffs[n] = coeff_factor * rho_coeffs[n+1]
    end
    exp_input = CFTTruncation.TruncLaurent(1, exp_input_coeffs, prec)
    exp_part = CFTTruncation.exp_series(exp_input)

    α_factor = exp(coeff_factor * rho_coeffs[1])

    fi_coeffs = ComplexF64[α_factor * c for c in exp_part.coeffs]
    f_series = CFTTruncation.TruncLaurent(1, fi_coeffs, prec)

    α_val = f_series[1]

    g_series = CFTTruncation.series_revert(f_series)

    StripArmData(label, x, σ, α_val, f_series, g_series)
end

# ╔═╡ a0000001-0026-0000-0000-000000000001
begin
    arm_L = build_strip_arm(:L, series_order)
    arm_R = build_strip_arm(:R, series_order)
    @printf("α_L = %.10f,  α_R = %.10f\n", real(arm_L.α), real(arm_R.α))
    @printf("|α_L · α_R| = %.10f\n", abs(arm_L.α * arm_R.α))
    @printf("|x_L - x_R| = %.1f\n", abs(arm_L.x - arm_R.x))
end

# ╔═╡ a0000001-0031-0000-0000-000000000001
function compute_strip_F_polys(arm::StripArmData, m_max::Int, prec::Int)
    fi = arm.f_series
    fi_inv = inv(fi)
    polys = Vector{CFTTruncation.TruncLaurent{ComplexF64}}(undef, m_max)
    current = fi_inv
    for m in 1:m_max
        polys[m] = CFTTruncation.singular_part(current)
        if m < m_max
            current = CFTTruncation._truncmul(current, fi_inv, fi.prec)
        end
    end
    polys
end

# ╔═╡ a0000001-0032-0000-0000-000000000001
function compose_Fm_strip(Fm, arm_i::StripArmData, arm_j::StripArmData, order::Int)
    prec = order
    if arm_i.label == arm_j.label
        # Same arm: compose F_m(g_i(ξ))
        g = arm_j.g_series
        result = CFTTruncation.TruncLaurent(0, zeros(ComplexF64, prec), prec)
        ginv = inv(g)
        for n in 1:length(Fm.coeffs)
            power = Fm.val + n - 1
            neg_power = -power
            g_to_power = CFTTruncation._power_of(ginv, neg_power, prec)
            coeff = Fm.coeffs[n]
            if coeff != zero(ComplexF64)
                scaled = CFTTruncation.TruncLaurent(g_to_power.val,
                    ComplexF64[coeff * c for c in g_to_power.coeffs], prec)
                result = CFTTruncation._add(result, scaled)
            end
        end
        return result
    else
        # Different arm: (z - x_i) = (x_j - x_i) + g_j(ξ)
        Δx = ComplexF64(arm_j.x - arm_i.x)
        g = arm_j.g_series
        w_coeffs = zeros(ComplexF64, prec)
        w_coeffs[1] = Δx
        for k in 1:min(length(g.coeffs), prec - 1)
            w_coeffs[k + 1] = g.coeffs[k]
        end
        w = CFTTruncation.TruncLaurent(0, w_coeffs, prec)
        winv = inv(w)
        result = CFTTruncation.TruncLaurent(0, zeros(ComplexF64, prec), prec)
        for n in 1:length(Fm.coeffs)
            power = Fm.val + n - 1
            neg_power = -power
            w_to_neg_power = CFTTruncation._power_of(winv, neg_power, prec)
            coeff = Fm.coeffs[n]
            if coeff != zero(ComplexF64)
                scaled = CFTTruncation.TruncLaurent(w_to_neg_power.val,
                    ComplexF64[coeff * c for c in w_to_neg_power.coeffs], prec)
                result = CFTTruncation._add(result, scaled)
            end
        end
        return result
    end
end

# ╔═╡ a0000001-0033-0000-0000-000000000001
function compute_strip_neumann(arm_L, arm_R, m_max, order)
    arms = (L=arm_L, R=arm_R)
    F_polys = Dict{Symbol, Vector}()
    for (label, arm) in pairs(arms)
        F_polys[label] = compute_strip_F_polys(arm, m_max, order + 1)
    end

    k_max = m_max
    𝒩 = Dict{Symbol, Matrix{Float64}}()
    for i in (:L, :R), j in (:L, :R)
        key = Symbol(i, j)
        mat = zeros(Float64, m_max, k_max)
        a_i = getfield(arms, i)
        a_j = getfield(arms, j)
        for m in 1:m_max
            composed = compose_Fm_strip(F_polys[i][m], a_i, a_j, order)
            for k in 0:k_max-1
                mat[m, k+1] = real(composed[k])
            end
        end
        𝒩[key] = mat
    end
    𝒩
end

# ╔═╡ a0000001-0041-0000-0000-000000000001
function strip_primary_vertex(n_L::Int, n_R::Int, arm_L, arm_R, R::Float64)
    n_L + n_R == 0 || return 0.0
    h = (n_L / R)^2 / 2
    α_L = abs(arm_L.α)
    α_R = abs(arm_R.α)
    d_LR = abs(arm_L.x - arm_R.x)  # = 2
    α_L^(2h) * α_R^(2h) / d_LR^(2h)
end

# ╔═╡ a0000001-0042-0000-0000-000000000001
let
    lines = ["  n_L    h         V_prim        |α|^{2h}/|Δx|^{2h}"]
    push!(lines, repeat("-", 55))
    for n in 0:3
        vp = strip_primary_vertex(n, -n, arm_L, arm_R, R_val)
        h = (n / R_val)^2 / 2
        push!(lines, @sprintf("  %d      %.1f       %.8f", n, h, vp))
    end
    Base.Text(join(lines, "\n"))
end

# ╔═╡ a0000001-0051-0000-0000-000000000001
"""
Dense 2D vertex array indexed by (n_L, n_R, α_L, α_R).
Parallel to VertexArray in src/Recursion.jl but for 2 arms.
"""
struct StripVertexArray
    data::Vector{Float64}
    offsets::Matrix{Int}
    dims::Matrix{Tuple{Int,Int}}
    n_off::Int
end

# ╔═╡ a0000001-0052-0000-0000-000000000001
function build_strip_vertex_array(basis)
    sectors = sort(collect(keys(basis.states)))
    n_min = minimum(sectors)
    n_max = maximum(sectors)
    n_off = 1 - n_min
    n_range = n_max - n_min + 1

    offsets = zeros(Int, n_range, n_range)
    dims = fill((0, 0), n_range, n_range)
    total = 0
    for n_L in sectors
        n_R = -n_L
        n_R in sectors || continue
        d_L = length(basis.states[n_L])
        d_R = length(basis.states[n_R])
        offsets[n_L + n_off, n_R + n_off] = total
        dims[n_L + n_off, n_R + n_off] = (d_L, d_R)
        total += d_L * d_R
    end
    StripVertexArray(zeros(Float64, total), offsets, dims, n_off)
end

# ╔═╡ a0000001-0053-0000-0000-000000000001
@inline function Base.getindex(va::StripVertexArray, n_L::Int, n_R::Int, α_L::Int, α_R::Int)
    @inbounds begin
        off = va.offsets[n_L + va.n_off, n_R + va.n_off]
        off == 0 && return 0.0
        _, d_R = va.dims[n_L + va.n_off, n_R + va.n_off]
        va.data[off + (α_L - 1) * d_R + α_R]
    end
end

# ╔═╡ a0000001-0010-0000-0000-000000000001
md"""
# 06 — Strip 2-Point Vertex (Sanity Check)

Compute the **2-point vertex** (strip propagator) using the same pipeline as the
T-vertex: SC map → local coordinates → Neumann coefficients → primary vertex →
Ward recursion.

The geometry is a strip of width $\pi$ with BCFT walls on both sides:

```
  ═══════════════════  (BCFT wall, Im w = π)
  ←── arm R          arm L ──→
  ═══════════════════  (BCFT wall, Im w = 0)
```

The SC map from UHP is $f(z) = \frac{1}{\pi}\log\frac{z-1}{z+1}$, with
punctures at $z = 1$ (arm L) and $z = -1$ (arm R).

**Expected result**: The vertex is $V : \mathbb{C} \leftarrow V \otimes V$, and
it should equal the BPZ form $\eta$ times a propagator $e^{-d \cdot L_0}$:

$$V_{\alpha\beta}^{(n_L, n_R)} = \delta_{n_L+n_R, 0} \cdot (-1)^N \cdot A \cdot e^{-d(h+N)}$$

The propagation distance $d$ and overall factor $A$ are determined by the
geometry (the $\alpha_i$ coefficients).
"""

# ╔═╡ a0000001-0020-0000-0000-000000000001
md"""
## Step 1: SC Map and Local Coordinates

The SC derivative for the strip is
$$f'(z) = \frac{2/\pi}{z^2 - 1}$$

Poles at $z = \pm 1$ with residues $\pm 1/\pi$ (same as the T-vertex L/R arms).
"""

# ╔═╡ a0000001-0021-0000-0000-000000000001
md"""
### Laurent expansion of $f'$ at $z = 1$ and $z = -1$

At $z = 1 + \zeta$:

$$f'(1+\zeta) = \frac{2/\pi}{\zeta(2+\zeta)} = \frac{1}{\pi\zeta} \cdot \frac{1}{1 + \zeta/2}$$

At $z = -1 + \zeta$:

$$f'(-1+\zeta) = \frac{2/\pi}{\zeta(-2+\zeta)} = \frac{-1}{\pi\zeta} \cdot \frac{1}{1 - \zeta/2}$$
"""

# ╔═╡ a0000001-0023-0000-0000-000000000001
md"""
### Build local coordinates $f_i(\zeta)$ and their inverses $g_i(\xi)$

Following exactly the same recipe as `LocalCoordinates.jl`:

$f_i(\zeta) = \zeta \cdot \exp\!\left(\frac{-\pi \sigma^*}{w} \cdot \rho(\zeta)\right)$

where $\rho(\zeta) = \int (f'(x_i+\zeta) - \text{Res}/\zeta)\,d\zeta$ is the regular part.
"""

# ╔═╡ a0000001-0030-0000-0000-000000000001
md"""
## Step 2: Neumann Coefficients

Only 4 blocks: LL, LR, RL, RR. Same recipe as `NeumannCoefficients.jl`:
compute $F_m^{(i)}(\zeta) = (f_i(\zeta))^{-m}\big|_{\text{singular part}}$,
then compose with $g_j$.
"""

# ╔═╡ a0000001-0034-0000-0000-000000000001
begin
    m_max_val = Int(h_max_val) + 2
    𝒩_strip = compute_strip_neumann(arm_L, arm_R, m_max_val, series_order)

    md"Neumann coefficients computed. m\_max = $(m_max_val)"
end

# ╔═╡ a0000001-0036-0000-0000-000000000001
let
    lines = ["  m    N^LL_{m,0}    N^LR_{m,0}    N^RL_{m,0}    N^RR_{m,0}"]
    push!(lines, repeat("-", 70))
    for m in 1:min(8, m_max_val)
        @sprintf("  %d    %10.6f    %10.6f    %10.6f    %10.6f",
            m, 𝒩_strip[:LL][m,1], 𝒩_strip[:LR][m,1],
            𝒩_strip[:RL][m,1], 𝒩_strip[:RR][m,1]) |> l -> push!(lines, l)
    end
    Base.Text(join(lines, "\n"))
end

# ╔═╡ a0000001-0035-0000-0000-000000000001
md"""
### Check: $N^{LR}_{m,0}$ (the "zero-mode" Neumann coefficients)

For the strip propagator, $N^{LR}_{m,0}$ encodes the conformal factor.
$N^{LL}_{m,k}$ should satisfy the self-Neumann relations.
"""

# ╔═╡ a0000001-0040-0000-0000-000000000001
md"""
## Step 3: Primary Vertex (2-point)

The boundary 2-point function on UHP:
$$\langle \phi_{h_L}(x_L) \, \phi_{h_R}(x_R) \rangle = \frac{\delta_{n_L+n_R,0}}{|x_L - x_R|^{2h}}$$

With Jacobian factors:
$$V_{\text{prim}} = |α_L|^{2h_L} \, |α_R|^{2h_R} \cdot \frac{\delta_{n_L+n_R,0}}{|x_L - x_R|^{2h}}$$

For the compact boson, $h_L = h_R = h = (n/R)^2/2$ when $n_L + n_R = 0$.
"""

# ╔═╡ a0000001-0050-0000-0000-000000000001
md"""
## Step 4: Ward Identity Recursion (2 arms)

Same recursion as the T-vertex, but with only 2 arms (L, R).
The Ward identity:

$$V(\ldots, J_{-m}|\mu\rangle_{\text{arm}}, \ldots) = -\sum_{j \in \{L,R\}} \sum_{k \geq 0} N^{\text{arm}\to j}_{m,k} \, V(\ldots, J_k \text{ on arm } j, \ldots)$$
"""

# ╔═╡ a0000001-0054-0000-0000-000000000001
@inline function Base.setindex!(va::StripVertexArray, val::Float64, n_L::Int, n_R::Int, α_L::Int, α_R::Int)
    @inbounds begin
        off = va.offsets[n_L + va.n_off, n_R + va.n_off]
        _, d_R = va.dims[n_L + va.n_off, n_R + va.n_off]
        va.data[off + (α_L - 1) * d_R + α_R] = val
    end
end

# ╔═╡ a0000001-0056-0000-0000-000000000001
function strip_recurse_entry(t::NTuple{4,Int}, raw::StripVertexArray, basis, 𝒩, J_sp)
    n_L, n_R, α_L, α_R = t
    λ_L = basis.states[n_L][α_L]
    λ_R = basis.states[n_R][α_R]

    # Pick an arm with nonempty partition (prefer L)
    arm = :none; m = 0
    if !isempty(λ_L)
        arm = :L; m = λ_L[1]
    elseif !isempty(λ_R)
        arm = :R; m = λ_R[1]
    else
        error("All partitions empty but total level > 0")
    end

    # Remove one copy of m to get residual state
    λ_arm = arm == :L ? λ_L : λ_R
    mk = count(==(m), λ_arm)
    norm_factor = 1.0 / sqrt(m * mk)

    μ = CFTTruncation._remove_part(λ_arm, m)
    n_arm = arm == :L ? n_L : n_R
    α_new = basis.partition_index[n_arm][μ]
    if arm == :L
        t_res = (n_L, n_R, α_new, α_R)
    else
        t_res = (n_L, n_R, α_L, α_new)
    end

    # Ward identity: sum over target arms j and mode k
    result = 0.0
    for j in (:L, :R)
        key = Symbol(arm, j)
        mat = 𝒩[key]
        m > size(mat, 1) && continue
        for k in 0:size(mat, 2)-1
            N_coeff = mat[m, k+1]
            N_coeff == 0.0 && continue

            # Apply J_k on arm j
            n_j = j == :L ? t_res[1] : t_res[2]
            α_j = j == :L ? t_res[3] : t_res[4]
            (target, coeff) = J_sp[n_j][k+1][α_j]
            target == 0 && continue

            if j == :L
                v = raw[t_res[1], t_res[2], target, t_res[4]]
            else
                v = raw[t_res[1], t_res[2], t_res[3], target]
            end
            result += N_coeff * coeff * v
        end
    end

    -norm_factor * result
end

# ╔═╡ a0000001-0055-0000-0000-000000000001
function compute_strip_vertex_raw(basis, arm_L, arm_R, 𝒩, J_sp, R)
    raw = build_strip_vertex_array(basis)
    sectors = collect(keys(basis.states))

    # Collect all valid pairs
    pairs_list = NTuple{4, Int}[]
    for n_L in sectors
        n_R = -n_L
        n_R in sectors || continue
        for α_L in eachindex(basis.states[n_L])
            for α_R in eachindex(basis.states[n_R])
                push!(pairs_list, (n_L, n_R, α_L, α_R))
            end
        end
    end

    # Sort by total level
    total_level(t) = basis.levels[t[1]][t[3]] + basis.levels[t[2]][t[4]]
    sort!(pairs_list, by=total_level)

    # Primary vertex (level 0)
    for t in pairs_list
        total_level(t) > 0 && break
        raw[t[1], t[2], t[3], t[4]] = strip_primary_vertex(t[1], t[2], arm_L, arm_R, R)
    end

    # Recursion (level ≥ 1)
    for t in pairs_list
        total_level(t) == 0 && continue
        raw[t[1], t[2], t[3], t[4]] = strip_recurse_entry(
            t, raw, basis, 𝒩, J_sp)
    end

    raw
end

# ╔═╡ a0000001-0057-0000-0000-000000000001
md"### Build sparse J matrices and run recursion"

# ╔═╡ a0000001-0058-0000-0000-000000000001
begin
    J_dense, J_sp = CFTTruncation.build_J_matrices(basis, m_max_val)
    raw_strip = compute_strip_vertex_raw(basis, arm_L, arm_R, 𝒩_strip, J_sp, R_val)
    md"Strip vertex computed."
end

# ╔═╡ a0000001-0060-0000-0000-000000000001
md"""
## Step 5: Analysis — Is it a BPZ-signed propagator?

If the vertex is $V_{αβ} = (-1)^N \cdot A \cdot e^{-d(h+N)} \cdot \delta_{αβ}$, then:

1. **Diagonality**: Off-diagonal elements should be zero.
2. **BPZ sign**: Ratio $V_{αα}/|V_{αα}|$ should be $(-1)^N$.
3. **Propagation distance**: $\log(|V_{00}|/|V_{11}|) = d$ (from level 0 to level 1).
"""

# ╔═╡ a0000001-0061-0000-0000-000000000001
let
    lines = String[]
    push!(lines, "="^80)
    push!(lines, "  SECTOR n=0 (vacuum module): Raw vertex block")
    push!(lines, "="^80)

    n = 0
    parts = basis.states[n]
    d = length(parts)
    push!(lines, "\n  States (first 8):")
    for i in 1:min(8, d)
        push!(lines, @sprintf("    α=%d: λ=%s, level=%d, h+N=%.1f",
            i, string(parts[i]), basis.levels[n][i],
            (n/R_val)^2/2 + basis.levels[n][i]))
    end

    push!(lines, "\n  Vertex matrix V[α_L, α_R] (first 8×8):")
    dshow = min(8, d)
    # Header
    hdr = "        "
    for j in 1:dshow
        hdr *= @sprintf("  α_R=%-4d", j)
    end
    push!(lines, hdr)

    for i in 1:dshow
        row = @sprintf("  α_L=%d ", i)
        for j in 1:dshow
            v = raw_strip[n, -n, i, j]
            row *= @sprintf(" %9.5f", v)
        end
        push!(lines, row)
    end

    push!(lines, "\n  Diagonal entries and signs:")
    for i in 1:min(8, d)
        v = raw_strip[n, -n, i, i]
        level = basis.levels[n][i]
        expected_sign = (-1)^level
        actual_sign = v > 0 ? +1 : -1
        push!(lines, @sprintf("    α=%d: V=%12.8f  level=%d  sign=%+d  expected=%+d  %s",
            i, v, level, actual_sign, expected_sign,
            actual_sign == expected_sign ? "✓" : "✗"))
    end

    # Off-diagonal fraction
    diag_sq = sum(raw_strip[n, -n, i, i]^2 for i in 1:d)
    total_sq = sum(raw_strip[n, -n, i, j]^2 for i in 1:d, j in 1:d)
    offdiag = sqrt(max(0, 1 - diag_sq / total_sq))
    push!(lines, @sprintf("\n  Off-diagonal fraction: %.2e", offdiag))

    Base.Text(join(lines, "\n"))
end

# ╔═╡ a0000001-0062-0000-0000-000000000001
md"""
### Extract propagation distance $d$

From the diagonal eigenvalues: $|V_{αα}| = A \cdot e^{-d(h+N)}$
"""

# ╔═╡ a0000001-0063-0000-0000-000000000001
let
    lines = String[]
    push!(lines, "="^80)
    push!(lines, "  PROPAGATION DISTANCE EXTRACTION")
    push!(lines, "="^80)

    n = 0
    parts = basis.states[n]
    h0 = (n / R_val)^2 / 2  # = 0 for n=0

    v0 = abs(raw_strip[n, -n, 1, 1])  # level 0
    v1 = abs(raw_strip[n, -n, 2, 2])  # level 1

    d_01 = log(v0 / v1)
    push!(lines, @sprintf("\n  |V[1,1]| (level 0) = %.10f", v0))
    push!(lines, @sprintf("  |V[2,2]| (level 1) = %.10f", v1))
    push!(lines, @sprintf("  d = log(|V₀₀|/|V₁₁|) = %.10f", d_01))
    push!(lines, @sprintf("  log(4) = %.10f", log(4)))
    push!(lines, @sprintf("  2·log(2) = %.10f", 2*log(2)))

    # Check consistency: d should be the same from any pair of adjacent levels
    push!(lines, "\n  Consistency check (d from adjacent levels):")
    for i in 1:min(6, length(parts)-1)
        vi = abs(raw_strip[n, -n, i, i])
        vj = abs(raw_strip[n, -n, i+1, i+1])
        if vi > 0 && vj > 0
            Ni = basis.levels[n][i]
            Nj = basis.levels[n][i+1]
            ΔN = Nj - Ni
            d_ij = log(vi / vj) / ΔN
            push!(lines, @sprintf("    levels %d→%d (ΔN=%d): d = %.10f", Ni, Nj, ΔN, d_ij))
        end
    end

    # Check: what is d in terms of geometric quantities?
    α_L = abs(arm_L.α)
    α_R = abs(arm_R.α)
    push!(lines, @sprintf("\n  Geometric quantities:"))
    push!(lines, @sprintf("    |α_L| = %.10f", α_L))
    push!(lines, @sprintf("    |α_R| = %.10f", α_R))
    push!(lines, @sprintf("    -log(|α_L|²) = %.10f", -2*log(α_L)))
    push!(lines, @sprintf("    -log(|α_R|²) = %.10f", -2*log(α_R)))
    push!(lines, @sprintf("    -log(|α_L·α_R|²) = %.10f", -2*log(α_L*α_R)))
    push!(lines, @sprintf("    -2·log(|α_L·α_R|/|Δx|) = %.10f",
        -2*log(α_L*α_R/abs(arm_L.x - arm_R.x))))

    # The expected formula: V_{00} for h=0 primaries is 1 (from primary vertex).
    # The eigenvalue at level N is V_{00} * e^{-d*N}, so:
    # e^{-d} = V_{11}/V_{00} (up to BPZ sign)
    # d = -log(|V_{11}/V_{00}|) which we already computed
    push!(lines, @sprintf("\n  Overall scale: V[1,1] (vacuum) = %.10f",
        raw_strip[n, -n, 1, 1]))

    Base.Text(join(lines, "\n"))
end

# ╔═╡ a0000001-0064-0000-0000-000000000001
md"""
### Check other charge sectors
"""

# ╔═╡ a0000001-0065-0000-0000-000000000001
let
    lines = String[]
    push!(lines, "  n_L    n_R    V_prim (level 0)     |V[1,1]|/V_prim    sign")
    push!(lines, repeat("-", 70))
    for n_L in sort(collect(keys(basis.states)))
        n_R = -n_L
        haskey(basis.states, n_R) || continue
        vp = strip_primary_vertex(n_L, n_R, arm_L, arm_R, R_val)
        v00 = raw_strip[n_L, n_R, 1, 1]
        ratio = vp != 0 ? abs(v00) / abs(vp) : NaN
        s = v00 >= 0 ? "+" : "-"
        push!(lines, @sprintf("  %+d     %+d     %12.8f         %8.5f          %s",
            n_L, n_R, vp, ratio, s))
    end
    Base.Text(join(lines, "\n"))
end

# ╔═╡ a0000001-0070-0000-0000-000000000001
md"""
## Step 6: Compare with direct BPZ × propagator

Construct the expected answer:
$$V^{\text{expected}}_{\alpha\beta}(n_L, n_R) = \delta_{n_L+n_R,0} \cdot \eta_{\alpha\beta} \cdot e^{-d(h+N)}$$

where $\eta_{\alpha\beta} = (-1)^N \delta_{\alpha\beta}$ is the BPZ form and
$d$ is extracted from the level-0/level-1 ratio.
"""

# ╔═╡ a0000001-0071-0000-0000-000000000001
let
    # Extract d from n=0 sector
    v0 = abs(raw_strip[0, 0, 1, 1])
    v1 = abs(raw_strip[0, 0, 2, 2])
    d = log(v0 / v1)

    lines = String[]
    push!(lines, "="^80)
    push!(lines, "  COMPARISON: recursion vs BPZ × propagator")
    push!(lines, @sprintf("  Using d = %.10f", d))
    push!(lines, "="^80)

    max_err = 0.0
    max_rel_err = 0.0
    n_checked = 0

    for n_L in sort(collect(keys(basis.states)))
        n_R = -n_L
        haskey(basis.states, n_R) || continue
        h = (n_L / R_val)^2 / 2

        d_L = length(basis.states[n_L])
        d_R = length(basis.states[n_R])

        for α_L in 1:d_L, α_R in 1:d_R
            v_rec = raw_strip[n_L, n_R, α_L, α_R]

            # Expected: δ_{αL,αR} · (-1)^N · A(h) · e^{-d(h+N)}
            # A(h) comes from the primary vertex (α factors)
            if α_L == α_R && n_L + n_R == 0
                N_level = basis.levels[n_L][α_L]
                A_h = strip_primary_vertex(n_L, n_R, arm_L, arm_R, R_val) * exp(d * h)
                v_exp = (-1)^N_level * A_h * exp(-d * (h + N_level))
            else
                v_exp = 0.0
            end

            err = abs(v_rec - v_exp)
            max_err = max(max_err, err)
            if abs(v_exp) > 1e-15
                max_rel_err = max(max_rel_err, err / abs(v_exp))
            end
            n_checked += 1
        end
    end

    push!(lines, @sprintf("\n  Entries checked: %d", n_checked))
    push!(lines, @sprintf("  Max absolute error: %.2e", max_err))
    push!(lines, @sprintf("  Max relative error: %.2e", max_rel_err))

    # Show a few sample comparisons
    push!(lines, "\n  Sample comparisons (n=0 sector):")
    push!(lines, "  α_L  α_R  level  V_recursion      V_expected       |diff|")
    push!(lines, repeat("-", 70))
    h = 0.0
    A_h = strip_primary_vertex(0, 0, arm_L, arm_R, R_val) * exp(d * h)
    for α in 1:min(6, length(basis.states[0]))
        N = basis.levels[0][α]
        v_rec = raw_strip[0, 0, α, α]
        v_exp = (-1)^N * A_h * exp(-d * (h + N))
        push!(lines, @sprintf("  %d    %d    %d      %12.8f    %12.8f    %.2e",
            α, α, N, v_rec, v_exp, abs(v_rec - v_exp)))
    end

    # Off-diagonal samples
    push!(lines, "\n  Off-diagonal samples (should be 0):")
    for α_L in 1:min(4, length(basis.states[0]))
        for α_R in 1:min(4, length(basis.states[0]))
            α_L == α_R && continue
            v = raw_strip[0, 0, α_L, α_R]
            abs(v) < 1e-15 && continue
            push!(lines, @sprintf("  α_L=%d  α_R=%d  V = %.2e", α_L, α_R, v))
        end
    end

    Base.Text(join(lines, "\n"))
end

# ╔═╡ a0000001-0080-0000-0000-000000000001
md"""
## Summary

| Quantity | Value | Interpretation |
|----------|-------|----------------|
| $\alpha_L$ | from SC map | Leading coeff of local coordinate at $z=1$ |
| $\alpha_R$ | from SC map | Leading coeff of local coordinate at $z=-1$ |
| $d$ | extracted | Propagation distance on the strip |
| BPZ sign | $(-1)^N$ | From Ward recursion |
| Diagonality | off-diag fraction | Should be $\sim 10^{-15}$ (exact) |

The strip 2-point vertex, computed via the full recursion pipeline,
should exactly equal the BPZ form times a propagator. This validates:
1. The Neumann coefficient computation
2. The Ward identity recursion
3. The BPZ sign conventions
4. The relationship between $\alpha_i$ and propagation distance
"""

# ╔═╡ Cell order:
# ╟─a0000001-0010-0000-0000-000000000001
# ╠═a0000001-0001-0000-0000-000000000001
# ╠═a0000001-0002-0000-0000-000000000001
# ╠═a0000001-0003-0000-0000-000000000001
# ╠═a0000001-0004-0000-0000-000000000001
# ╠═a0000001-0011-0000-0000-000000000001
# ╠═a0000001-0012-0000-0000-000000000001
# ╟─a0000001-0020-0000-0000-000000000001
# ╟─a0000001-0021-0000-0000-000000000001
# ╠═a0000001-0022-0000-0000-000000000001
# ╟─a0000001-0023-0000-0000-000000000001
# ╠═a0000001-0024-0000-0000-000000000001
# ╠═a0000001-0025-0000-0000-000000000001
# ╠═a0000001-0026-0000-0000-000000000001
# ╟─a0000001-0030-0000-0000-000000000001
# ╠═a0000001-0031-0000-0000-000000000001
# ╠═a0000001-0032-0000-0000-000000000001
# ╠═a0000001-0033-0000-0000-000000000001
# ╠═a0000001-0034-0000-0000-000000000001
# ╟─a0000001-0035-0000-0000-000000000001
# ╠═a0000001-0036-0000-0000-000000000001
# ╟─a0000001-0040-0000-0000-000000000001
# ╠═a0000001-0041-0000-0000-000000000001
# ╠═a0000001-0042-0000-0000-000000000001
# ╟─a0000001-0050-0000-0000-000000000001
# ╠═a0000001-0051-0000-0000-000000000001
# ╠═a0000001-0052-0000-0000-000000000001
# ╠═a0000001-0053-0000-0000-000000000001
# ╠═a0000001-0054-0000-0000-000000000001
# ╠═a0000001-0055-0000-0000-000000000001
# ╠═a0000001-0056-0000-0000-000000000001
# ╟─a0000001-0057-0000-0000-000000000001
# ╠═a0000001-0058-0000-0000-000000000001
# ╟─a0000001-0060-0000-0000-000000000001
# ╠═a0000001-0061-0000-0000-000000000001
# ╟─a0000001-0062-0000-0000-000000000001
# ╠═a0000001-0063-0000-0000-000000000001
# ╟─a0000001-0064-0000-0000-000000000001
# ╠═a0000001-0065-0000-0000-000000000001
# ╟─a0000001-0070-0000-0000-000000000001
# ╠═a0000001-0071-0000-0000-000000000001
# ╟─a0000001-0080-0000-0000-000000000001
