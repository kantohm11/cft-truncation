### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ a0000002-0001-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ a0000002-0002-0000-0000-000000000001
using CFTTruncation

# ╔═╡ a0000002-0003-0000-0000-000000000001
using LinearAlgebra: norm

# ╔═╡ a0000002-0004-0000-0000-000000000001
using Printf

# ╔═╡ a0000002-0005-0000-0000-000000000001
using Random

# ╔═╡ a0000002-0010-0000-0000-000000000001
md"""
# 02 — Truncation Convergence of the Modified Vertex

Diagnostic: for the modified vertex $\widetilde{V} = e^{(H_L+H_R)\ell/2} V_\ell$
computed at $h_{\max} = 6$, measure

$$r(h_{\text{trunc}}) = \frac{\|\Pi_{h \le h_{\text{trunc}}-1}\, \widetilde{V}\|}{\|\Pi_{h \le h_{\text{trunc}}}\, \widetilde{V}\|}$$

where $\Pi_{h \le H}$ projects **each tensor factor** to the subspace with
conformal dimension $\le H$.  Close to 1 means the $h_{\text{trunc}}$-shell
adds little — the truncation at $h_{\text{trunc}}$ is nearly converged.

**Key efficiency**: the vertex is computed once at $h_{\max} = 6$ per $\ell$.
Different $h_{\text{trunc}}$ values (2, 3, 4, 5, 6) are just different filters
on the same raw data.

### Naming convention
- **$h_\psi$**: conformal weight of the fixed/inserted state (the "probe").
- **$h_{\text{trunc}}$**: the truncation level being tested.

Three experiments:
1. **Full 3-leg**: no contraction, the entire $\widetilde{V}$.
2. **Fix $\psi_T$ at weight $h_\psi$**: contract T arm with a random unit vector in the weight-$h_\psi$ shell.
3. **Fix $\psi_T$ and $\psi_L$**: contract both T and L arms.

For each $h_\psi$, one table with rows = $\ell$, columns = $h_{\text{trunc}} \in \{2,3,4,5,6\}$.
"""

# ╔═╡ a0000002-0011-0000-0000-000000000001
begin
    R = 1.0
    c_cft = 1.0
    H_MAX = 6.0
    h_truncs = [2.0, 3.0, 4.0, 5.0, 6.0]
    ells = vcat(collect(0.05:0.05:0.5), collect(0.6:0.1:1.5))
    Random.seed!(42)
end

# ╔═╡ a0000002-0012-0000-0000-000000000001
md"### Build CFT and precompute vertices at $h_{\max}=6$"

# ╔═╡ a0000002-0013-0000-0000-000000000001
cft6 = CompactBosonCFT(R=R, trunc=TruncationSpec(H_MAX))

# ╔═╡ a0000002-0014-0000-0000-000000000001
# Precompute modified vertex for each ℓ (the expensive part — once per ℓ)
mod_raws = let d = Dict{Float64, Dict{NTuple{6,Int}, Float64}}()
    for ℓ in ells
        vd = compute_vertex(cft6, ℓ)
        d[ℓ] = modified_vertex_raw(vd; c=c_cft)
    end
    d
end

# ╔═╡ a0000002-0015-0000-0000-000000000001
# A reference VertexData for basis access (all share the same basis at h_max=6)
vd_ref = compute_vertex(cft6, ells[1])

# ╔═╡ a0000002-0016-0000-0000-000000000001
md"### Helpers"

# ╔═╡ a0000002-0017-0000-0000-000000000001
"""
Compute ‖Π_{h≤h_cut} v‖ for a 3-leg raw dict.
"""
function projected_norm_3leg(raw, basis_bond, basis_phys, h_cut)
    sq = 0.0
    for ((n_T, n_L, n_R, αT, αL, αR), val) in raw
        conformal_dim(basis_bond, n_L, αL) ≤ h_cut + 1e-10 &&
        conformal_dim(basis_bond, n_R, αR) ≤ h_cut + 1e-10 &&
        conformal_dim(basis_phys, n_T, αT) ≤ h_cut + 1e-10 || continue
        sq += val^2
    end
    sqrt(sq)
end

# ╔═╡ a0000002-0018-0000-0000-000000000001
function projected_norm_2leg(contracted, basis_bond, h_cut)
    sq = 0.0
    for ((n_L, n_R, αL, αR), val) in contracted
        conformal_dim(basis_bond, n_L, αL) ≤ h_cut + 1e-10 &&
        conformal_dim(basis_bond, n_R, αR) ≤ h_cut + 1e-10 || continue
        sq += val^2
    end
    sqrt(sq)
end

# ╔═╡ a0000002-0019-0000-0000-000000000001
function projected_norm_1leg(contracted, basis_bond, h_cut)
    sq = 0.0
    for ((n_R, αR), val) in contracted
        conformal_dim(basis_bond, n_R, αR) ≤ h_cut + 1e-10 || continue
        sq += val^2
    end
    sqrt(sq)
end

# ╔═╡ a0000002-001a-0000-0000-000000000001
"""
For each h_trunc, ratio = ‖Π_{h_trunc-1}‖ / ‖Π_{h_trunc}‖.
`norm_fn(h_cut)` should return the projected norm at that cutoff.
"""
function convergence_ratios(norm_fn, h_truncs)
    [let n_lo = norm_fn(ht - 1.0); n_hi = norm_fn(ht)
         n_hi > 0 ? n_lo / n_hi : NaN
     end
     for ht in h_truncs]
end

# ╔═╡ a0000002-001b-0000-0000-000000000001
"""
Enumerate all distinct conformal weights in `basis`.
For each, return (weight, states, random_unit_coeffs).
"""
function random_weight_shells(basis; rng=Random.GLOBAL_RNG)
    h_map = Dict{Float64, Vector{Tuple{Int,Int}}}()
    for n in keys(basis.states), α in eachindex(basis.states[n])
        h = round(conformal_dim(basis, n, α); digits=6)
        push!(get!(h_map, h, []), (n, α))
    end
    result = []
    for h in sort(collect(keys(h_map)))
        states = h_map[h]
        c = randn(rng, length(states))
        nrm = norm(c); nrm > 0 && (c ./= nrm)
        push!(result, (h_psi=h, states=states, coeffs=c))
    end
    result
end

# ╔═╡ a0000002-001c-0000-0000-000000000001
function contract_T_vec(raw, vec_T)
    result = Dict{NTuple{4,Int}, Float64}()
    for (st, coeff) in zip(vec_T.states, vec_T.coeffs)
        coeff == 0.0 && continue
        n_T, αT = st
        for (key, val) in raw
            (key[1] == n_T && key[4] == αT) || continue
            rk = (key[2], key[3], key[5], key[6])
            result[rk] = get(result, rk, 0.0) + coeff * val
        end
    end
    result
end

# ╔═╡ a0000002-001d-0000-0000-000000000001
function contract_TL_vec(raw, vec_T, vec_L)
    result = Dict{NTuple{2,Int}, Float64}()
    for (sT, cT) in zip(vec_T.states, vec_T.coeffs)
        cT == 0.0 && continue
        for (sL, cL) in zip(vec_L.states, vec_L.coeffs)
            cL == 0.0 && continue
            n_T, αT = sT; n_L, αL = sL
            for (key, val) in raw
                (key[1]==n_T && key[4]==αT && key[2]==n_L && key[5]==αL) || continue
                rk = (key[3], key[6])
                result[rk] = get(result, rk, 0.0) + cT * cL * val
            end
        end
    end
    result
end

# ╔═╡ a0000002-001e-0000-0000-000000000001
function format_table(title, ells, h_truncs, data_matrix)
    header = title * "  " * join([@sprintf("h_tr=%d", Int(h)) for h in h_truncs], "  ")
    lines = [header, repeat("-", length(header))]
    for (i, ℓ) in enumerate(ells)
        push!(lines, @sprintf("  %5.2f   ", ℓ) *
              join([@sprintf("%6.3f", data_matrix[i, j]) for j in eachindex(h_truncs)], "  "))
    end
    join(lines, "\n")
end

# ╔═╡ a0000002-0020-0000-0000-000000000001
md"## Experiment 1: Full 3-leg"

# ╔═╡ a0000002-0021-0000-0000-000000000001
exp1_table = let
    bb = cft6.basis_bond; bp = cft6.basis_phys
    data = zeros(length(ells), length(h_truncs))
    for (i, ℓ) in enumerate(ells)
        raw = mod_raws[ℓ]
        rats = convergence_ratios(h_truncs) do hc
            projected_norm_3leg(raw, bb, bp, hc)
        end
        data[i, :] .= rats
    end
    format_table("  ℓ    ", ells, h_truncs, data)
end

# ╔═╡ a0000002-0022-0000-0000-000000000001
Text(exp1_table)

# ╔═╡ a0000002-0030-0000-0000-000000000001
md"""
## Experiment 2: Fix $\psi_T$ at weight $h_\psi$, 2-leg convergence

One table per $h_\psi$. Rows = $\ell$, columns = $h_{\text{trunc}}$.
"""

# ╔═╡ a0000002-0031-0000-0000-000000000001
shells_phys = random_weight_shells(cft6.basis_phys; rng=Random.MersenneTwister(42))

# ╔═╡ a0000002-0032-0000-0000-000000000001
exp2_tables = let
    bb = cft6.basis_bond
    tables = String[]
    for shell in shells_phys
        data = zeros(length(ells), length(h_truncs))
        for (i, ℓ) in enumerate(ells)
            c2 = contract_T_vec(mod_raws[ℓ], shell)
            rats = convergence_ratios(h_truncs) do hc
                projected_norm_2leg(c2, bb, hc)
            end
            data[i, :] .= rats
        end
        push!(tables, format_table(@sprintf("h_ψ=%4.1f  ℓ", shell.h_psi), ells, h_truncs, data))
    end
    tables
end

# ╔═╡ a0000002-0033-0000-0000-000000000001
Text(join(exp2_tables, "\n\n"))

# ╔═╡ a0000002-0040-0000-0000-000000000001
md"""
## Experiment 3: Fix $\psi_T$ and $\psi_L$, 1-leg convergence

Use $h_{\psi_T} = h_{\psi_L}$ (same weight, independent random vectors).
One table per shared $h_\psi$.
"""

# ╔═╡ a0000002-0041-0000-0000-000000000001
shells_bond = random_weight_shells(cft6.basis_bond; rng=Random.MersenneTwister(123))

# ╔═╡ a0000002-0042-0000-0000-000000000001
exp3_tables = let
    bb = cft6.basis_bond
    # Use weights present in both bases
    h_phys_set = Set(s.h_psi for s in shells_phys)
    h_bond_set = Set(s.h_psi for s in shells_bond)
    h_common = sort(collect(h_phys_set ∩ h_bond_set))
    shell_phys_map = Dict(s.h_psi => s for s in shells_phys)
    shell_bond_map = Dict(s.h_psi => s for s in shells_bond)
    tables = String[]
    for h in h_common
        sT = shell_phys_map[h]; sL = shell_bond_map[h]
        data = zeros(length(ells), length(h_truncs))
        for (i, ℓ) in enumerate(ells)
            c3 = contract_TL_vec(mod_raws[ℓ], sT, sL)
            rats = convergence_ratios(h_truncs) do hc
                projected_norm_1leg(c3, bb, hc)
            end
            data[i, :] .= rats
        end
        push!(tables, format_table(@sprintf("h_ψ=%4.1f  ℓ", h), ells, h_truncs, data))
    end
    tables
end

# ╔═╡ a0000002-0043-0000-0000-000000000001
Text(join(exp3_tables, "\n\n"))

# ╔═╡ a0000002-0050-0000-0000-000000000001
md"""
## Summary

- $r(h_{\text{trunc}}) = \|\Pi_{h_{\text{trunc}}-1}\widetilde{V}\| / \|\Pi_{h_{\text{trunc}}}\widetilde{V}\|$
- Close to 1 ⟹ the $h_{\text{trunc}}$-shell adds little (truncation converged at that level).
- Expect: $r \to 1$ as $h_{\text{trunc}}$ increases (for fixed $\ell$), and as $\ell \to 0$ (for fixed $h_{\text{trunc}}$).
- Heavier $\psi$ (larger $h_\psi$) leaves less room for bond-arm descendants, so convergence should improve.
"""

# ╔═╡ Cell order:
# ╟─a0000002-0010-0000-0000-000000000001
# ╠═a0000002-0001-0000-0000-000000000001
# ╠═a0000002-0002-0000-0000-000000000001
# ╠═a0000002-0003-0000-0000-000000000001
# ╠═a0000002-0004-0000-0000-000000000001
# ╠═a0000002-0005-0000-0000-000000000001
# ╠═a0000002-0011-0000-0000-000000000001
# ╟─a0000002-0012-0000-0000-000000000001
# ╠═a0000002-0013-0000-0000-000000000001
# ╠═a0000002-0014-0000-0000-000000000001
# ╠═a0000002-0015-0000-0000-000000000001
# ╟─a0000002-0016-0000-0000-000000000001
# ╠═a0000002-0017-0000-0000-000000000001
# ╠═a0000002-0018-0000-0000-000000000001
# ╠═a0000002-0019-0000-0000-000000000001
# ╠═a0000002-001a-0000-0000-000000000001
# ╠═a0000002-001b-0000-0000-000000000001
# ╠═a0000002-001c-0000-0000-000000000001
# ╠═a0000002-001d-0000-0000-000000000001
# ╠═a0000002-001e-0000-0000-000000000001
# ╟─a0000002-0020-0000-0000-000000000001
# ╠═a0000002-0021-0000-0000-000000000001
# ╠═a0000002-0022-0000-0000-000000000001
# ╟─a0000002-0030-0000-0000-000000000001
# ╠═a0000002-0031-0000-0000-000000000001
# ╠═a0000002-0032-0000-0000-000000000001
# ╠═a0000002-0033-0000-0000-000000000001
# ╟─a0000002-0040-0000-0000-000000000001
# ╠═a0000002-0041-0000-0000-000000000001
# ╠═a0000002-0042-0000-0000-000000000001
# ╠═a0000002-0043-0000-0000-000000000001
# ╟─a0000002-0050-0000-0000-000000000001
