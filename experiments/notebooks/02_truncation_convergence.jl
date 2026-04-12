### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ b0000001-0001-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ b0000001-0002-0000-0000-000000000001
using CFTTruncation

# ╔═╡ b0000001-0003-0000-0000-000000000001
using TensorKit: norm

# ╔═╡ b0000001-0004-0000-0000-000000000001
using Printf

# ╔═╡ b0000001-0005-0000-0000-000000000001
using Random

# ╔═╡ b0000001-0010-0000-0000-000000000001
md"""
# 02 — Truncation Convergence of the Modified Vertex

Diagnostic: for the modified vertex $\widetilde{V} = e^{(H_L+H_R)\ell/2} V_\ell$
computed at $h_{\max} = 6$, measure

$$r(h_{\text{trunc}}) = \frac{\|\Pi_{h \le h_{\text{trunc}}-1}\, \widetilde{V}\|}{\|\Pi_{h \le h_{\text{trunc}}}\, \widetilde{V}\|}$$

where $\Pi_{h \le H}$ projects **each tensor factor** to the subspace with
conformal dimension $\le H$. Close to 1 ⟹ the $h_{\text{trunc}}$-shell adds
little — the truncation at $h_{\text{trunc}}$ is nearly converged.

**$h_\psi$** = conformal weight of the fixed/inserted state (probe).
**$h_{\text{trunc}}$** = truncation level being tested.
"""

# ╔═╡ b0000001-0018-0000-0000-000000000001
import LinearAlgebra

# ╔═╡ b0000001-0011-0000-0000-000000000001
begin
    R_val = 1.0
    c_cft = 1.0
    H_MAX = 6.0
    h_truncs = [2.0, 3.0, 4.0, 5.0, 6.0]
    ells = vcat(collect(0.05:0.05:0.5), collect(0.6:0.1:1.5))
    Random.seed!(42)
end

# ╔═╡ b0000001-0012-0000-0000-000000000001
md"### Build CFT and cache modified vertices (TensorMaps)"

# ╔═╡ b0000001-0013-0000-0000-000000000001
cft6 = CompactBosonCFT(R=R_val, trunc=TruncationSpec(H_MAX))

# ╔═╡ b0000001-0014-0000-0000-000000000001
# Cache: Dict{Float64, TensorMap}. Computed once per ℓ.
cache = modified_vertex_cache(cft6, ells; c=c_cft)

# ╔═╡ b0000001-0015-0000-0000-000000000001
md"### Notebook-local helpers"

# ╔═╡ b0000001-0016-0000-0000-000000000001
"""Ratio = ‖Π_{h_trunc-1}‖ / ‖Π_{h_trunc}‖ for each h_trunc."""
function convergence_ratios(norm_fn, h_truncs)
    [let n_lo = norm_fn(ht - 1.0); n_hi = norm_fn(ht)
         n_hi > 0 ? n_lo / n_hi : NaN
     end for ht in h_truncs]
end

# ╔═╡ b0000001-0017-0000-0000-000000000001
"""Enumerate distinct conformal weights with random unit vectors."""
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
        nrm = LinearAlgebra.norm(c); nrm > 0 && (c ./= nrm)
        push!(result, (h_psi=h, states=states, coeffs=c))
    end
    result
end

# ╔═╡ b0000001-0019-0000-0000-000000000001
function format_table(title, ells, h_truncs, data_matrix)
    header = title * "  " * join([@sprintf("h_tr=%d", Int(h)) for h in h_truncs], "  ")
    lines = [header, repeat("-", length(header))]
    for (i, ℓ) in enumerate(ells)
        push!(lines, @sprintf("  %5.2f   ", ℓ) *
              join([@sprintf("%6.3f", data_matrix[i, j]) for j in eachindex(h_truncs)], "  "))
    end
    join(lines, "\n")
end

# ╔═╡ b0000001-0020-0000-0000-000000000001
md"## Experiment 1: Full 3-leg"

# ╔═╡ b0000001-0021-0000-0000-000000000001
exp1_table = let
    bp = cft6.basis_phys; bb = cft6.basis_bond
    bases = [bp, bb, bb]
    data = zeros(length(ells), length(h_truncs))
    for (i, ℓ) in enumerate(ells)
        Vm = cache[ℓ]
        rats = convergence_ratios(h_truncs) do hc
            norm(project_to_hcut(Vm, bases, hc))
        end
        data[i, :] .= rats
    end
    format_table("  ℓ    ", ells, h_truncs, data)
end

# ╔═╡ b0000001-0022-0000-0000-000000000001
Text(exp1_table)

# ╔═╡ b0000001-0030-0000-0000-000000000001
md"""
## Experiment 2: Fix $\psi_T$ at weight $h_\psi$, 2-leg convergence

One table per $h_\psi$. Rows = $\ell$, columns = $h_{\text{trunc}}$.
"""

# ╔═╡ b0000001-0031-0000-0000-000000000001
shells_phys = random_weight_shells(cft6.basis_phys; rng=Random.MersenneTwister(42))

# ╔═╡ b0000001-0032-0000-0000-000000000001
exp2_tables = let
    bp = cft6.basis_phys; bb = cft6.basis_bond
    tables = String[]
    for shell in shells_phys
        vec_T = collect(zip(shell.states, shell.coeffs))
        data = zeros(length(ells), length(h_truncs))
        for (i, ℓ) in enumerate(ells)
            Vm = cache[ℓ]
            rats = convergence_ratios(h_truncs) do hc
                projected_norm_after_contract_T(Vm, bp, bb, vec_T, hc)
            end
            data[i, :] .= rats
        end
        push!(tables, format_table(@sprintf("h_ψ=%4.1f  ℓ", shell.h_psi), ells, h_truncs, data))
    end
    tables
end

# ╔═╡ b0000001-0033-0000-0000-000000000001
Text(join(exp2_tables, "\n\n"))

# ╔═╡ b0000001-0040-0000-0000-000000000001
md"""
## Experiment 3: Fix $\psi_T$ and $\psi_L$, 1-leg convergence

$h_{\psi_T} = h_{\psi_L}$ (same weight, independent random vectors).
"""

# ╔═╡ b0000001-0041-0000-0000-000000000001
shells_bond = random_weight_shells(cft6.basis_bond; rng=Random.MersenneTwister(123))

# ╔═╡ b0000001-0042-0000-0000-000000000001
exp3_tables = let
    bp = cft6.basis_phys; bb = cft6.basis_bond
    h_phys_set = Set(s.h_psi for s in shells_phys)
    h_bond_set = Set(s.h_psi for s in shells_bond)
    h_common = sort(collect(h_phys_set ∩ h_bond_set))
    shell_phys_map = Dict(s.h_psi => s for s in shells_phys)
    shell_bond_map = Dict(s.h_psi => s for s in shells_bond)
    tables = String[]
    for h in h_common
        sT = shell_phys_map[h]; sL = shell_bond_map[h]
        vec_T = collect(zip(sT.states, sT.coeffs))
        vec_L = collect(zip(sL.states, sL.coeffs))
        data = zeros(length(ells), length(h_truncs))
        for (i, ℓ) in enumerate(ells)
            Vm = cache[ℓ]
            rats = convergence_ratios(h_truncs) do hc
                projected_norm_after_contract_TL(Vm, bp, bb, vec_T, vec_L, hc)
            end
            data[i, :] .= rats
        end
        push!(tables, format_table(@sprintf("h_ψ=%4.1f  ℓ", h), ells, h_truncs, data))
    end
    tables
end

# ╔═╡ b0000001-0043-0000-0000-000000000001
Text(join(exp3_tables, "\n\n"))

# ╔═╡ b0000001-0050-0000-0000-000000000001
md"""
## Summary

- $r(h_{\text{trunc}}) = \|\Pi_{h_{\text{trunc}}-1}\widetilde{V}\| / \|\Pi_{h_{\text{trunc}}}\widetilde{V}\|$
- Close to 1 ⟹ the $h_{\text{trunc}}$-shell adds little (truncation converged).
- Expect: $r \to 1$ as $h_{\text{trunc}}$ increases and as $\ell \to 0$.
"""

# ╔═╡ Cell order:
# ╟─b0000001-0010-0000-0000-000000000001
# ╠═b0000001-0001-0000-0000-000000000001
# ╠═b0000001-0002-0000-0000-000000000001
# ╠═b0000001-0003-0000-0000-000000000001
# ╠═b0000001-0004-0000-0000-000000000001
# ╠═b0000001-0005-0000-0000-000000000001
# ╠═b0000001-0018-0000-0000-000000000001
# ╠═b0000001-0011-0000-0000-000000000001
# ╟─b0000001-0012-0000-0000-000000000001
# ╠═b0000001-0013-0000-0000-000000000001
# ╠═b0000001-0014-0000-0000-000000000001
# ╟─b0000001-0015-0000-0000-000000000001
# ╠═b0000001-0016-0000-0000-000000000001
# ╠═b0000001-0017-0000-0000-000000000001
# ╠═b0000001-0019-0000-0000-000000000001
# ╟─b0000001-0020-0000-0000-000000000001
# ╠═b0000001-0021-0000-0000-000000000001
# ╠═b0000001-0022-0000-0000-000000000001
# ╟─b0000001-0030-0000-0000-000000000001
# ╠═b0000001-0031-0000-0000-000000000001
# ╠═b0000001-0032-0000-0000-000000000001
# ╠═b0000001-0033-0000-0000-000000000001
# ╟─b0000001-0040-0000-0000-000000000001
# ╠═b0000001-0041-0000-0000-000000000001
# ╠═b0000001-0042-0000-0000-000000000001
# ╠═b0000001-0043-0000-0000-000000000001
# ╟─b0000001-0050-0000-0000-000000000001
