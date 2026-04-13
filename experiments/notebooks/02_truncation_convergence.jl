### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ c0000001-0001-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ c0000001-0002-0000-0000-000000000001
using CFTTruncation

# ╔═╡ c0000001-0003-0000-0000-000000000001
using TensorKit: norm

# ╔═╡ c0000001-0004-0000-0000-000000000001
using Plots

# ╔═╡ c0000001-0005-0000-0000-000000000001
using Random

# ╔═╡ c0000001-0010-0000-0000-000000000001
md"""
# 02 — Truncation Convergence of the Modified Vertex

$$r(h_{\text{trunc}}) = \frac{\|\Pi_{h \le h_{\text{trunc}}-1}\, \widetilde{V}\|}{\|\Pi_{h \le h_{\text{trunc}}}\, \widetilde{V}\|}$$

Close to 1 ⟹ the $h_{\text{trunc}}$-shell adds little (truncation converged).
"""

# ╔═╡ c0000001-0006-0000-0000-000000000001
import LinearAlgebra

# ╔═╡ c0000001-0011-0000-0000-000000000001
begin
    R_val = 1.0
    H_MAX = 8.0
    h_truncs = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    ells = collect(0.025:0.025:0.5)  # dense grid, ℓ ≤ 0.5 only
    Random.seed!(42)
    # Enable disk caching so re-runs (and HTML export) are fast
    set_cache_dir(joinpath(@__DIR__, "..", "..", "experiments", "results", "cache"))
end

# ╔═╡ c0000001-0012-0000-0000-000000000001
cft6 = CompactBosonCFT(R=R_val, trunc=TruncationSpec(H_MAX))

# ╔═╡ c0000001-0013-0000-0000-000000000001
cache = modified_vertex_cache(cft6, ells)

# ╔═╡ c0000001-0014-0000-0000-000000000001
md"### Helpers"

# ╔═╡ c0000001-0015-0000-0000-000000000001
function convergence_ratios(norm_fn, h_truncs)
    [let n_lo = norm_fn(ht - 1.0); n_hi = norm_fn(ht)
         n_hi > 0 ? n_lo / n_hi : NaN
     end for ht in h_truncs]
end

# ╔═╡ c0000001-0016-0000-0000-000000000001
function random_weight_shells(basis; rng=Random.GLOBAL_RNG)
    h_map = Dict{Float64, Vector{Tuple{Int,Int}}}()
    for n in keys(basis.states), a in eachindex(basis.states[n])
        h = round(conformal_dim(basis, n, a); digits=6)
        push!(get!(h_map, h, []), (n, a))
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

# ╔═╡ c0000001-0020-0000-0000-000000000001
md"## Experiment 1: Full 3-leg"

# ╔═╡ c0000001-0021-0000-0000-000000000001
exp1_data = let
    bp = cft6.basis_phys; bb = cft6.basis_bond
    bases = [bp, bb, bb]
    data = zeros(length(ells), length(h_truncs))
    for (i, l) in enumerate(ells)
        Vm = cache[l]
        rats = convergence_ratios(h_truncs) do hc
            norm(project_to_hcut(Vm, bases, hc))
        end
        data[i, :] .= rats
    end
    data
end

# ╔═╡ c0000001-0022-0000-0000-000000000001
let
    ell_sel = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    p = plot(; xlabel="h_trunc", ylabel="r",
             title="Exp 1: Full 3-leg convergence ratio",
             legend=:bottomright, size=(700, 400))
    for l in ell_sel
        idx = findfirst(==(l), ells)
        idx === nothing && continue
        plot!(p, h_truncs, exp1_data[idx, :]; label="ℓ=$l",
              marker=:circle, markersize=4)
    end
    p
end

# ╔═╡ c0000001-0030-0000-0000-000000000001
md"## Experiment 2: Contract $\psi_T$ at weight $h_\psi$"

# ╔═╡ c0000001-0031-0000-0000-000000000001
shells_phys = random_weight_shells(cft6.basis_phys; rng=Random.MersenneTwister(42))

# ╔═╡ c0000001-0032-0000-0000-000000000001
exp2_data = let
    bp = cft6.basis_phys; bb = cft6.basis_bond
    result = Dict{Float64, Matrix{Float64}}()
    for shell in shells_phys
        vec_T = collect(zip(shell.states, shell.coeffs))
        data = zeros(length(ells), length(h_truncs))
        for (i, l) in enumerate(ells)
            Vm = cache[l]
            rats = convergence_ratios(h_truncs) do hc
                projected_norm_after_contract_T(Vm, bp, bb, vec_T, hc)
            end
            data[i, :] .= rats
        end
        result[shell.h_psi] = data
    end
    result
end

# ╔═╡ c0000001-0033-0000-0000-000000000001
let
    h_psis = sort(collect(keys(exp2_data)))
    ell_sel = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    ncols = min(3, length(h_psis))
    nrows = cld(length(h_psis), ncols)
    plts = []
    for (idx, hp) in enumerate(h_psis)
        p = plot(; xlabel="h_trunc", ylabel="r",
                 title="h_ψ=$(round(hp; digits=1))", ylims=(0, 1.05),
                 legend=false)
        for l in ell_sel
            i = findfirst(==(l), ells)
            i === nothing && continue
            plot!(p, h_truncs, exp2_data[hp][i, :]; marker=:circle, markersize=3, label="ℓ=$l")
        end
        push!(plts, p)
    end
    plot(plts...; layout=(nrows, ncols), size=(300*ncols, 250*nrows),
         plot_title="Exp 2: Contract ψ_T, 2-leg")
end

# ╔═╡ c0000001-0040-0000-0000-000000000001
md"## Experiment 3: Contract $\psi_T$ and $\psi_L$"

# ╔═╡ c0000001-0041-0000-0000-000000000001
shells_bond = random_weight_shells(cft6.basis_bond; rng=Random.MersenneTwister(123))

# ╔═╡ c0000001-0042-0000-0000-000000000001
exp3_data = let
    bp = cft6.basis_phys; bb = cft6.basis_bond
    h_phys_set = Set(s.h_psi for s in shells_phys)
    h_bond_set = Set(s.h_psi for s in shells_bond)
    h_common = sort(collect(h_phys_set ∩ h_bond_set))
    shell_phys_map = Dict(s.h_psi => s for s in shells_phys)
    shell_bond_map = Dict(s.h_psi => s for s in shells_bond)
    result = Dict{Float64, Matrix{Float64}}()
    for h in h_common
        sT = shell_phys_map[h]; sL = shell_bond_map[h]
        vec_T = collect(zip(sT.states, sT.coeffs))
        vec_L = collect(zip(sL.states, sL.coeffs))
        data = zeros(length(ells), length(h_truncs))
        for (i, l) in enumerate(ells)
            Vm = cache[l]
            rats = convergence_ratios(h_truncs) do hc
                projected_norm_after_contract_TL(Vm, bp, bb, vec_T, vec_L, hc)
            end
            data[i, :] .= rats
        end
        result[h] = data
    end
    result
end

# ╔═╡ c0000001-0043-0000-0000-000000000001
let
    h_psis = sort(collect(keys(exp3_data)))
    ell_sel = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    ncols = min(3, length(h_psis))
    nrows = cld(length(h_psis), ncols)
    plts = []
    for (idx, hp) in enumerate(h_psis)
        p = plot(; xlabel="h_trunc", ylabel="r",
                 title="h_ψ=$(round(hp; digits=1))", ylims=(0, 1.05),
                 legend=false)
        for l in ell_sel
            i = findfirst(==(l), ells)
            i === nothing && continue
            plot!(p, h_truncs, exp3_data[hp][i, :]; marker=:circle, markersize=3, label="ℓ=$l")
        end
        push!(plts, p)
    end
    plot(plts...; layout=(nrows, ncols), size=(300*ncols, 250*nrows),
         plot_title="Exp 3: Contract ψ_T+ψ_L, 1-leg")
end

# ╔═╡ c0000001-0050-0000-0000-000000000001
md"""
## Summary

- $r \to 1$ as $h_{\text{trunc}}$ increases (for fixed $\ell$) and as $\ell \to 0$.
- Heavier $\psi$ (larger $h_\psi$) leaves less room for bond descendants ⟹ better convergence.
"""

# ╔═╡ Cell order:
# ╟─c0000001-0010-0000-0000-000000000001
# ╠═c0000001-0001-0000-0000-000000000001
# ╠═c0000001-0002-0000-0000-000000000001
# ╠═c0000001-0003-0000-0000-000000000001
# ╠═c0000001-0004-0000-0000-000000000001
# ╠═c0000001-0005-0000-0000-000000000001
# ╠═c0000001-0006-0000-0000-000000000001
# ╠═c0000001-0011-0000-0000-000000000001
# ╠═c0000001-0012-0000-0000-000000000001
# ╠═c0000001-0013-0000-0000-000000000001
# ╟─c0000001-0014-0000-0000-000000000001
# ╠═c0000001-0015-0000-0000-000000000001
# ╠═c0000001-0016-0000-0000-000000000001
# ╟─c0000001-0020-0000-0000-000000000001
# ╠═c0000001-0021-0000-0000-000000000001
# ╠═c0000001-0022-0000-0000-000000000001
# ╟─c0000001-0030-0000-0000-000000000001
# ╠═c0000001-0031-0000-0000-000000000001
# ╠═c0000001-0032-0000-0000-000000000001
# ╠═c0000001-0033-0000-0000-000000000001
# ╟─c0000001-0040-0000-0000-000000000001
# ╠═c0000001-0041-0000-0000-000000000001
# ╠═c0000001-0042-0000-0000-000000000001
# ╠═c0000001-0043-0000-0000-000000000001
# ╟─c0000001-0050-0000-0000-000000000001
