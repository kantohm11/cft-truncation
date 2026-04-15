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
    R_val = 1.3
    H_MAX = 8.0
    h_truncs = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    ells = collect(0.1:0.05:0.6)  # dense grid, small ℓ regime
    Random.seed!(42)
    # Enable disk caching so re-runs (and HTML export) are fast
    set_cache_dir(joinpath(@__DIR__, "..", "..", "experiments", "results", "cache"))
end

# ╔═╡ c0000001-0012-0000-0000-000000000001
cft6 = CompactBosonCFT(R=R_val, trunc=TruncationSpec(H_MAX))

# ╔═╡ c0000001-0013-0000-0000-000000000001
begin
    cache_mod = modified_vertex_cache(cft6, ells)
    cache_raw = Dict(Float64(l) => compute_vertex(cft6, l).vertex for l in ells)
end

# ╔═╡ c0000001-0014-0000-0000-000000000001
md"### Helpers"

# ╔═╡ c0000001-0015-0000-0000-000000000001
function convergence_ratios(norm_fn, h_truncs)
    [let n_lo = norm_fn(ht - 1.0); n_hi = norm_fn(ht)
         n_hi > 0 ? n_lo / n_hi : NaN
     end for ht in h_truncs]
end

# ╔═╡ c0000001-0016-0000-0000-000000000001
function random_state_up_to(basis, h_cut; rng=Random.GLOBAL_RNG)
    states = Tuple{Int,Int}[]
    for n in keys(basis.states), a in eachindex(basis.states[n])
        conformal_dim(basis, n, a) ≤ h_cut + 1e-10 && push!(states, (n, a))
    end
    c = randn(rng, length(states))
    nrm = LinearAlgebra.norm(c); nrm > 0 && (c ./= nrm)
    collect(zip(states, c))
end

# ╔═╡ c0000001-0020-0000-0000-000000000001
md"## Experiment 1: Full 3-leg"

# ╔═╡ c0000001-0021-0000-0000-000000000001
exp1_data = let
    bp = cft6.basis_phys; bb = cft6.basis_bond
    bases = [bp, bb, bb]
    data_mod = zeros(length(ells), length(h_truncs))
    data_raw = zeros(length(ells), length(h_truncs))
    for (i, l) in enumerate(ells)
        Vm = cache_mod[l]; Vr = cache_raw[l]
        data_mod[i, :] .= convergence_ratios(h_truncs) do hc
            norm(project_to_hcut(Vm, bases, hc))
        end
        data_raw[i, :] .= convergence_ratios(h_truncs) do hc
            norm(project_to_hcut(Vr, bases, hc))
        end
    end
    (mod=data_mod, raw=data_raw)
end

# ╔═╡ c0000001-0022-0000-0000-000000000001
let
    ell_sel = ells
    p1 = plot(; xlabel="h_trunc", ylabel="r",
              title="Modified vertex", legend=:bottomright, ylims=(0, 1.05))
    p2 = plot(; xlabel="h_trunc", ylabel="r",
              title="Raw vertex", legend=:bottomright, ylims=(0, 1.05))
    for l in ell_sel
        idx = findfirst(==(l), ells)
        idx === nothing && continue
        plot!(p1, h_truncs, exp1_data.mod[idx, :]; label="ℓ=$(round(l;digits=3))",
              marker=:circle, markersize=3)
        plot!(p2, h_truncs, exp1_data.raw[idx, :]; label="ℓ=$(round(l;digits=3))",
              marker=:circle, markersize=3)
    end
    plot(p1, p2; layout=(1, 2), size=(1000, 400),
         plot_title="Exp 1: Full 3-leg convergence ratio")
end

# ╔═╡ c0000001-0030-0000-0000-000000000001
md"## Experiment 2: Contract random $\psi_T(h_{\text{cut}})$, 2-leg"

# ╔═╡ c0000001-0031-0000-0000-000000000001
h_cuts = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

# ╔═╡ c0000001-0032-0000-0000-000000000001
exp2_data = let
    bp = cft6.basis_phys; bb = cft6.basis_bond
    res_mod = Dict{Float64, Matrix{Float64}}()
    res_raw = Dict{Float64, Matrix{Float64}}()
    for hc_psi in h_cuts
        vT = random_state_up_to(bp, hc_psi; rng=Random.MersenneTwister(42))
        dm = zeros(length(ells), length(h_truncs))
        dr = zeros(length(ells), length(h_truncs))
        for (i, l) in enumerate(ells)
            dm[i, :] .= convergence_ratios(h_truncs) do hc
                projected_norm_after_contract_T(cache_mod[l], bp, bb, vT, hc)
            end
            dr[i, :] .= convergence_ratios(h_truncs) do hc
                projected_norm_after_contract_T(cache_raw[l], bp, bb, vT, hc)
            end
        end
        res_mod[hc_psi] = dm; res_raw[hc_psi] = dr
    end
    (mod=res_mod, raw=res_raw)
end

# ╔═╡ c0000001-0033-0000-0000-000000000001
let
    ell_sel = ells
    ncols = min(4, length(h_cuts)); nrows = cld(length(h_cuts), ncols)
    plts_mod = []; plts_raw = []
    for hc in h_cuts
        pm = plot(; xlabel="h_trunc", ylabel="r", title="mod h_cut=$hc", ylims=(0,1.05), legend=false)
        pr = plot(; xlabel="h_trunc", ylabel="r", title="raw h_cut=$hc", ylims=(0,1.05), legend=false)
        for l in ell_sel
            i = findfirst(==(l), ells); i === nothing && continue
            plot!(pm, h_truncs, exp2_data.mod[hc][i,:]; marker=:circle, markersize=3)
            plot!(pr, h_truncs, exp2_data.raw[hc][i,:]; marker=:circle, markersize=3)
        end
        push!(plts_mod, pm); push!(plts_raw, pr)
    end
    p1 = plot(plts_mod...; layout=(nrows,ncols), size=(250*ncols, 220*nrows),
              plot_title="Exp 2 modified: ψ_T(h≤h_cut)")
    p2 = plot(plts_raw...; layout=(nrows,ncols), size=(250*ncols, 220*nrows),
              plot_title="Exp 2 raw: ψ_T(h≤h_cut)")
    plot(p1, p2; layout=(2,1), size=(250*ncols, 440*nrows))
end

# ╔═╡ c0000001-0040-0000-0000-000000000001
md"## Experiment 3: Contract $\psi_T$ and $\psi_L$"

# ╔═╡ c0000001-0041-0000-0000-000000000001
md"Contract random $\psi_T(h \le h_{\text{cut}})$ and $\psi_L(h \le h_{\text{cut}})$"

# ╔═╡ c0000001-0042-0000-0000-000000000001
exp3_data = let
    bp = cft6.basis_phys; bb = cft6.basis_bond
    res_mod = Dict{Float64, Matrix{Float64}}()
    res_raw = Dict{Float64, Matrix{Float64}}()
    for hc_psi in h_cuts
        vT = random_state_up_to(bp, hc_psi; rng=Random.MersenneTwister(42))
        vL = random_state_up_to(bb, hc_psi; rng=Random.MersenneTwister(123))
        dm = zeros(length(ells), length(h_truncs))
        dr = zeros(length(ells), length(h_truncs))
        for (i, l) in enumerate(ells)
            dm[i, :] .= convergence_ratios(h_truncs) do hc
                projected_norm_after_contract_TL(cache_mod[l], bp, bb, vT, vL, hc)
            end
            dr[i, :] .= convergence_ratios(h_truncs) do hc
                projected_norm_after_contract_TL(cache_raw[l], bp, bb, vT, vL, hc)
            end
        end
        res_mod[hc_psi] = dm; res_raw[hc_psi] = dr
    end
    (mod=res_mod, raw=res_raw)
end

# ╔═╡ c0000001-0043-0000-0000-000000000001
let
    ell_sel = ells
    ncols = min(4, length(h_cuts)); nrows = cld(length(h_cuts), ncols)
    plts_mod = []; plts_raw = []
    for hc in h_cuts
        pm = plot(; xlabel="h_trunc", ylabel="r", title="mod h_cut=$hc", ylims=(0,1.05), legend=false)
        pr = plot(; xlabel="h_trunc", ylabel="r", title="raw h_cut=$hc", ylims=(0,1.05), legend=false)
        for l in ell_sel
            i = findfirst(==(l), ells); i === nothing && continue
            plot!(pm, h_truncs, exp3_data.mod[hc][i,:]; marker=:circle, markersize=3)
            plot!(pr, h_truncs, exp3_data.raw[hc][i,:]; marker=:circle, markersize=3)
        end
        push!(plts_mod, pm); push!(plts_raw, pr)
    end
    p1 = plot(plts_mod...; layout=(nrows,ncols), size=(250*ncols, 220*nrows),
              plot_title="Exp 3 modified: ψ_T+ψ_L(h≤h_cut)")
    p2 = plot(plts_raw...; layout=(nrows,ncols), size=(250*ncols, 220*nrows),
              plot_title="Exp 3 raw: ψ_T+ψ_L(h≤h_cut)")
    plot(p1, p2; layout=(2,1), size=(250*ncols, 440*nrows))
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
