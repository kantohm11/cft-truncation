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
using TensorKit: dim, domain, codomain

# ╔═╡ a0000001-0004-0000-0000-000000000001
using LinearAlgebra: norm

# ╔═╡ a0000001-0005-0000-0000-000000000001
using Printf

# ╔═╡ a0000001-0006-0000-0000-000000000001
using Random

# ╔═╡ a0000001-0010-0000-0000-000000000001
md"""
# 02 — Truncation Convergence of the Modified Vertex

Test whether the level truncation at $h_{\max}$ makes sense for the
**modified vertex** $\widetilde{V}_\ell = e^{(H_L + H_R)\ell/2} \cdot V_\ell$.

The diagnostic: the normalized overlap
$$r = \frac{\|\Pi_{h \le h_{\max}-1}\, \widetilde{V}\|}{\|\widetilde{V}\|}$$
measures how much of the modified vertex lives **below** the top shell.
Close to 1 = converged (top-level states contribute little).

Three experiments:
1. **Full 3-leg** — $\widetilde{V}$ as a vector in $V'_{\text{phys}} \otimes V'^{2}_{\text{bond}}$.
2. **Fix random $\psi_T$ at weight $h$** — contract with a random unit vector in the weight-$h$ subspace of $V_{\text{phys}}$; result in $V'^{2}_{\text{bond}}$.
3. **Fix random $\psi_T$ and $\psi_L$** — result in $V'_{\text{bond}}$.

Sweep: $h_{\max} \in \{2,3,4,5,6\}$, $\ell$ dense for $\ell \le 0.5$ (step 0.05) and coarser for $\ell > 0.5$ (step 0.1).
Fixed states: random unit vectors in each available conformal-weight shell.
"""

# ╔═╡ a0000001-0011-0000-0000-000000000001
begin
    R = 1.0
    c_cft = 1.0   # central charge
    h_maxs = [2.0, 3.0, 4.0, 5.0, 6.0]
    ells = vcat(collect(0.05:0.05:0.5), collect(0.6:0.1:1.5))
    Random.seed!(42)
end

# ╔═╡ a0000001-0012-0000-0000-000000000001
md"""
### Helper: enumerate conformal-weight shells and build random vectors
"""

# ╔═╡ a0000001-0013-0000-0000-000000000001
"""
Find all distinct conformal dimensions present in `basis`, and for each,
return a list of (n, α) pairs and a random unit-vector of coefficients.
"""
function random_weight_vectors(basis, rng=Random.GLOBAL_RNG)
    # Collect all (h, [(n,α)...]) groups
    h_to_states = Dict{Float64, Vector{Tuple{Int,Int}}}()
    for n in keys(basis.states)
        for α in eachindex(basis.states[n])
            h = conformal_dim(basis, n, α)
            hr = round(h; digits=6)  # avoid float noise in keys
            push!(get!(h_to_states, hr, []), (n, α))
        end
    end
    # For each h, random unit vector
    result = Dict{Float64, Tuple{Vector{Tuple{Int,Int}}, Vector{Float64}}}()
    for (h, states) in h_to_states
        coeffs = randn(rng, length(states))
        nrm = norm(coeffs)
        if nrm > 0
            coeffs ./= nrm
        end
        result[h] = (states, coeffs)
    end
    result
end

# ╔═╡ a0000001-0014-0000-0000-000000000001
"""
Contract the raw vertex with a linear combination on the T arm.
`vec_T` is a list of ((n_T, αT), coeff) pairs.
Returns a Dict keyed by (n_L, n_R, αL, αR).
"""
function contract_T_random(raw, vec_T)
    result = Dict{NTuple{4,Int}, Float64}()
    for ((n_T, αT), coeff) in vec_T
        coeff == 0.0 && continue
        for (key, val) in raw
            kn_T, n_L, n_R, kαT, αL, αR = key
            (kn_T == n_T && kαT == αT) || continue
            rkey = (n_L, n_R, αL, αR)
            result[rkey] = get(result, rkey, 0.0) + coeff * val
        end
    end
    result
end

# ╔═╡ a0000001-0015-0000-0000-000000000001
"""
Contract with random vectors on BOTH T and L arms.
Returns a Dict keyed by (n_R, αR).
"""
function contract_TL_random(raw, vec_T, vec_L)
    result = Dict{NTuple{2,Int}, Float64}()
    for ((n_T, αT), cT) in vec_T
        cT == 0.0 && continue
        for ((n_L, αL), cL) in vec_L
            cL == 0.0 && continue
            for (key, val) in raw
                kn_T, kn_L, n_R, kαT, kαL, αR = key
                (kn_T == n_T && kαT == αT && kn_L == n_L && kαL == αL) || continue
                rkey = (n_R, αR)
                result[rkey] = get(result, rkey, 0.0) + cT * cL * val
            end
        end
    end
    result
end

# ╔═╡ a0000001-0020-0000-0000-000000000001
md"""
## Experiment 1: Full 3-leg convergence

Ratio $r(\ell)$ for each $h_{\max}$.
"""

# ╔═╡ a0000001-0021-0000-0000-000000000001
exp1_results = let results = Dict()
    for h_max in h_maxs
        cft = CompactBosonCFT(R=R, trunc=TruncationSpec(h_max))
        h_cut = h_max - 1.0
        ratios = Float64[]
        for ℓ in ells
            vd = compute_vertex(cft, ℓ)
            mod_raw = modified_vertex_raw(vd; c=c_cft)
            r = convergence_ratio(mod_raw, vd; h_cut=h_cut)
            push!(ratios, r.ratio)
        end
        results[h_max] = ratios
    end
    results
end

# ╔═╡ a0000001-0023-0000-0000-000000000001
let
    header = "   ℓ    " * join([@sprintf("h=%d", Int(h)) for h in h_maxs], "    ")
    lines = [header, repeat("-", length(header))]
    for (i, ℓ) in enumerate(ells)
        vals = [exp1_results[h][i] for h in h_maxs]
        push!(lines, @sprintf(" %5.2f  ", ℓ) * join([@sprintf("%7.4f", v) for v in vals], "  "))
    end
    Text(join(lines, "\n"))
end

# ╔═╡ a0000001-0030-0000-0000-000000000001
md"""
## Experiment 2: Contract with random $\psi_T$ at weight $h$, 2-leg convergence

For each available conformal weight $h$ in $V_{\text{phys}}$, build a random unit
vector in that weight shell, contract with the modified vertex, and compute
the 2-leg convergence ratio.

Rows: $\ell$. Columns: conformal weight $h$ of the inserted state.
One table per $h_{\max}$.
"""

# ╔═╡ a0000001-0031-0000-0000-000000000001
exp2_results = let results = Dict()
    for h_max in h_maxs
        cft = CompactBosonCFT(R=R, trunc=TruncationSpec(h_max))
        h_cut = h_max - 1.0
        rng = Random.MersenneTwister(42)
        wvecs = random_weight_vectors(cft.basis_phys, rng)
        h_vals = sort(collect(keys(wvecs)))
        for ℓ in ells
            vd = compute_vertex(cft, ℓ)
            mod_raw = modified_vertex_raw(vd; c=c_cft)
            for h_fixed in h_vals
                (states, coeffs) = wvecs[h_fixed]
                vec_T = collect(zip(states, coeffs))
                c2 = contract_T_random(mod_raw, vec_T)
                r = convergence_ratio_2leg(c2, vd.basis_bond; h_cut=h_cut)
                push!(get!(results, (h_max, h_fixed), Float64[]), r.ratio)
            end
        end
    end
    results
end

# ╔═╡ a0000001-0032-0000-0000-000000000001
md"""
### Exp 2 tables

One table per $h_{\max}$. Columns are the conformal weight $h$ of the random $\psi_T$.
"""

# ╔═╡ a0000001-0033-0000-0000-000000000001
let
    tables = []
    for h_max in h_maxs
        # Find all h_fixed values for this h_max
        h_vals = sort([h for (hm, h) in keys(exp2_results) if hm == h_max])
        isempty(h_vals) && continue
        header = @sprintf("h_max=%d  ℓ    ", Int(h_max)) * join([@sprintf("h=%.1f", h) for h in h_vals], "  ")
        lines = [header, repeat("-", length(header))]
        for (i, ℓ) in enumerate(ells)
            vals = [exp2_results[(h_max, h)][i] for h in h_vals]
            push!(lines, @sprintf("         %5.2f  ", ℓ) * join([@sprintf("%6.3f", v) for v in vals], "  "))
        end
        push!(tables, join(lines, "\n"))
    end
    Text(join(tables, "\n\n"))
end

# ╔═╡ a0000001-0040-0000-0000-000000000001
md"""
## Experiment 3: Contract with random $\psi_T$ and $\psi_L$, 1-leg convergence

Fix random unit vectors on both T (weight $h_T$) and L (weight $h_L$) arms.
The remaining vector lives in $V'_{\text{bond}}$.

For brevity, use $h_T = h_L$ (same weight on both inserted arms).
"""

# ╔═╡ a0000001-0041-0000-0000-000000000001
exp3_results = let results = Dict()
    for h_max in h_maxs
        cft = CompactBosonCFT(R=R, trunc=TruncationSpec(h_max))
        h_cut = h_max - 1.0
        rng = Random.MersenneTwister(42)
        wvecs_phys = random_weight_vectors(cft.basis_phys, rng)
        wvecs_bond = random_weight_vectors(cft.basis_bond, Random.MersenneTwister(123))
        # Use h_T = h_L for simplicity; pick weights present in BOTH bases
        h_vals_phys = Set(keys(wvecs_phys))
        h_vals_bond = Set(keys(wvecs_bond))
        h_vals = sort(collect(h_vals_phys ∩ h_vals_bond))
        for ℓ in ells
            vd = compute_vertex(cft, ℓ)
            mod_raw = modified_vertex_raw(vd; c=c_cft)
            for h_fixed in h_vals
                (states_T, coeffs_T) = wvecs_phys[h_fixed]
                (states_L, coeffs_L) = wvecs_bond[h_fixed]
                vec_T = collect(zip(states_T, coeffs_T))
                vec_L = collect(zip(states_L, coeffs_L))
                c3 = contract_TL_random(mod_raw, vec_T, vec_L)
                r = convergence_ratio_1leg(c3, vd.basis_bond; h_cut=h_cut)
                push!(get!(results, (h_max, h_fixed), Float64[]), r.ratio)
            end
        end
    end
    results
end

# ╔═╡ a0000001-0042-0000-0000-000000000001
md"""
### Exp 3 tables

One table per $h_{\max}$. Columns are the shared weight $h_T = h_L$.
"""

# ╔═╡ a0000001-0043-0000-0000-000000000001
let
    tables = []
    for h_max in h_maxs
        h_vals = sort([h for (hm, h) in keys(exp3_results) if hm == h_max])
        isempty(h_vals) && continue
        header = @sprintf("h_max=%d  ℓ    ", Int(h_max)) * join([@sprintf("h=%.1f", h) for h in h_vals], "  ")
        lines = [header, repeat("-", length(header))]
        for (i, ℓ) in enumerate(ells)
            vals = [exp3_results[(h_max, h)][i] for h in h_vals]
            push!(lines, @sprintf("         %5.2f  ", ℓ) * join([@sprintf("%6.3f", v) for v in vals], "  "))
        end
        push!(tables, join(lines, "\n"))
    end
    Text(join(tables, "\n\n"))
end

# ╔═╡ a0000001-0050-0000-0000-000000000001
md"""
## Summary

- **Exp 1** (full vertex): ratio $r$ measures overall truncation convergence.
- **Exp 2** (fix $\psi_T$): how convergence depends on the weight of the inserted state. Heavier states should be easier to truncate (less room for bond descendants).
- **Exp 3** (fix $\psi_T, \psi_L$): same, with both T and L arms contracted.

Convergence ratio → 1 as $h_{\max} \to \infty$ at fixed $\ell$. For fixed $h_{\max}$, expect better convergence (higher $r$) at small $\ell$ (thin strip, less propagation to compensate).
"""

# ╔═╡ Cell order:
# ╟─a0000001-0010-0000-0000-000000000001
# ╠═a0000001-0001-0000-0000-000000000001
# ╠═a0000001-0002-0000-0000-000000000001
# ╠═a0000001-0003-0000-0000-000000000001
# ╠═a0000001-0004-0000-0000-000000000001
# ╠═a0000001-0005-0000-0000-000000000001
# ╠═a0000001-0006-0000-0000-000000000001
# ╠═a0000001-0011-0000-0000-000000000001
# ╟─a0000001-0012-0000-0000-000000000001
# ╠═a0000001-0013-0000-0000-000000000001
# ╠═a0000001-0014-0000-0000-000000000001
# ╠═a0000001-0015-0000-0000-000000000001
# ╟─a0000001-0020-0000-0000-000000000001
# ╠═a0000001-0021-0000-0000-000000000001
# ╠═a0000001-0023-0000-0000-000000000001
# ╟─a0000001-0030-0000-0000-000000000001
# ╠═a0000001-0031-0000-0000-000000000001
# ╟─a0000001-0032-0000-0000-000000000001
# ╠═a0000001-0033-0000-0000-000000000001
# ╟─a0000001-0040-0000-0000-000000000001
# ╠═a0000001-0041-0000-0000-000000000001
# ╟─a0000001-0042-0000-0000-000000000001
# ╠═a0000001-0043-0000-0000-000000000001
# ╟─a0000001-0050-0000-0000-000000000001
