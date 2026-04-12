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

# ╔═╡ a0000001-0024-0000-0000-000000000001
using Printf

# ╔═╡ a0000001-0010-0000-0000-000000000001
md"""
# 02 — Truncation Convergence of the Modified Vertex

Test whether the level truncation at $h_{\max}$ makes sense for the
**modified vertex** $\widetilde{V}_\ell = e^{(H_L + H_R)\ell/2} \cdot V_\ell$.

The diagnostic: the normalized overlap
$$r = \frac{\|\Pi_{h \le h_{\max}-1}\, \widetilde{V}\|}{\|\widetilde{V}\|}$$
measures how much of the modified vertex lives **below** the top shell.
Close to 1 means the truncation is converged (the top-level states contribute
little).

Three experiments with increasing contraction:
1. **Full 3-leg** — $\widetilde{V}$ as a vector in $V'_{\text{phys}} \otimes V'^2_{\text{bond}}$.
2. **Fix $\psi_T$** — contract with a low-$h$ primary on the T arm; result in $V'^2_{\text{bond}}$.
3. **Fix $\psi_T$ and $\psi_L$** — result in $V'_{\text{bond}}$.

Parameters: compact boson at $R = 1$, $c = 1$, $h_{\max} \in \{2, 3, 4, 5\}$,
$\ell \in (0, 3/2]$.  Primary test states: $|0; n\rangle$ for $n \in \{0, \pm 1\}$.
"""

# ╔═╡ a0000001-0011-0000-0000-000000000001
begin
    R = 1.0
    c = 1.0
    h_maxs = [2.0, 3.0, 4.0, 5.0]
    ells = collect(0.1:0.1:1.5)
    # Primary test states: (sector n, basis index 1 = primary)
    test_states_T = [(0, 1), (1, 1), (-1, 1)]
    test_states_L = [(0, 1), (1, 1), (-1, 1)]
end

# ╔═╡ a0000001-0020-0000-0000-000000000001
md"""
## Experiment 1: Full 3-leg convergence

Compute $r(\ell)$ for each $h_{\max}$. Each point requires building the vertex
at $h_{\max}$ and projecting to $h_{\max} - 1$.
"""

# ╔═╡ a0000001-0021-0000-0000-000000000001
exp1_results = let results = Dict()
    for h_max in h_maxs
        cft = CompactBosonCFT(R=R, trunc=TruncationSpec(h_max))
        h_cut = h_max - 1.0
        ratios = Float64[]
        for ℓ in ells
            vd = compute_vertex(cft, ℓ)
            mod_raw = modified_vertex_raw(vd; c=c)
            r = convergence_ratio(mod_raw, vd; h_cut=h_cut)
            push!(ratios, r.ratio)
        end
        results[h_max] = ratios
    end
    results
end

# ╔═╡ a0000001-0022-0000-0000-000000000001
md"""
### Exp 1 results (ratio $r$ vs $\ell$)

Each row is an $\ell$ value; columns are $h_{\max} \in \{2, 3, 4, 5\}$.
"""

# ╔═╡ a0000001-0023-0000-0000-000000000001
let
    header = "  ℓ   " * join(["h=$(Int(h))" for h in h_maxs], "    ")
    lines = [header, repeat("-", length(header))]
    for (i, ℓ) in enumerate(ells)
        vals = [round(exp1_results[h][i]; digits=4) for h in h_maxs]
        push!(lines, " $(round(ℓ, digits=1))  " * join([@sprintf("%7.4f", v) for v in vals], "  "))
    end
    Text(join(lines, "\n"))
end

# ╔═╡ a0000001-0030-0000-0000-000000000001
md"""
## Experiment 2: Contract with $\psi_T$ (primary), 2-leg convergence

For each primary test state $\psi_T = |0; n_T\rangle$ ($n_T \in \{0, \pm1\}$),
contract the modified vertex and compute the 2-leg convergence ratio.
"""

# ╔═╡ a0000001-0031-0000-0000-000000000001
exp2_results = let results = Dict()
    for (n_T, αT) in test_states_T
        for h_max in h_maxs
            cft = CompactBosonCFT(R=R, trunc=TruncationSpec(h_max))
            haskey(cft.basis_phys.states, n_T) || continue
            αT > length(cft.basis_phys.states[n_T]) && continue
            h_cut = h_max - 1.0
            ratios = Float64[]
            for ℓ in ells
                vd = compute_vertex(cft, ℓ)
                mod_raw = modified_vertex_raw(vd; c=c)
                c2 = contract_T(mod_raw, vd, n_T, αT)
                r = convergence_ratio_2leg(c2, vd.basis_bond; h_cut=h_cut)
                push!(ratios, r.ratio)
            end
            results[(n_T, h_max)] = ratios
        end
    end
    results
end

# ╔═╡ a0000001-0032-0000-0000-000000000001
md"""
### Exp 2 results: $\psi_T = |0; 0\rangle$ (identity primary)
"""

# ╔═╡ a0000001-0033-0000-0000-000000000001
let
    header = "  ℓ   " * join(["h=$(Int(h))" for h in h_maxs], "    ")
    lines = [header, repeat("-", length(header))]
    for (i, ℓ) in enumerate(ells)
        vals = Float64[]
        for h in h_maxs
            key = (0, h)
            if haskey(exp2_results, key) && i ≤ length(exp2_results[key])
                push!(vals, round(exp2_results[key][i]; digits=4))
            else
                push!(vals, NaN)
            end
        end
        push!(lines, " $(round(ℓ, digits=1))  " * join([@sprintf("%7.4f", v) for v in vals], "  "))
    end
    Text(join(lines, "\n"))
end

# ╔═╡ a0000001-0034-0000-0000-000000000001
md"""
### Exp 2 results: $\psi_T = |0; 1\rangle$ (n=1 primary, h=1/2)
"""

# ╔═╡ a0000001-0035-0000-0000-000000000001
let
    header = "  ℓ   " * join(["h=$(Int(h))" for h in h_maxs], "    ")
    lines = [header, repeat("-", length(header))]
    for (i, ℓ) in enumerate(ells)
        vals = Float64[]
        for h in h_maxs
            key = (1, h)
            if haskey(exp2_results, key) && i ≤ length(exp2_results[key])
                push!(vals, round(exp2_results[key][i]; digits=4))
            else
                push!(vals, NaN)
            end
        end
        push!(lines, " $(round(ℓ, digits=1))  " * join([@sprintf("%7.4f", v) for v in vals], "  "))
    end
    Text(join(lines, "\n"))
end

# ╔═╡ a0000001-0040-0000-0000-000000000001
md"""
## Experiment 3: Contract with $\psi_T$ and $\psi_L$, 1-leg convergence

Contract with primaries on both T and L arms. The remaining object
is a vector in $V'_{\text{bond}}$.
"""

# ╔═╡ a0000001-0041-0000-0000-000000000001
exp3_results = let results = Dict()
    for (n_T, αT) in test_states_T
        for (n_L, αL) in test_states_L
            for h_max in h_maxs
                cft = CompactBosonCFT(R=R, trunc=TruncationSpec(h_max))
                haskey(cft.basis_phys.states, n_T) || continue
                haskey(cft.basis_bond.states, n_L) || continue
                αT > length(cft.basis_phys.states[n_T]) && continue
                αL > length(cft.basis_bond.states[n_L]) && continue
                h_cut = h_max - 1.0
                ratios = Float64[]
                for ℓ in ells
                    vd = compute_vertex(cft, ℓ)
                    mod_raw = modified_vertex_raw(vd; c=c)
                    c3 = contract_TL(mod_raw, vd, n_T, αT, n_L, αL)
                    r = convergence_ratio_1leg(c3, vd.basis_bond; h_cut=h_cut)
                    push!(ratios, r.ratio)
                end
                results[(n_T, n_L, h_max)] = ratios
            end
        end
    end
    results
end

# ╔═╡ a0000001-0042-0000-0000-000000000001
md"""
### Exp 3 results: $\psi_T = |0; 0\rangle$, $\psi_L = |0; 0\rangle$
"""

# ╔═╡ a0000001-0043-0000-0000-000000000001
let
    header = "  ℓ   " * join(["h=$(Int(h))" for h in h_maxs], "    ")
    lines = [header, repeat("-", length(header))]
    for (i, ℓ) in enumerate(ells)
        vals = Float64[]
        for h in h_maxs
            key = (0, 0, h)
            if haskey(exp3_results, key) && i ≤ length(exp3_results[key])
                push!(vals, round(exp3_results[key][i]; digits=4))
            else
                push!(vals, NaN)
            end
        end
        push!(lines, " $(round(ℓ, digits=1))  " * join([@sprintf("%7.4f", v) for v in vals], "  "))
    end
    Text(join(lines, "\n"))
end

# ╔═╡ a0000001-0044-0000-0000-000000000001
md"""
### Exp 3 results: $\psi_T = |0; 1\rangle$, $\psi_L = |0; -1\rangle$
"""

# ╔═╡ a0000001-0045-0000-0000-000000000001
let
    header = "  ℓ   " * join(["h=$(Int(h))" for h in h_maxs], "    ")
    lines = [header, repeat("-", length(header))]
    for (i, ℓ) in enumerate(ells)
        vals = Float64[]
        for h in h_maxs
            key = (1, -1, h)
            if haskey(exp3_results, key) && i ≤ length(exp3_results[key])
                push!(vals, round(exp3_results[key][i]; digits=4))
            else
                push!(vals, NaN)
            end
        end
        push!(lines, " $(round(ℓ, digits=1))  " * join([@sprintf("%7.4f", v) for v in vals], "  "))
    end
    Text(join(lines, "\n"))
end

# ╔═╡ a0000001-0050-0000-0000-000000000001
md"""
## Summary

Convergence ratio close to 1 means the top shell ($h = h_{\max}$) contributes
little — the truncation is converged at that $h_{\max}$ for that $\ell$.

Expect: convergence improves (ratio → 1) as $h_{\max}$ increases and as $\ell$
decreases. If the ratio stays far from 1 even at $h_{\max} = 5$ for moderate
$\ell$, the truncation needs higher cutoff.
"""

# ╔═╡ Cell order:
# ╟─a0000001-0010-0000-0000-000000000001
# ╠═a0000001-0001-0000-0000-000000000001
# ╠═a0000001-0002-0000-0000-000000000001
# ╠═a0000001-0003-0000-0000-000000000001
# ╠═a0000001-0004-0000-0000-000000000001
# ╠═a0000001-0024-0000-0000-000000000001
# ╠═a0000001-0011-0000-0000-000000000001
# ╟─a0000001-0020-0000-0000-000000000001
# ╠═a0000001-0021-0000-0000-000000000001
# ╟─a0000001-0022-0000-0000-000000000001
# ╠═a0000001-0023-0000-0000-000000000001
# ╟─a0000001-0030-0000-0000-000000000001
# ╠═a0000001-0031-0000-0000-000000000001
# ╟─a0000001-0032-0000-0000-000000000001
# ╠═a0000001-0033-0000-0000-000000000001
# ╟─a0000001-0034-0000-0000-000000000001
# ╠═a0000001-0035-0000-0000-000000000001
# ╟─a0000001-0040-0000-0000-000000000001
# ╠═a0000001-0041-0000-0000-000000000001
# ╟─a0000001-0042-0000-0000-000000000001
# ╠═a0000001-0043-0000-0000-000000000001
# ╟─a0000001-0044-0000-0000-000000000001
# ╠═a0000001-0045-0000-0000-000000000001
# ╟─a0000001-0050-0000-0000-000000000001
