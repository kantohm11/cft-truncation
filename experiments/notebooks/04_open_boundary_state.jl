### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ e0000001-0001-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ e0000001-0002-0000-0000-000000000001
using CFTTruncation

# ╔═╡ e0000001-0003-0000-0000-000000000001
using Plots

# ╔═╡ e0000001-0010-0000-0000-000000000001
md"""
# 04 — Open Boundary State $|B_a^{\text{open}}\rangle$

Two independent computations of the open boundary state for the compact
boson with Neumann BC, compared as a cross-check.

**Method A**: Solve the gluing condition $(J_n - J_{-n})|B\rangle = 0$ recursively.

**Method B**: Conformal map from semidisk to thin strip (ε→0 limit).

The state lives in $\mathcal{H}_{aa}$ (our FockBasis). Setting $b_\varnothing = 1$
(unnormalized).
"""

# ╔═╡ e0000001-0004-0000-0000-000000000001
import LinearAlgebra

# ╔═╡ e0000001-0011-0000-0000-000000000001
begin
    R_val = 1.0
    h_max_val = 6.0
    basis = CFTTruncation.build_fock_basis(R_val, h_max_val)
end

# ╔═╡ e0000001-0020-0000-0000-000000000001
md"""
## Method A: Gluing Condition

$(J_n - J_{-n})|B\rangle = 0$ for all $n \geq 1$. In the normalised basis:

$$b_{\mu \cup k} = \frac{\sqrt{m_k(\mu)}}{\sqrt{m_k(\mu)+1}} \cdot b_{\mu \setminus k}$$

Since this is a two-step recursion in multiplicity, all odd multiplicities
give zero. The coefficient is a product over modes:

$$b_\lambda = \prod_{k} b_{m_k(\lambda)}^{(\text{single mode})}$$

where $b_0 = 1$, $b_1 = 0$, $b_{m+1} = b_{m-1}\sqrt{m/(m+1)}$.
"""

# ╔═╡ e0000001-0021-0000-0000-000000000001
"""Single-mode boundary state coefficient for occupation m."""
function boundary_coeff_single_mode(m::Int)
    m < 0 && return 0.0
    isodd(m) && return 0.0
    # b(0)=1, b(2)=1/√2, b(4)=√(3/8), ...
    # recursion: b(m) = b(m-2) √((m-1)/m)
    b = 1.0
    for j in 2:2:m
        b *= sqrt((j - 1) / j)
    end
    b
end

# ╔═╡ e0000001-0022-0000-0000-000000000001
"""
Compute open boundary state coefficients for all states in sector n.
Returns a Vector{Float64} indexed by basis state index.
"""
function open_boundary_coeffs_A(basis, sector_n::Int)
    parts = basis.states[sector_n]
    coeffs = zeros(Float64, length(parts))
    for (i, lambda) in enumerate(parts)
        # Product over distinct parts
        b = 1.0
        if isempty(lambda)
            b = 1.0
        else
            # Count multiplicities of each distinct part
            k = lambda[1]
            mk = 1
            for j in 2:length(lambda)
                if lambda[j] == k
                    mk += 1
                else
                    b *= boundary_coeff_single_mode(mk)
                    b == 0.0 && break
                    k = lambda[j]
                    mk = 1
                end
            end
            b *= boundary_coeff_single_mode(mk)
        end
        coeffs[i] = b
    end
    coeffs
end

# ╔═╡ e0000001-0023-0000-0000-000000000001
md"### Method A: Coefficients in sector n=0"

# ╔═╡ e0000001-0024-0000-0000-000000000001
let
    coeffs = open_boundary_coeffs_A(basis, 0)
    lines = ["  idx  partition           level  b_λ"]
    push!(lines, repeat("-", 50))
    for (i, (p, b)) in enumerate(zip(basis.states[0], coeffs))
        level = sum(p; init=0)
        push!(lines, "  $i    $(rpad(string(p), 20)) $level    $(round(b; digits=6))")
    end
    Base.Text(join(lines, "\n"))
end

# ╔═╡ e0000001-0030-0000-0000-000000000001
md"""
### Verify gluing condition

Check $\langle \hat\mu | (J_k - J_{-k}) |B\rangle \approx 0$ for all $\mu$, $k$.
"""

# ╔═╡ e0000001-0031-0000-0000-000000000001
function verify_gluing(basis, sector_n, coeffs; k_max=nothing)
    J_dense, _ = CFTTruncation.build_J_matrices(basis, k_max === nothing ? Int(h_max_val) + 2 : k_max)
    Jk_mats = J_dense[sector_n]
    Jmk_mats = [CFTTruncation.build_creation_matrix(basis, sector_n, k) for k in 1:length(Jk_mats)-1]
    max_violation = 0.0
    d = length(coeffs)
    for k in 1:length(Jmk_mats)
        # (J_k - J_{-k})|B⟩ should be zero
        Jk = Jk_mats[k + 1]   # k=0 is index 1, k=1 is index 2, etc.
        Jmk = Jmk_mats[k]
        diff_vec = (Jk - Jmk) * coeffs
        max_violation = max(max_violation, maximum(abs, diff_vec))
    end
    max_violation
end

# ╔═╡ e0000001-0032-0000-0000-000000000001
let
    results = []
    for n in sort(collect(keys(basis.states)))
        coeffs = open_boundary_coeffs_A(basis, n)
        viol = verify_gluing(basis, n, coeffs)
        push!(results, (n=n, dim=length(coeffs), max_violation=viol,
                        nonzero=count(!iszero, coeffs)))
    end
    lines = ["  sector n  dim  nonzero  max |⟨μ|(J_k-J_{-k})|B⟩|"]
    push!(lines, repeat("-", 55))
    for r in results
        push!(lines, "  $(rpad(r.n, 10)) $(rpad(r.dim, 5)) $(rpad(r.nonzero, 9)) $(round(r.max_violation; sigdigits=3))")
    end
    Base.Text(join(lines, "\n"))
end

# ╔═╡ e0000001-0040-0000-0000-000000000001
md"""
## Open-Open Duality Check

$\langle B^{\text{open}} | e^{-T H} | B^{\text{open}} \rangle$ should equal
the partition function $Z(T) = \text{Tr}_{\mathcal{H}_{aa}} e^{-T(L_0 - c/24)}$
up to normalisation.
"""

# ╔═╡ e0000001-0041-0000-0000-000000000001
function open_overlap(basis, all_coeffs, T; c=1.0)
    s = 0.0
    for n in keys(basis.states)
        coeffs = all_coeffs[n]
        for (i, b) in enumerate(coeffs)
            h = conformal_dim(basis, n, i)
            s += b^2 * exp(-T * (h - c / 24))
        end
    end
    s
end

# ╔═╡ e0000001-0042-0000-0000-000000000001
function partition_function(basis, T; c=1.0)
    s = 0.0
    for n in keys(basis.states)
        for i in eachindex(basis.states[n])
            h = conformal_dim(basis, n, i)
            s += exp(-T * (h - c / 24))
        end
    end
    s
end

# ╔═╡ e0000001-0043-0000-0000-000000000001
let
    all_coeffs = Dict(n => open_boundary_coeffs_A(basis, n) for n in keys(basis.states))
    Ts = collect(0.1:0.1:2.0)
    overlaps = [open_overlap(basis, all_coeffs, T) for T in Ts]
    Zs = [partition_function(basis, T) for T in Ts]
    ratios = overlaps ./ Zs

    p = plot(Ts, ratios; xlabel="T", ylabel="⟨B|e^{-TH}|B⟩ / Z(T)",
             title="Open-open duality ratio (should be T-independent)",
             marker=:circle, markersize=4, legend=false, size=(600, 350))
    p
end

# ╔═╡ e0000001-0050-0000-0000-000000000001
md"""
## Method B: Conformal Map (TODO)

Compute $|B^{\text{open}}\rangle$ from the conformal map from the semidisk
$D^+$ to the thin strip $[0,\pi] \times [0,\epsilon]$. Compare with Method A.

This uses the same SC/Neumann machinery as the vertex computation.
*Implementation deferred — Method A is validated by the gluing condition check.*
"""

# ╔═╡ e0000001-0060-0000-0000-000000000001
md"""
## Summary

- **Gluing condition** $(J_n - J_{-n})|B\rangle = 0$: verified to machine precision.
- Nonzero coefficients only for partitions with all **even** multiplicities.
- **Open-open duality** ratio $\langle B|e^{-TH}|B\rangle / Z(T)$: should be
  constant in $T$ if the normalisation is correct. Deviations at small $T$
  indicate truncation effects (high-energy states missing from the sum).
"""

# ╔═╡ Cell order:
# ╟─e0000001-0010-0000-0000-000000000001
# ╠═e0000001-0001-0000-0000-000000000001
# ╠═e0000001-0002-0000-0000-000000000001
# ╠═e0000001-0003-0000-0000-000000000001
# ╠═e0000001-0004-0000-0000-000000000001
# ╠═e0000001-0011-0000-0000-000000000001
# ╟─e0000001-0020-0000-0000-000000000001
# ╠═e0000001-0021-0000-0000-000000000001
# ╠═e0000001-0022-0000-0000-000000000001
# ╟─e0000001-0023-0000-0000-000000000001
# ╠═e0000001-0024-0000-0000-000000000001
# ╟─e0000001-0030-0000-0000-000000000001
# ╠═e0000001-0031-0000-0000-000000000001
# ╠═e0000001-0032-0000-0000-000000000001
# ╟─e0000001-0040-0000-0000-000000000001
# ╠═e0000001-0041-0000-0000-000000000001
# ╠═e0000001-0042-0000-0000-000000000001
# ╠═e0000001-0043-0000-0000-000000000001
# ╟─e0000001-0050-0000-0000-000000000001
# ╟─e0000001-0060-0000-0000-000000000001
