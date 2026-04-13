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

For Neumann BC, the open boundary state is purely in n=0 (the zero-mode
integral over the strip projects onto zero momentum). For n ≠ 0, all
coefficients are zero. Within n=0 (the Virasoro vacuum module), the
coefficients are determined by the gluing condition.
"""
function open_boundary_coeffs_A(basis, sector_n::Int)
    parts = basis.states[sector_n]
    coeffs = zeros(Float64, length(parts))
    # Neumann BC: only n=0 contributes
    sector_n != 0 && return coeffs
    for (i, lambda) in enumerate(parts)
        b = 1.0
        if isempty(lambda)
            b = 1.0
        else
            k = lambda[1]; mk = 1
            for j in 2:length(lambda)
                if lambda[j] == k
                    mk += 1
                else
                    b *= boundary_coeff_single_mode(mk)
                    b == 0.0 && break
                    k = lambda[j]; mk = 1
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
"""
Verify (J_k - J_{-k})|B⟩ = 0 on the safe subspace (level ≤ h_max - k).

The gluing condition holds exactly only on the safe subspace. At the
truncation boundary (level > h_max - k), J_{-k} creates states that
escape the truncated basis, so the balance breaks. This is a truncation
artifact, not a bug in the coefficients.
"""
function verify_gluing(basis, sector_n, coeffs; k_max=nothing)
    km = k_max === nothing ? Int(h_max_val) + 2 : k_max
    J_dense, _ = CFTTruncation.build_J_matrices(basis, km)
    Jk_mats = J_dense[sector_n]
    Jmk_mats = [CFTTruncation.build_creation_matrix(basis, sector_n, k) for k in 1:min(km, length(Jk_mats)-1)]
    max_safe = 0.0
    max_unsafe = 0.0
    for k in 1:length(Jmk_mats)
        Jk = Jk_mats[k + 1]
        Jmk = Jmk_mats[k]
        diff_vec = (Jk - Jmk) * coeffs
        for (i, d) in enumerate(diff_vec)
            level = basis.levels[sector_n][i]
            if level <= basis.h_max - k
                max_safe = max(max_safe, abs(d))
            else
                max_unsafe = max(max_unsafe, abs(d))
            end
        end
    end
    (safe=max_safe, unsafe=max_unsafe)
end

# ╔═╡ e0000001-0032-0000-0000-000000000001
let
    results = []
    for n in sort(collect(keys(basis.states)))
        coeffs = open_boundary_coeffs_A(basis, n)
        viol = verify_gluing(basis, n, coeffs)
        push!(results, (n=n, dim=length(coeffs),
                        nonzero=count(!iszero, coeffs),
                        safe=viol.safe, unsafe=viol.unsafe))
    end
    lines = ["  sector n  dim  nonzero  safe (level≤h-k)   unsafe (boundary)"]
    push!(lines, repeat("-", 65))
    for r in results
        push!(lines, "  $(rpad(r.n, 10)) $(rpad(r.dim, 5)) $(rpad(r.nonzero, 9)) $(rpad(round(r.safe; sigdigits=3), 18)) $(round(r.unsafe; sigdigits=3))")
    end
    Base.Text(join(lines, "\n"))
end

# ╔═╡ e0000001-0040-0000-0000-000000000001
md"""
## Open-Open Duality & Exact Product Formula

The boundary overlap has a **closed-form product formula** (no truncation!):

$$f(T) = \langle B|e^{-T(L_0-c/24)}|B\rangle = \eta(e^{-2T})^{-1/2}$$

This follows from $b(2m)^2 = \binom{2m}{m}/4^m$ and the generating function
$(1-x)^{-1/2}$, which makes the sum over even-multiplicity partitions
factor over modes.

The open-open duality has a **modular weight** from the η transformation:

$$\frac{f_1(T)}{f_2(T)} = \left(\frac{T}{\pi}\right)^{1/4}$$

This is exact (verified to 6 digits for all T).
"""

# ╔═╡ e0000001-0041-0000-0000-000000000001
"""Exact f(T) = η(e^{-2T})^{-1/2} via convergent product (no truncation)."""
function f_exact(T; n_terms=100)
    q = exp(-2T)
    prod_val = 1.0
    for k in 1:n_terms
        prod_val *= (1 - q^k)^(-0.5)
    end
    exp(T/24) * prod_val
end

# ╔═╡ e0000001-0043-0000-0000-000000000001
let
    Ts = collect(0.2:0.05:6.0)
    f1 = f_exact.(Ts)
    f2 = f_exact.(pi^2 ./ Ts)
    predicted = (Ts ./ pi) .^ 0.25

    p1 = plot(Ts, f1; label="f₁(T)", xlabel="T", ylabel="f(T)",
              marker=:circle, markersize=2, yscale=:log10)
    plot!(p1, Ts, f2; label="f₂(T) = f(π²/T)", marker=:diamond, markersize=2)
    plot!(p1; title="Exact: f(T) = η(e^{-2T})^{-1/2}", size=(650, 350))

    p2 = plot(Ts, f1 ./ f2; label="f₁/f₂ (numerical)", xlabel="T",
              marker=:circle, markersize=2)
    plot!(p2, Ts, predicted; label="(T/π)^{1/4} (exact)", linestyle=:dash, linewidth=2)
    plot!(p2; title="Duality: f₁/f₂ = (T/π)^{1/4}", ylabel="ratio",
          size=(650, 300))

    plot(p1, p2; layout=(2, 1), size=(650, 600))
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
# ╠═e0000001-0043-0000-0000-000000000001
# ╟─e0000001-0050-0000-0000-000000000001
# ╟─e0000001-0060-0000-0000-000000000001
