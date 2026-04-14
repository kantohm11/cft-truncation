### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ f0000001-0001-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ f0000001-0002-0000-0000-000000000001
using CFTTruncation

# ╔═╡ f0000001-0003-0000-0000-000000000001
using TensorKit

# ╔═╡ f0000001-0004-0000-0000-000000000001
using Plots

# ╔═╡ f0000001-0005-0000-0000-000000000001
using Printf

# ╔═╡ f0000001-0010-0000-0000-000000000001
md"""
# 05 — Contract Vertex with $|B^{\text{open}}\rangle$: Propagator Check

Contract the **raw** T-vertex $V_\ell$ with $|B^{\text{open}}\rangle$ on the T arm.
The result should be a **propagator** (diagonal operator) with propagation
distance $d = \pi\ell$ (in the code's width-1 convention).

The geometry: inserting $|B^{\text{open}}\rangle$ seals the T arm notch,
producing a straight strip of width $\ell$ between the L and R mouths.
"""

# ╔═╡ f0000001-0011-0000-0000-000000000001
begin
    R_val = 1.0
    h_bond = 4.0
end

# ╔═╡ f0000001-0012-0000-0000-000000000001
md"### Open boundary state coefficients"

# ╔═╡ f0000001-0013-0000-0000-000000000001
"""Single-mode squeezed vacuum coefficient: b(even)=∏√((2j-1)/(2j)), b(odd)=0."""
_bc(m) = m < 0 ? 0.0 : isodd(m) ? 0.0 : (b = 1.0; for j in 2:2:m; b *= sqrt((j-1)/j); end; b)

# ╔═╡ f0000001-0014-0000-0000-000000000001
function boundary_coeffs(basis)
    parts = basis.states[0]
    coeffs = zeros(Float64, length(parts))
    for (i, lambda) in enumerate(parts)
        b = isempty(lambda) ? 1.0 : begin
            k = lambda[1]; mk = 1; bv = 1.0
            for j in 2:length(lambda)
                lambda[j] == k ? (mk += 1) : (bv *= _bc(mk); bv == 0 && break; k = lambda[j]; mk = 1)
            end
            bv * _bc(mk)
        end
        coeffs[i] = b
    end
    coeffs
end

# ╔═╡ f0000001-0020-0000-0000-000000000001
md"""
### Compute $O(\ell)$: contract raw vertex with $|B^{\text{open}}\rangle$

$O_{\alpha_L, \alpha_R} = \sum_{\alpha_T} B_{\alpha_T} \cdot \eta_{\alpha_L} \cdot V(\alpha_T, \alpha_L, \alpha_R)$

where $\eta$ is the BPZ sign (applied to raise the L index).
"""

# ╔═╡ f0000001-0021-0000-0000-000000000001
function compute_O(ell, h_bond, h_phys)
    cft = CompactBosonCFT(R=R_val, trunc=TruncationSpec(h_bond=h_bond, h_phys=h_phys))
    vd = compute_vertex(cft, ell; cache=:off)
    V = vd.vertex
    B = boundary_coeffs(cft.basis_phys)

    d_L = length(cft.basis_bond.states[0])
    O = zeros(Float64, d_L, d_L)
    for (f1, f2) in fusiontrees(V)
        all(Int(f2.uncoupled[i].charge) == 0 for i in 1:3) || continue
        blk = V[f1, f2]
        for aL in 1:size(blk, 2), aR in 1:size(blk, 3)
            val = sum(B[aT] * blk[aT, aL, aR] for aT in 1:size(blk, 1))
            O[aL, aR] += CFTTruncation._bpz_sign(cft.basis_bond, 0, aL) * val
        end
    end
    O, cft.basis_bond
end

# ╔═╡ f0000001-0022-0000-0000-000000000001
function analyze_O(O, basis)
    d = size(O, 1)
    v0 = abs(O[1,1]); v1 = d >= 2 ? abs(O[2,2]) : NaN
    d_val = v1 > 0 ? log(v0/v1) : NaN
    diag_sq = sum(O[i,i]^2 for i in 1:d)
    offdiag = sqrt(max(0, 1 - diag_sq / sum(O .^ 2)))
    (d=d_val, offdiag=offdiag)
end

# ╔═╡ f0000001-0030-0000-0000-000000000001
md"""
### Results: $d/(\pi\ell)$ and off-diagonal fraction

Expect $d = \pi\ell$ (from the strip width $\ell$ in the width-1 convention).
Off-diagonal should be small (operator is a propagator).

**Note**: For $\ell < 1$, numerical stability degrades ($|\alpha_T|$ small →
Neumann coefficients lose precision). Results are reliable for $\ell \geq 1$.
"""

# ╔═╡ f0000001-0031-0000-0000-000000000001
let
    ells = [1.0, 1.5, 2.0, 2.5, 3.0]
    h_phys = 8.0

    ds = Float64[]; ods = Float64[]
    lines = ["  ℓ      d        d/(πℓ)    offdiag    |α_T|"]
    push!(lines, repeat("-", 55))

    for ell in ells
        O, bb = compute_O(ell, h_bond, h_phys)
        r = analyze_O(O, bb)
        push!(ds, r.d); push!(ods, r.offdiag)

        geom = CFTTruncation.compute_geometry(ell, 20)
        aT = abs(geom.arms.T.α)
        push!(lines, @sprintf("  %.1f    %6.3f   %6.4f    %6.4f     %.4f",
            ell, r.d, r.d/(π*ell), r.offdiag, aT))
    end

    Base.Text(join(lines, "\n"))
end

# ╔═╡ f0000001-0032-0000-0000-000000000001
md"### Plot: $d/(\pi\ell)$ vs $\ell$"

# ╔═╡ f0000001-0033-0000-0000-000000000001
let
    ells = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    h_phys = 8.0

    ratios = Float64[]; offdiags = Float64[]
    for ell in ells
        O, bb = compute_O(ell, h_bond, h_phys)
        r = analyze_O(O, bb)
        push!(ratios, r.d / (π * ell))
        push!(offdiags, r.offdiag)
    end

    p1 = plot(ells, ratios; xlabel="ℓ", ylabel="d/(πℓ)",
              title="Propagation distance ratio",
              marker=:circle, markersize=5, legend=false, size=(600, 300))
    hline!(p1, [1.0]; color=:red, linestyle=:dash)

    p2 = plot(ells, offdiags; xlabel="ℓ", ylabel="off-diagonal fraction",
              title="Diagonality (0 = perfect propagator)",
              marker=:circle, markersize=5, legend=false, size=(600, 300))

    plot(p1, p2; layout=(2, 1), size=(650, 550))
end

# ╔═╡ f0000001-0034-0000-0000-000000000001
md"### Diagonal entries of $O$ (ℓ = 2)"

# ╔═╡ f0000001-0035-0000-0000-000000000001
let
    O, bb = compute_O(2.0, h_bond, 8.0)
    parts = bb.states[0]
    d = size(O, 1)

    lines = ["  α   level  partition         O[α,α]          sign  BPZ"]
    push!(lines, repeat("-", 65))
    for i in 1:min(10, d)
        v = O[i, i]; lv = bb.levels[0][i]; λ = parts[i]
        s = v > 0 ? "+" : "-"
        bpz = CFTTruncation._bpz_sign(bb, 0, i) > 0 ? "+" : "-"
        push!(lines, @sprintf("  %2d  %d      %-15s  %+12.6f      %s     %s",
            i, lv, string(λ), v, s, bpz))
    end

    push!(lines, "")
    push!(lines, @sprintf("  Off-diagonal max: %.2e", maximum(abs(O[i,j]) for i in 1:d, j in 1:d if i != j)))

    Base.Text(join(lines, "\n"))
end

# ╔═╡ f0000001-0040-0000-0000-000000000001
md"""
## Summary

- $O = \langle B^{\text{open}} | V_\ell \rangle$ is the BPZ-signed propagator
- Propagation distance $d \approx \pi\ell$ (width-1 convention)
- Off-diagonal fraction small for $\ell \geq 1$ (where $|\alpha_T|$ is manageable)
- Numerical stability limited by $|\alpha_T| = e^{-\pi\tau_{\text{corner}}/\ell}$
  (exponentially small for small $\ell$)
"""

# ╔═╡ Cell order:
# ╟─f0000001-0010-0000-0000-000000000001
# ╠═f0000001-0001-0000-0000-000000000001
# ╠═f0000001-0002-0000-0000-000000000001
# ╠═f0000001-0003-0000-0000-000000000001
# ╠═f0000001-0004-0000-0000-000000000001
# ╠═f0000001-0005-0000-0000-000000000001
# ╠═f0000001-0011-0000-0000-000000000001
# ╟─f0000001-0012-0000-0000-000000000001
# ╠═f0000001-0013-0000-0000-000000000001
# ╠═f0000001-0014-0000-0000-000000000001
# ╟─f0000001-0020-0000-0000-000000000001
# ╠═f0000001-0021-0000-0000-000000000001
# ╠═f0000001-0022-0000-0000-000000000001
# ╟─f0000001-0030-0000-0000-000000000001
# ╠═f0000001-0031-0000-0000-000000000001
# ╟─f0000001-0032-0000-0000-000000000001
# ╠═f0000001-0033-0000-0000-000000000001
# ╟─f0000001-0034-0000-0000-000000000001
# ╠═f0000001-0035-0000-0000-000000000001
# ╟─f0000001-0040-0000-0000-000000000001
