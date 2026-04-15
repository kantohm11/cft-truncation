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

**Geometry**: inserting $|B^{\text{open}}\rangle$ seals the T arm notch,
producing a straight strip of width 1 between the L and R mouths, with
propagation distance $d = \pi\ell$.

**Convergence**: The Neumann series on the T arm has $R_{\text{conv}} = 1$
(corners at $|\xi_T| = \pm 1$). This is borderline for the boundary state
contraction. At small $\ell$ the T arm contribution is suppressed
($|\alpha_T| \approx 2/\ell \gg 1$) and the series converges.
Adding damping $e^{-\varepsilon L_0}$ improves convergence at the cost
of modifying the physics (sealing the arm deeper, not at the corner).
"""

# ╔═╡ f0000001-0011-0000-0000-000000000001
begin
    R_val = 1.0
    h_bond = 4.0
end

# ╔═╡ f0000001-0012-0000-0000-000000000001
md"## Helpers"

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
## Contraction

$$O_{\alpha_L, \alpha_R} = \sum_{\alpha_T} B_{\alpha_T} \cdot e^{-\varepsilon N_T} \cdot \eta_{\alpha_L} \cdot V(\alpha_T, \alpha_L, \alpha_R)$$

where $\eta$ is the BPZ sign, $\varepsilon \geq 0$ is optional damping,
and $N_T$ is the descendant level on the T arm.
"""

# ╔═╡ f0000001-0021-0000-0000-000000000001
function compute_O(ell, h_bond, h_phys; eps=0.0)
    cft = CompactBosonCFT(R=R_val, trunc=TruncationSpec(h_bond=h_bond, h_phys=h_phys))
    vd = compute_vertex(cft, ell; cache=:off)
    V = vd.vertex
    B = boundary_coeffs(cft.basis_phys)
    if eps > 0
        for (i, lambda) in enumerate(cft.basis_phys.states[0])
            N = sum(lambda; init=0)
            B[i] *= exp(-eps * N)
        end
    end

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
## 1. Small-$\ell$ limit (undamped, $\varepsilon = 0$)

At small $\ell$, the T arm contribution is suppressed ($|\alpha_T| \approx 2/\ell \gg 1$),
so the boundary state contraction converges even without damping.
Expect $d/(\pi\ell) \to 1$ and off-diagonal $\to 0$ as $\ell \to 0$.
"""

# ╔═╡ f0000001-0031-0000-0000-000000000001
let
    ells = [0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    h_phys = 10.0

    lines = ["  ℓ       d/(πℓ)    offdiag    |α_T|     |Δr| (h8 vs h10)"]
    push!(lines, repeat("-", 62))

    ratios = Float64[]; offdiags = Float64[]

    for ell in ells
        O8, bb8 = compute_O(ell, h_bond, 8.0)
        r8 = analyze_O(O8, bb8)
        O10, bb = compute_O(ell, h_bond, h_phys)
        r10 = analyze_O(O10, bb)
        geom = CFTTruncation.compute_geometry(ell, 20)
        aT = abs(geom.arms.T.α)
        delta = abs(r10.d/(π*ell) - r8.d/(π*ell))
        push!(ratios, r10.d/(π*ell))
        push!(offdiags, r10.offdiag)
        push!(lines, @sprintf("  %.3f    %7.5f   %7.5f    %6.1f     %.1e",
            ell, r10.d/(π*ell), r10.offdiag, aT, delta))
    end

    Base.Text(join(lines, "\n"))
end

# ╔═╡ f0000001-0032-0000-0000-000000000001
let
    ells = [0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    h_phys = 10.0

    ratios = Float64[]; offdiags = Float64[]
    for ell in ells
        O, bb = compute_O(ell, h_bond, h_phys)
        r = analyze_O(O, bb)
        push!(ratios, r.d / (π * ell))
        push!(offdiags, r.offdiag)
    end

    p1 = plot(ells, ratios; xlabel="ℓ", ylabel="d/(πℓ)",
              title="Propagation distance ratio (eps=0)",
              marker=:circle, markersize=4, legend=false, size=(600, 280))
    hline!(p1, [1.0]; color=:red, linestyle=:dash)

    p2 = plot(ells, offdiags; xlabel="ℓ", ylabel="off-diagonal fraction",
              title="Diagonality (0 = perfect propagator)",
              marker=:circle, markersize=4, legend=false, size=(600, 280))

    plot(p1, p2; layout=(2, 1), size=(650, 550))
end

# ╔═╡ f0000001-0040-0000-0000-000000000001
md"""
## 2. Convergence with damping ($\ell = 0.05$ and $0.1$)

The damped state $e^{-\varepsilon L_0}|B^{\text{open}}\rangle$ seals the T arm
at distance $\varepsilon$ from the corner, **not** at the corner itself.
The resulting operator is not a propagator even at $h_{\text{phys}} \to \infty$ —
it has genuine off-diagonal terms from the T-arm stub.
But convergence improves, confirming the vertex is correct.
"""

# ╔═╡ f0000001-0041-0000-0000-000000000001
let
    epss = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    lines = String[]
    for ell in [0.05, 0.1]
        push!(lines, @sprintf("ℓ = %.2f", ell))
        push!(lines, "  eps     d/(πℓ)    offdiag   |Δr| (h8 vs h10)")
        push!(lines, repeat("-", 50))
        for eps in epss
            O8, bb8 = compute_O(ell, h_bond, 8.0; eps=eps)
            r8 = analyze_O(O8, bb8)
            O10, bb = compute_O(ell, h_bond, 10.0; eps=eps)
            r10 = analyze_O(O10, bb)
            delta = abs(r10.d/(π*ell) - r8.d/(π*ell))
            push!(lines, @sprintf("  %.2f    %6.4f    %6.4f    %.1e",
                eps, r10.d/(π*ell), r10.offdiag, delta))
        end
        push!(lines, "")
    end

    Base.Text(join(lines, "\n"))
end

# ╔═╡ f0000001-0050-0000-0000-000000000001
md"""
## 3. $h_{\text{phys}}$ convergence at moderate $\ell$

At $\ell = 1$, the undamped series is marginally divergent ($R_{\text{conv}} = 1$).
Adding damping $\varepsilon > 0.3$ stabilizes the sum (identical across
$h_{\text{phys}} = 6, 8, 10$).
"""

# ╔═╡ f0000001-0051-0000-0000-000000000001
let
    epss = [0.0, 0.1, 0.3, 0.5, 1.0]
    ell = 1.0

    lines = ["ℓ = $ell, h_phys = 6 / 8 / 10"]
    push!(lines, "  eps     r(6)     r(8)     r(10)    off(10)   stable?")
    push!(lines, repeat("-", 60))

    for eps in epss
        rs = Float64[]; os = Float64[]
        for hp in [6.0, 8.0, 10.0]
            O, bb = compute_O(ell, h_bond, hp; eps=eps)
            r = analyze_O(O, bb)
            push!(rs, r.d/(π*ell)); push!(os, r.offdiag)
        end
        stable = abs(rs[3] - rs[2]) < 1e-3 ? "yes" : "no"
        push!(lines, @sprintf("  %.1f    %6.4f   %6.4f   %6.4f   %6.4f     %s",
            eps, rs[1], rs[2], rs[3], os[3], stable))
    end

    Base.Text(join(lines, "\n"))
end

# ╔═╡ f0000001-0060-0000-0000-000000000001
md"""
## 4. Diagonal entries at $\ell = 0.05$
"""

# ╔═╡ f0000001-0061-0000-0000-000000000001
let
    ell = 0.05
    O, bb = compute_O(ell, h_bond, 10.0)
    parts = bb.states[0]
    d = size(O, 1)
    r = analyze_O(O, bb)

    lines = [@sprintf("d/(πℓ) = %.5f,  off-diagonal = %.5f", r.d/(π*ell), r.offdiag)]
    push!(lines, "")
    push!(lines, "  α   level  partition         O[α,α]")
    push!(lines, repeat("-", 50))
    for i in 1:min(10, d)
        v = O[i, i]; lv = bb.levels[0][i]; λ = parts[i]
        push!(lines, @sprintf("  %2d  %d      %-15s  %+14.8f",
            i, lv, string(λ), v))
    end
    push!(lines, "")
    push!(lines, @sprintf("  Off-diagonal max: %.2e",
        maximum(abs(O[i,j]) for i in 1:d, j in 1:d if i != j)))

    Base.Text(join(lines, "\n"))
end

# ╔═╡ f0000001-0070-0000-0000-000000000001
md"""
## Summary

- The vertex produces the correct propagator $d = \pi\ell$ in the $\ell \to 0$
  limit ($d/(\pi\ell) \to 1$, off-diagonal $\to 0$), even **without damping**.
- At moderate $\ell$, the Neumann series has $R_{\text{conv}} = 1$ (corners at
  $|\xi_T| = \pm 1$), which is borderline. Adding $\varepsilon > 0$ damping
  stabilizes the series but changes the physical answer (seals the arm deeper,
  producing a non-diagonal operator with a T-arm stub).
- No numerical issues at very small $\ell$ ($|\alpha_T| \approx 2/\ell$ up to 500+).
- The departure from the exact propagator scales as $O(\ell^2)$, consistent with
  the design document (plaquette\_amplitude.md, Section 5.4).
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
# ╠═f0000001-0032-0000-0000-000000000001
# ╟─f0000001-0040-0000-0000-000000000001
# ╠═f0000001-0041-0000-0000-000000000001
# ╟─f0000001-0050-0000-0000-000000000001
# ╠═f0000001-0051-0000-0000-000000000001
# ╟─f0000001-0060-0000-0000-000000000001
# ╠═f0000001-0061-0000-0000-000000000001
# ╟─f0000001-0070-0000-0000-000000000001
