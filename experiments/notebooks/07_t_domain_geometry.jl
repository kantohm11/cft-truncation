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
using Plots

# ╔═╡ b0000001-0004-0000-0000-000000000001
using Printf

# ╔═╡ b0000001-0010-0000-0000-000000000001
md"""
# 07 — T-Domain Geometry

Visualize the Schwarz-Christoffel map $f(z) = \int f'(z)\,dz$ for the T-shaped domain.

$f'(z) = \frac{C\sqrt{z^2 - p^2}}{z(z^2 - 1)}$

- **3 semi-infinite arms** at poles $z = 0, \pm 1$
- **2 reflex corners** (270°) at zeros $z = \pm p$
- **T arm width** = $\ell$ (verified exactly)
"""

# ╔═╡ b0000001-0020-0000-0000-000000000001
md"## Numerical integration of $f(z)$ along boundary"

# ╔═╡ b0000001-0021-0000-0000-000000000001
"""
Trace the T-domain boundary by integrating f'(z) along z = x + iε,
x from +xr to -xr. Uses the correct sqrt branch (Im ≥ 0 in UHP).
Returns (σ, τ, corner_indices, p).
"""
function trace_boundary(ell; npts=500000, xr=10.0, eps=1e-4)
    sc = CFTTruncation.compute_sc_params(ell)
    p = sc.p; C = sc.C
    function fp(z)
        w = z^2 - p^2; s = sqrt(w)
        if imag(s) < 0; s = -s; end
        C * s / (z * (z^2 - 1))
    end
    dx = -2xr / npts
    ss = Float64[]; tt = Float64[]
    w = ComplexF64(0)
    for i in 0:npts
        x = xr + i * dx
        push!(ss, real(w)); push!(tt, imag(w))
        if i < npts
            w += fp((x + dx/2) + eps*im) * dx
        end
    end
    ip = round(Int, (p - xr) / dx) + 1
    ilp = round(Int, (-p - xr) / dx) + 1
    # Center: place R corner at σ = -ℓ/2 (geometric condition)
    shift = -ell/2 - ss[ip]
    ss .+= shift
    return ss, tt, ip, ilp, p
end

# ╔═╡ b0000001-0022-0000-0000-000000000001
"""
Extend the T arm walls to τ → ∞ using the T arm series (which includes
log(z) analytically). Matches the numerical integration at the corner z = p.
"""
function t_arm_extension(ell, ss_num, tt_num, ir; npts=300)
    geom = CFTTruncation.compute_geometry(ell, 30)
    p = geom.sc.p
    arm_T = geom.arms.T
    freg = CFTTruncation.regular_part(arm_T.fprime_series)
    order = length(freg.coeffs)
    res_T = -im * ell / pi

    function eval_T_raw(z)
        w = res_T * log(Complex(z))
        zp = ComplexF64(1)
        for n in 1:order; zp *= z; w += freg[n-1]/n * zp; end
        w
    end

    # ρ₀^T from matching at corner z = p
    w_corner_num = ss_num[ir] + im * tt_num[ir]
    rho0_T = w_corner_num - eval_T_raw(p)
    eval_T(z) = eval_T_raw(z) + rho0_T

    # Right wall: z from p → 0⁺ (log scale)
    zs_r = [p * 10.0^(-t) for t in range(0, 8, length=npts)]
    wr = eval_T.(zs_r)
    # Left wall: z from -p → 0⁻
    zs_l = [-p * 10.0^(-t) for t in range(0, 8, length=npts)]
    wl = eval_T.(zs_l)

    return real.(wr), imag.(wr), real.(wl), imag.(wl)
end

# ╔═╡ b0000001-0030-0000-0000-000000000001
md"## T-domain for multiple $\ell$ values"

# ╔═╡ b0000001-0031-0000-0000-000000000001
let
    plt = plot(layout=(1, 3), size=(1500, 550), margin=6Plots.mm)

    for (col, ell) in enumerate([0.5, 1.0, 2.0])
        s, t, ir, il, p = trace_boundary(ell)
        sr, tr, sl, tl = t_arm_extension(ell, s, t, ir)

        width = s[il] - s[ir]
        xlo = s[ir] - 1.5; xhi = s[il] + 1.5
        yhi = min(max(tr..., tl...) * 0.3, 10.0)

        plot!(plt, s, t; subplot=col, xlabel="σ", ylabel="τ",
              title=@sprintf("ℓ = %.1f  (width = %.4f)", ell, width),
              legend=false, lw=2, color=:blue,
              xlim=(xlo, xhi), ylim=(-0.5, yhi))
        # T arm walls from series (extending to ∞)
        plot!(plt, sr, tr; subplot=col, lw=2, color=:red)
        plot!(plt, sl, tl; subplot=col, lw=2, color=:red)
        # Corners
        scatter!(plt, [s[ir], s[il]], [t[ir], t[il]];
                 subplot=col, color=:black, ms=5)
        # Annotations
        annotate!(plt, subplot=col, [
            (s[ir], t[ir]-0.3, text(@sprintf("σ=%.2f", s[ir]), 8)),
            (s[il], t[il]-0.3, text(@sprintf("σ=%.2f", s[il]), 8))])
    end
    plt
end

# ╔═╡ b0000001-0040-0000-0000-000000000001
md"""
## Key properties

| Property | Value |
|----------|-------|
| T arm width | $= \ell$ exactly |
| Both corners | at $\tau = 1$ |
| R arm | $\sigma \to -\infty$ at $z = +1$ (lower: $\tau=0$, upper: $\tau=1$) |
| L arm | $\sigma \to +\infty$ at $z = -1$ (lower: $\tau=0$, upper: $\tau=1$) |
| T arm | $\tau \to +\infty$ at $z = 0$ (right wall: $\sigma = -\ell/2$, left wall: $\sigma = +\ell/2$) |
| Outer boundary | connects at $z \to \pm\infty$ |

The horizontal strip has width 1 (in the width-1 convention, physical width $\pi$).

When $|B^{\text{open}}\rangle$ seals the T arm: the notch closes, giving a straight strip
of width 1 with the L and R arms separated by $\ell$.
"""

# ╔═╡ b0000001-0050-0000-0000-000000000001
md"## Corner positions vs $\ell$"

# ╔═╡ b0000001-0051-0000-0000-000000000001
let
    ells = 0.1:0.1:3.0
    widths = Float64[]
    for ell in ells
        s, t, ir, il, p = trace_boundary(ell; npts=200000)
        push!(widths, s[il] - s[ir])
    end
    plot(collect(ells), widths;
         xlabel="ℓ", ylabel="T arm width",
         title="T arm width vs ℓ (should be identity)",
         marker=:circle, markersize=3, legend=false, size=(600, 400))
    plot!(collect(ells), collect(ells); linestyle=:dash, color=:red, label="y = ℓ")
end

# ╔═╡ Cell order:
# ╟─b0000001-0010-0000-0000-000000000001
# ╠═b0000001-0001-0000-0000-000000000001
# ╠═b0000001-0002-0000-0000-000000000001
# ╠═b0000001-0003-0000-0000-000000000001
# ╠═b0000001-0004-0000-0000-000000000001
# ╟─b0000001-0020-0000-0000-000000000001
# ╠═b0000001-0021-0000-0000-000000000001
# ╠═b0000001-0022-0000-0000-000000000001
# ╟─b0000001-0030-0000-0000-000000000001
# ╠═b0000001-0031-0000-0000-000000000001
# ╟─b0000001-0040-0000-0000-000000000001
# ╟─b0000001-0050-0000-0000-000000000001
# ╠═b0000001-0051-0000-0000-000000000001
