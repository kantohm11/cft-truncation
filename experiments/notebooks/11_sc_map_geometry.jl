### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ f0000011-0001-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ f0000011-0002-0000-0000-000000000001
using CFTTruncation

# ╔═╡ f0000011-0003-0000-0000-000000000001
using Plots

# ╔═╡ f0000011-0004-0000-0000-000000000001
using Printf

# ╔═╡ f0000011-0010-0000-0000-000000000001
md"""
# 11 — SC Map Geometry Verification

Two complementary sanity checks for the Schwarz–Christoffel maps used
in this project:

## 1. **R_conv = 1** — the operational invariant

For each arm $i$, the local coordinate $f_i = \exp(-\pi \sigma_i^\ast / w_i \cdot f)$
maps a neighbourhood of the arm preimage $x_i$ into the upper unit
semidisk $D^+$ so that the arm infinity sits at $\xi_i = 0$ and the
**nearest corner(s) sit on the unit semicircle $|\xi_i| = 1$**. This
is the convergence radius of the Neumann series downstream, fixed by
the integration constant $\rho_0$ (see `local_coordinates_and_strip.md`
and the April 2026 session memo
`session_memo_rconv_fix.md`).

The visual test: for each arm, evaluate the library-built `f_series`
at the corner $\zeta$ and plot the result in the $\xi$-plane against
the unit circle. This is a **library end-to-end** check (uses
`compute_geometry` → `ArmData.f_series` directly) for the T-shape.
The cross analogue is deferred until the 4-arm pipeline is in place.

## 2. **Image under $f$** — the geometric sanity check

Plot the image of UHP horizontal lines $z = t + i\eta$ under $f(z)$ to
confirm the intended cross / T geometry. This is *not* library
end-to-end — the library never path-integrates $f'$; it uses local
Laurent expansions around each arm pole. But it's a useful picture.

For the cross this uses the library `fprime_exact_cross` directly.
For the T-shape it re-implements `fprime_exact` with a factored sqrt
branch (continuous in UHP, unlike the library's principal-of-product
choice which has a cut along the positive imaginary axis). The
library's choice is correct for its downstream pipeline (Laurent
expansion is local, never crosses branch cuts); it's only the
global-path visualisation that cares.
"""

# ╔═╡ f0000011-0020-0000-0000-000000000001
md"""
## Setup
"""

# ╔═╡ f0000011-0021-0000-0000-000000000001
ell_demo = 1.0

# ╔═╡ f0000011-0022-0000-0000-000000000001
series_order = 20

# ╔═╡ f0000011-0030-0000-0000-000000000001
md"""
## T-shape: R_conv = 1 check (library end-to-end)

Use `compute_geometry(ℓ, order)` to build the three arm data, then
evaluate each arm's `f_series` at the corner preimage(s) it sees.

- L arm ($x_L = -1$): one near corner at $z = -p$, local $\zeta = -p - (-1) = 1 - p$.
- R arm ($x_R = +1$): one near corner at $z = +p$, local $\zeta = p - 1$.
- T arm ($x_T = 0$): two corners at $z = \pm p$, local $\zeta = \pm p$.

For the library's $\rho_0$ convention (corners on the unit circle),
we expect $\xi_L(-p) = -1$, $\xi_R(+p) = +1$, $\xi_T(\pm p) = \pm 1$.
All on $|\xi| = 1$.
"""

# ╔═╡ f0000011-0031-0000-0000-000000000001
"""Evaluate a TruncLaurent series at a point ζ."""
function eval_series(s::CFTTruncation.TruncLaurent, ζ)
    result = zero(eltype(s.coeffs)) * one(complex(ζ))
    ζ_pow = complex(ζ)^s.val
    for c in s.coeffs
        result += c * ζ_pow
        ζ_pow *= ζ
    end
    result
end

# ╔═╡ f0000011-0032-0000-0000-000000000001
geom_T = CFTTruncation.compute_geometry(ell_demo, series_order)

# ╔═╡ f0000011-0033-0000-0000-000000000001
sc_T = geom_T.sc

# ╔═╡ f0000011-0034-0000-0000-000000000001
# For each arm, evaluate f_i at the adjacent corner ζ-values.
# Returns a vector of (arm_label, corner_z, ζ, ξ).
begin
    p = sc_T.p
    arm_corner_data_T = Tuple{Symbol, Float64, ComplexF64, ComplexF64}[]

    # R arm: near corner at z = +p, ζ = p - 1 (negative real).
    arm_R = geom_T.arms.R
    ζ_R = p - 1.0
    ξ_R = eval_series(arm_R.f_series, ζ_R)
    push!(arm_corner_data_T, (:R, +p, complex(ζ_R), ξ_R))

    # L arm: near corner at z = -p, ζ = 1 - p (positive real).
    arm_L = geom_T.arms.L
    ζ_L = 1.0 - p
    ξ_L = eval_series(arm_L.f_series, ζ_L)
    push!(arm_corner_data_T, (:L, -p, complex(ζ_L), ξ_L))

    # T arm: two corners at z = ±p, ζ = ±p.
    arm_T = geom_T.arms.T
    for corner_z in (+p, -p)
        ζ = corner_z - 0.0
        ξ = eval_series(arm_T.f_series, complex(ζ))
        push!(arm_corner_data_T, (:T, corner_z, complex(ζ), ξ))
    end

    arm_corner_data_T
end

# ╔═╡ f0000011-0035-0000-0000-000000000001
let
    lines = ["T-shape R_conv check (ℓ=$(ell_demo), series order=$(series_order)):"]
    push!(lines, @sprintf("  %-4s %-10s %-20s %-30s %-8s",
        "arm", "corner z", "ζ = z - x_arm", "ξ = f_i(ζ)", "|ξ|"))
    push!(lines, repeat("-", 80))
    for (arm, z, ζ, ξ) in arm_corner_data_T
        push!(lines, @sprintf("  %-4s %-10.5f (%+7.5f%+7.5fi) (%+10.5f%+10.5fi) %8.5f",
            string(arm), z, real(ζ), imag(ζ), real(ξ), imag(ξ), abs(ξ)))
    end
    Base.Text(join(lines, "\n"))
end

# ╔═╡ f0000011-0036-0000-0000-000000000001
# Plot: unit circle + corner positions in ξ-plane, per arm.
let
    plts = Plots.Plot[]
    for arm_name in (:L, :R, :T)
        p = plot(; aspect_ratio=:equal, xlims=(-1.3, 1.3), ylims=(-1.3, 1.3),
                  xlabel="Re ξ", ylabel="Im ξ",
                  title="ξ_$(arm_name) patch",
                  legend=:bottomright, legendfontsize=7)
        # Upper unit semicircle
        θ = range(0, π; length=200)
        plot!(p, cos.(θ), sin.(θ); color=:black, linewidth=1.5,
              label="|ξ|=1 (semicircle)")
        # Real axis segment (boundary of semidisk)
        plot!(p, [-1, 1], [0, 0]; color=:black, linewidth=1.5, label="")
        # Arm infinity at origin
        scatter!(p, [0], [0]; color=:blue, markersize=6, label="arm ∞ (ξ=0)")
        # Corner positions for this arm
        xs_c = Float64[]; ys_c = Float64[]
        for (arm_l, z, ζ, ξ) in arm_corner_data_T
            arm_l == arm_name || continue
            push!(xs_c, real(ξ)); push!(ys_c, imag(ξ))
        end
        scatter!(p, xs_c, ys_c; color=:red, markersize=7, label="corner(s) ξ_i(corner)")
        push!(plts, p)
    end
    plot(plts...; layout=(1, 3), size=(1200, 400))
end

# ╔═╡ f0000011-0040-0000-0000-000000000001
md"""
## Image plots: T-shape and cross

These use a *factored-branch* $f'(z)$ evaluated directly, integrated
along horizontal UHP lines to trace out the target geometry.

- For the T-shape: re-implemented locally below (library's
  `fprime_exact` uses a principal-of-product branch whose cut sits
  along the positive imaginary axis, which breaks this path
  integration; the library is *correct* for its Laurent-expansion
  downstream use, but wrong for us here).
- For the cross: uses the library `fprime_exact_cross` directly (the
  factored-branch sqrt I built into it last session).
"""

# ╔═╡ f0000011-0041-0000-0000-000000000001
"""T-shape f' with factored sqrt (for path-integration visualisation only)."""
function fprime_T_factored(z::Number, sc::CFTTruncation.SCParams)
    zc = complex(z)
    sq = sqrt(zc - sc.p) * sqrt(zc + sc.p)
    return sc.C * sq / (zc * (zc^2 - 1))
end

# ╔═╡ f0000011-0042-0000-0000-000000000001
function integrate_line(sc, z0, z1, fprime_func; n=50)
    s = 0.0 + 0.0im
    fp_prev = fprime_func(z0, sc)
    for i in 1:n
        z = z0 + i/n * (z1 - z0)
        fp = fprime_func(z, sc)
        dz = (z1 - z0) / n
        s += (fp_prev + fp) / 2 * dz
        fp_prev = fp
    end
    s
end

# ╔═╡ f0000011-0043-0000-0000-000000000001
function image_plot(sc, fprime_func, title_str, xlim, ylim)
    z_ref = 0.0 + 1.0im
    ts = vcat(collect(range(-12, -1.3; length=150)),
              collect(range(-1.3, -0.7; length=250)),
              collect(range(-0.7, -0.1; length=150)),
              collect(range(-0.1, 0.1; length=150)),
              collect(range(0.1, 0.7; length=150)),
              collect(range(0.7, 1.3; length=250)),
              collect(range(1.3, 12; length=150)))
    sort!(ts); unique!(ts)
    ηs = [0.002, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0]

    p = plot(; size=(600, 600), aspect_ratio=:equal,
               xlabel="Re f", ylabel="Im f", title=title_str,
               legend=:topleft, legendfontsize=7,
               xlims=xlim, ylims=ylim)
    colors = [:red, :cyan, :blue, :purple, :magenta, :orange, :green]
    for (η, col) in zip(ηs, colors)
        start = 0.0 + η*im
        f_start = integrate_line(sc, z_ref, start, fprime_func; n=500)
        fR = ComplexF64[]; f_cum = f_start; z_prev = start
        for t in ts[ts .> 0]
            z_cur = t + η*im
            f_cum += integrate_line(sc, z_prev, z_cur, fprime_func; n=10)
            push!(fR, f_cum); z_prev = z_cur
        end
        fL = ComplexF64[]; f_cum = f_start; z_prev = start
        for t in reverse(ts[ts .< 0])
            z_cur = t + η*im
            f_cum += integrate_line(sc, z_prev, z_cur, fprime_func; n=10)
            pushfirst!(fL, f_cum); z_prev = z_cur
        end
        all_f = vcat(fL, [f_start], fR)
        plot!(p, real.(all_f), imag.(all_f); color=col, linewidth=1.3, label="η=$η")
    end
    p
end

# ╔═╡ f0000011-0044-0000-0000-000000000001
image_plot(sc_T, fprime_T_factored, "T-shape (ℓ=$(ell_demo))", (-5, 5), (-1.5, 6))

# ╔═╡ f0000011-0045-0000-0000-000000000001
sc_C_demo = CFTTruncation.compute_sc_params_cross(ell_demo)

# ╔═╡ f0000011-0046-0000-0000-000000000001
image_plot(sc_C_demo, CFTTruncation.fprime_exact_cross, "Cross (ℓ=$(ell_demo))", (-3, 3), (-3, 3))

# ╔═╡ f0000011-0050-0000-0000-000000000001
md"""
## Cross SC parameters across ℓ

Closed-form from `src/SCMap.jl`:

$$q_1(\ell) = \frac{\sqrt{1+\ell^2} - 1}{\ell}, \quad q_2(\ell) = 1/q_1, \quad C(\ell) = \ell/\pi.$$
"""

# ╔═╡ f0000011-0051-0000-0000-000000000001
let
    lines = ["Cross SC parameters:"]
    push!(lines, @sprintf("  %-6s %-10s %-10s %-10s %-12s",
        "ℓ", "q_1", "q_2", "q_1·q_2", "C = ℓ/π"))
    push!(lines, repeat("-", 55))
    for ℓ in (0.1, 0.5, 1.0, 2.0, 5.0)
        sc = CFTTruncation.compute_sc_params_cross(ℓ)
        push!(lines, @sprintf("  %-6.2f %-10.5f %-10.5f %-10.5f %-12.5f",
            ℓ, sc.q1, sc.q2, sc.q1*sc.q2, sc.C))
    end
    Base.Text(join(lines, "\n"))
end

# ╔═╡ f0000011-0060-0000-0000-000000000001
md"""
## Cross at multiple ℓ (image plots)

Quick visual sweep showing how the cross geometry degenerates as
ℓ → 0 (horizontal strip) and ℓ → ∞ (vertical strip).
"""

# ╔═╡ f0000011-0061-0000-0000-000000000001
let
    plts = Plots.Plot[]
    for ℓ in (0.5, 1.0, 2.0)
        sc = CFTTruncation.compute_sc_params_cross(ℓ)
        lim = max(3.0, ℓ/2 + 1.5)
        push!(plts, image_plot(sc, CFTTruncation.fprime_exact_cross,
                                 "Cross ℓ=$ℓ", (-lim, lim), (-lim, lim)))
    end
    plot(plts...; layout=(1, 3), size=(1500, 500))
end

# ╔═╡ f0000011-0080-0000-0000-000000000001
md"""
## Cross R_conv = 1 check (library end-to-end)

The library now has `compute_geometry_cross(ℓ, order)` producing four
`ArmData` entries (L, R, T, B). Each arm's `f_series` is evaluated at
its adjacent corner ζ values; by the ρ₀ convention, the east-neighbour
corner lands at $\xi_i = +1$ on the unit semicircle. For T and B arms
(which see two adjacent corners within their convergence radius), the
west-neighbour corner lands at $\xi_i = -1$. For L and R arms, the
west corner is at distance > R_conv, so series evaluation diverges
there — only the east-corner check is within convergence.
"""

# ╔═╡ f0000011-0081-0000-0000-000000000001
geom_C = CFTTruncation.compute_geometry_cross(ell_demo, series_order)

# ╔═╡ f0000011-0082-0000-0000-000000000001
# Evaluate each cross arm's f_series at its adjacent (east / west) corners.
# Returns (arm_label, corner_label, ζ, ξ).
begin
    qc1 = geom_C.sc.q1
    cross_corner_data = Tuple{Symbol, String, ComplexF64, ComplexF64}[]
    # R arm: east at z=+q1 (ζ=q1-1), west at z=+q2 (ζ=q2-1, OUTSIDE R_conv).
    for (lbl, z_c) in [("east +q1", qc1), ("west +q2", geom_C.sc.q2)]
        ζ = complex(z_c - 1.0)
        ξ = eval_series(geom_C.arms.R.f_series, ζ)
        push!(cross_corner_data, (:R, lbl, ζ, ξ))
    end
    # L arm: east at z=-q1 (ζ=1-q1), west at z=-q2 (ζ=1-q2, OUTSIDE R_conv).
    for (lbl, z_c) in [("east -q1", -qc1), ("west -q2", -geom_C.sc.q2)]
        ζ = complex(z_c - (-1.0))
        ξ = eval_series(geom_C.arms.L.f_series, ζ)
        push!(cross_corner_data, (:L, lbl, ζ, ξ))
    end
    # T arm: east at z=+q1 (ζ=+q1), west at z=-q1 (ζ=-q1). Both within R_conv.
    for (lbl, ζ_val) in [("east +q1", +qc1), ("west -q1", -qc1)]
        ξ = eval_series(geom_C.arms.T.f_series, complex(ζ_val))
        push!(cross_corner_data, (:T, lbl, complex(ζ_val), ξ))
    end
    # B arm: local coord u = 1/z. East corner at u=1/q2=q1, west at u=-q1.
    for (lbl, u_val) in [("east u=+q1", +qc1), ("west u=-q1", -qc1)]
        ξ = eval_series(geom_C.arms.B.f_series, complex(u_val))
        push!(cross_corner_data, (:B, lbl, complex(u_val), ξ))
    end
    cross_corner_data
end

# ╔═╡ f0000011-0083-0000-0000-000000000001
let
    lines = ["Cross R_conv check (ℓ=$(ell_demo), order=$(series_order)):"]
    push!(lines, @sprintf("  %-4s %-12s %-32s %-10s",
        "arm", "corner", "ξ = f_i(ζ)", "|ξ|"))
    push!(lines, repeat("-", 70))
    for (arm, lbl, ζ, ξ) in cross_corner_data
        push!(lines, @sprintf("  %-4s %-12s (%+11.5e %+11.5ei) %10.4e",
            string(arm), lbl, real(ξ), imag(ξ), abs(ξ)))
    end
    Base.Text(join(lines, "\n"))
end

# ╔═╡ f0000011-0084-0000-0000-000000000001
# Plot: unit circle + corner positions in ξ-plane, per arm.
let
    plts = Plots.Plot[]
    for arm_name in (:L, :R, :T, :B)
        p = plot(; aspect_ratio=:equal, xlims=(-1.3, 1.3), ylims=(-1.3, 1.3),
                  xlabel="Re ξ", ylabel="Im ξ",
                  title="ξ_$(arm_name) patch (ℓ=$(ell_demo))",
                  legend=:bottomleft, legendfontsize=7)
        θ = range(0, π; length=200)
        plot!(p, cos.(θ), sin.(θ); color=:black, linewidth=1.5, label="|ξ|=1")
        plot!(p, [-1, 1], [0, 0]; color=:black, linewidth=1.5, label="")
        scatter!(p, [0], [0]; color=:blue, markersize=6, label="arm ∞ (ξ=0)")
        # Plot only the corners within convergence (magnitude < ~1.1).
        xs_c = Float64[]; ys_c = Float64[]; lbls_c = String[]
        for (arm_l, lbl, ζ, ξ) in cross_corner_data
            arm_l == arm_name || continue
            abs(ξ) > 2.0 && continue
            push!(xs_c, real(ξ)); push!(ys_c, imag(ξ)); push!(lbls_c, lbl)
        end
        scatter!(p, xs_c, ys_c; color=:red, markersize=7,
                 label="corner ξ_i")
        push!(plts, p)
    end
    plot(plts...; layout=(2, 2), size=(900, 900))
end

# ╔═╡ f0000011-0090-0000-0000-000000000001
md"""
## $g_i$ preimage in the $z$-plane: $g_i(|\xi|=1-\varepsilon)$

For each arm $i$, the inverse local coordinate $g_i$ maps a
neighbourhood of the origin in $\xi$-space back to a neighbourhood of
$x_i$ in $z$-space. The **preimage of the unit semicircle**
$\{|\xi|=1-\varepsilon\}$ under $g_i$ is a curve in $z$-space that
encloses $x_i$ and approaches the adjacent corners as $\varepsilon \to 0$
(since those corners sit at $|\xi| = 1$).

Below: for each ℓ, trace the preimages of $\xi = (1-\varepsilon) e^{i\theta}$
(upper semicircle, $\theta \in [0, \pi]$) for all arms, plot in the
$z$-plane, and mark the arm preimages $x_i$ and the corner preimages
$p_j$.

Computation is library end-to-end: directly sum the `g_series`
stored in `ArmData`, extend the result with $x_i$ for finite arms
(or $z = 1/u$ for the B arm in the cross).
"""

# ╔═╡ f0000011-0091-0000-0000-000000000001
"""Trace g_i on the upper half of |ξ|=r under the Pluto-safe assumption
   that the series has val=1 and converges for |ξ| ≤ 1. For the B arm
   (`is_B_arm=true`), apply z = 1/u after summing the series."""
function trace_gi_preimage(arm::CFTTruncation.ArmData; r=0.999, n=400, is_B_arm=false)
    θs = range(0, π; length=n)
    zs = ComplexF64[]
    for θ in θs
        ξ = r * cis(θ)
        # g_series(ξ) — series in ξ with val=1, gives z - x_i (or u - 0 for B).
        ζ = CFTTruncation.evaluate(arm.g_series, ξ)
        z = is_B_arm ? 1 / ζ : arm.x + ζ
        push!(zs, z)
    end
    zs
end

# ╔═╡ f0000011-0092-0000-0000-000000000001
"""T-shape g_i preimage plot for a given ℓ."""
function plot_T_preimages(ℓ; r=0.999, zlim=5.0)
    geom = CFTTruncation.compute_geometry(ℓ, series_order)
    p = plot(; aspect_ratio=:equal, xlims=(-zlim, zlim), ylims=(-0.5, zlim),
              xlabel="Re z", ylabel="Im z",
              title="T-shape: g_i(|ξ|=1-ε) preimages, ℓ=$ℓ",
              legend=:topright, legendfontsize=7)
    plot!(p, [-zlim, zlim], [0, 0]; color=:gray, linewidth=1, alpha=0.5, label="")
    colors = Dict(:L=>:blue, :R=>:red, :T=>:green)
    for name in (:L, :R, :T)
        arm = getfield(geom.arms, name)
        zs = trace_gi_preimage(arm; r=r)
        plot!(p, real.(zs), imag.(zs); color=colors[name], linewidth=1.8,
              label="g_$name")
    end
    scatter!(p, [geom.arms.L.x, geom.arms.T.x, geom.arms.R.x], [0, 0, 0];
             color=:black, markersize=8, marker=:xcross, label="x_i")
    scatter!(p, [-geom.sc.p, +geom.sc.p], [0, 0];
             color=:orange, markersize=7, marker=:diamond, label="corners ±p")
    p
end

# ╔═╡ f0000011-0093-0000-0000-000000000001
let
    plts = [plot_T_preimages(ℓ; zlim=4.0) for ℓ in (0.5, 1.0, 2.0)]
    plot(plts...; layout=(1, 3), size=(1500, 500))
end

# ╔═╡ f0000011-0094-0000-0000-000000000001
"""Cross g_i preimage plot for a given ℓ (all 4 arms, including B via u = 1/z)."""
function plot_cross_preimages(ℓ; r=0.999, zlim=5.0)
    geom = CFTTruncation.compute_geometry_cross(ℓ, series_order)
    p = plot(; aspect_ratio=:equal, xlims=(-zlim, zlim), ylims=(-0.5, zlim),
              xlabel="Re z", ylabel="Im z",
              title="Cross: g_i(|ξ|=1-ε) preimages, ℓ=$ℓ",
              legend=:topright, legendfontsize=7)
    plot!(p, [-zlim, zlim], [0, 0]; color=:gray, linewidth=1, alpha=0.5, label="")
    colors = Dict(:L=>:blue, :R=>:red, :T=>:green, :B=>:purple)
    for name in (:L, :R, :T, :B)
        arm = getfield(geom.arms, name)
        zs = trace_gi_preimage(arm; r=r, is_B_arm=(name == :B))
        # Skip points that wander outside the plot window (B arm preimage
        # at small |u| gives |z| → ∞).
        mask = abs.(zs) .< zlim * 2
        plot!(p, real.(zs[mask]), imag.(zs[mask]);
              color=colors[name], linewidth=1.8, label="g_$name")
    end
    # Mark finite arm preimages
    scatter!(p, [geom.arms.L.x, geom.arms.T.x, geom.arms.R.x], [0, 0, 0];
             color=:black, markersize=8, marker=:xcross, label="x_L, x_T, x_R")
    # B arm is at infinity — no marker to draw.
    # Corner preimages ±q1, ±q2
    q1 = geom.sc.q1; q2 = geom.sc.q2
    scatter!(p, [-q2, -q1, q1, q2], [0, 0, 0, 0];
             color=:orange, markersize=7, marker=:diamond, label="corners ±q1, ±q2")
    p
end

# ╔═╡ f0000011-0095-0000-0000-000000000001
let
    plts = [plot_cross_preimages(ℓ; zlim=4.0) for ℓ in (0.5, 1.0, 2.0)]
    plot(plts...; layout=(1, 3), size=(1500, 500))
end

# ╔═╡ f0000011-0070-0000-0000-000000000001
md"""
## Notes

1. The **R_conv checks** validate the library's full pipeline for both
   T-shape and cross: SC parameters → $f'$ Laurent → $\rho_0$ → $f_i$
   series → $\xi_i$ at corners. All adjacent corners should give
   $|\xi_i| \approx 1$ within ~1e-3 at order 30 (slow convergence
   at the series boundary — higher order tightens this).

2. The **$g_i$ preimage plots** are the clearest visual of the local
   coordinate patches: each curve encloses its arm preimage $x_i$
   and stretches out to the adjacent corner preimages $p_j$, which
   sit on $|\xi|=1$. Increasing $\varepsilon$ (shrinking $r$)
   retracts the curves toward $x_i$.

3. The **image plots** (earlier section) are *not* end-to-end: they
   use direct path integration of a re-branched $f'$. The library
   uses local Laurent expansions for its downstream pipeline and
   never path-integrates globally, so the principal-of-product
   branch is fine in-pipeline but breaks the viz.

4. **Cross downstream pipeline** (4×4 Neumann matrix, 4-arm
   recursion, primary vertex, cache shape tag) is the next step —
   see `truncation_strategies.md` §5–6. It'll need to handle
   $x_B = \infty$ (the B arm stores `x = Inf` and uses the local
   coordinate $u = 1/z$).
"""

# ╔═╡ Cell order:
# ╠═f0000011-0001-0000-0000-000000000001
# ╠═f0000011-0002-0000-0000-000000000001
# ╠═f0000011-0003-0000-0000-000000000001
# ╠═f0000011-0004-0000-0000-000000000001
# ╟─f0000011-0010-0000-0000-000000000001
# ╟─f0000011-0020-0000-0000-000000000001
# ╠═f0000011-0021-0000-0000-000000000001
# ╠═f0000011-0022-0000-0000-000000000001
# ╟─f0000011-0030-0000-0000-000000000001
# ╠═f0000011-0031-0000-0000-000000000001
# ╠═f0000011-0032-0000-0000-000000000001
# ╠═f0000011-0033-0000-0000-000000000001
# ╠═f0000011-0034-0000-0000-000000000001
# ╠═f0000011-0035-0000-0000-000000000001
# ╠═f0000011-0036-0000-0000-000000000001
# ╟─f0000011-0040-0000-0000-000000000001
# ╠═f0000011-0041-0000-0000-000000000001
# ╠═f0000011-0042-0000-0000-000000000001
# ╠═f0000011-0043-0000-0000-000000000001
# ╠═f0000011-0044-0000-0000-000000000001
# ╠═f0000011-0045-0000-0000-000000000001
# ╠═f0000011-0046-0000-0000-000000000001
# ╟─f0000011-0050-0000-0000-000000000001
# ╠═f0000011-0051-0000-0000-000000000001
# ╟─f0000011-0060-0000-0000-000000000001
# ╠═f0000011-0061-0000-0000-000000000001
# ╟─f0000011-0080-0000-0000-000000000001
# ╠═f0000011-0081-0000-0000-000000000001
# ╠═f0000011-0082-0000-0000-000000000001
# ╠═f0000011-0083-0000-0000-000000000001
# ╠═f0000011-0084-0000-0000-000000000001
# ╟─f0000011-0090-0000-0000-000000000001
# ╠═f0000011-0091-0000-0000-000000000001
# ╠═f0000011-0092-0000-0000-000000000001
# ╠═f0000011-0093-0000-0000-000000000001
# ╠═f0000011-0094-0000-0000-000000000001
# ╠═f0000011-0095-0000-0000-000000000001
# ╟─f0000011-0070-0000-0000-000000000001
