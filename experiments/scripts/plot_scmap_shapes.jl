using CFTTruncation: compute_sc_params, compute_sc_params_cross
using Plots

# Factored-branch sqrt for T-shape f'(z) = C √(z² - p²) / (z(z²-1)).
# √(z² - p²) = √(z - p) · √(z + p) with Julia's principal sqrt per factor.
# Each factor's cut is on R from its branch point to -∞; none cross UHP.
function fprime_T_factored(z::Number, sc)
    zc = complex(z)
    sq = sqrt(zc - sc.p) * sqrt(zc + sc.p)
    return sc.C * sq / (zc * (zc^2 - 1))
end

# Cross uses same pattern.
function fprime_C_factored(z::Number, sc)
    zc = complex(z)
    sq = sqrt(zc - sc.q1) * sqrt(zc + sc.q1) * sqrt(zc - sc.q2) * sqrt(zc + sc.q2)
    return sc.C * (-1im) * sq / ((zc^2 - 1) * zc)
end

function integrate_line(z0, z1, fprime_func, sc; n=50)
    s = 0.0 + 0.0im
    fp_prev = fprime_func(z0, sc)
    for i in 1:n
        z = z0 + i/n * (z1 - z0)
        fp = fprime_func(z, sc)
        dz = (z1 - z0) / n
        s += (fp_prev + fp)/2 * dz
        fp_prev = fp
    end
    s
end

function make_plot(sc, fprime_func, title_str, xlim, ylim)
    z_ref = 0.0 + 1.0im
    ts = vcat(collect(range(-8, -1.3; length=100)),
              collect(range(-1.3, -0.7; length=200)),
              collect(range(-0.7, -0.1; length=100)),
              collect(range(-0.1, 0.1; length=100)),
              collect(range(0.1, 0.7; length=100)),
              collect(range(0.7, 1.3; length=200)),
              collect(range(1.3, 8; length=100)))
    sort!(ts); unique!(ts)
    ηs = [0.01, 0.05, 0.1, 0.3, 1.0, 3.0]

    p = plot(size=(600, 600), aspect_ratio=:equal,
             xlabel="Re f", ylabel="Im f", title=title_str,
             legend=:topleft, legendfontsize=7,
             xlims=xlim, ylims=ylim)

    colors = [:cyan, :blue, :purple, :magenta, :orange, :green]
    for (η, col) in zip(ηs, colors)
        start = 0.0 + η*im
        f_start = integrate_line(z_ref, start, fprime_func, sc; n=500)
        f_cum = f_start; z_prev = start
        fvals_R = ComplexF64[]
        for t in ts[ts .> 0]
            z_cur = t + η*im
            f_cum += integrate_line(z_prev, z_cur, fprime_func, sc; n=10)
            push!(fvals_R, f_cum); z_prev = z_cur
        end
        f_cum = f_start; z_prev = start
        fvals_L = ComplexF64[]
        for t in reverse(ts[ts .< 0])
            z_cur = t + η*im
            f_cum += integrate_line(z_prev, z_cur, fprime_func, sc; n=10)
            pushfirst!(fvals_L, f_cum); z_prev = z_cur
        end
        all_f = vcat(fvals_L, [f_start], fvals_R)
        plot!(p, real.(all_f), imag.(all_f); color=col, linewidth=1.3, label="η=$η")
    end
    p
end

sc_T = compute_sc_params(1.0)
p1 = make_plot(sc_T, fprime_T_factored, "T-shape (ℓ=1)", (-5, 5), (-1.5, 6))

sc_C = compute_sc_params_cross(1.0)
p2 = make_plot(sc_C, fprime_C_factored, "Cross (ℓ=1)", (-3, 3), (-3, 3))

plt = plot(p1, p2, layout=(1, 2), size=(1200, 600))
savefig(plt, "/Users/kantaro/repositories/cft-truncation/docs/design/figures/cross_geometry_check.png")
println("Saved.")
