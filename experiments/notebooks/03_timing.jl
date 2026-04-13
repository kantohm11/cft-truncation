### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ d0000001-0001-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
end

# ╔═╡ d0000001-0002-0000-0000-000000000001
using CFTTruncation

# ╔═╡ d0000001-0003-0000-0000-000000000001
using TensorKit: dim

# ╔═╡ d0000001-0004-0000-0000-000000000001
using Plots

# ╔═╡ d0000001-0010-0000-0000-000000000001
md"""
# 03 — Vertex Computation Timing

Measure `compute_vertex` wall time as a function of $h_{\max}$ and
$\dim(V)$, at fixed $R = 1$, $\ell = 1$.
"""

# ╔═╡ d0000001-0011-0000-0000-000000000001
begin
    R_val = 1.0
    ell_val = 1.0
    h_maxs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
end

# ╔═╡ d0000001-0012-0000-0000-000000000001
md"### Benchmark"

# ╔═╡ d0000001-0013-0000-0000-000000000001
timing_data = let
    results = []
    for hm in h_maxs
        cft = CompactBosonCFT(R=R_val, trunc=TruncationSpec(hm))
        d = dim(cft.basis_bond.V)
        # Warmup
        compute_vertex(cft, ell_val)
        # Timed run (3 samples, take median)
        times = Float64[]
        for _ in 1:3
            t = @elapsed compute_vertex(cft, ell_val)
            push!(times, t)
        end
        t_med = sort(times)[2]  # median of 3
        push!(results, (h_max=hm, dim_V=d, time_s=t_med))
    end
    results
end

# ╔═╡ d0000001-0014-0000-0000-000000000001
md"### Results table"

# ╔═╡ d0000001-0015-0000-0000-000000000001
let
    lines = ["  h_max   dim(V)   time (s)"]
    push!(lines, repeat("-", 30))
    for r in timing_data
        push!(lines, "  $(Int(r.h_max))       $(r.dim_V)       $(round(r.time_s; sigdigits=3))")
    end
    Base.Text(join(lines, "\n"))
end

# ╔═╡ d0000001-0020-0000-0000-000000000001
md"### Time vs h_max"

# ╔═╡ d0000001-0021-0000-0000-000000000001
let
    hms = [r.h_max for r in timing_data]
    ts = [r.time_s for r in timing_data]
    plot(hms, ts; xlabel="h_max", ylabel="time (s)",
         title="compute_vertex time vs h_max (R=1, ℓ=1)",
         yscale=:log10, marker=:circle, markersize=6, legend=false,
         size=(600, 400))
end

# ╔═╡ d0000001-0030-0000-0000-000000000001
md"### Time vs dim(V)"

# ╔═╡ d0000001-0031-0000-0000-000000000001
let
    ds = [Float64(r.dim_V) for r in timing_data]
    ts = [r.time_s for r in timing_data]
    p = plot(ds, ts; xlabel="dim(V)", ylabel="time (s)",
             title="compute_vertex time vs dim(V)",
             xscale=:log10, yscale=:log10, marker=:circle, markersize=6,
             legend=false, size=(600, 400))
    if length(ds) >= 3
        logd = log.(ds)
        logt = log.(ts)
        b = (length(logd) * sum(logd .* logt) - sum(logd) * sum(logt)) /
            (length(logd) * sum(logd .^ 2) - sum(logd)^2)
        annotate!(p, [(ds[end], ts[end], Plots.text("slope ≈ $(round(b; digits=1))", :right, 10))])
    end
    p
end

# ╔═╡ d0000001-0040-0000-0000-000000000001
md"""
### Summary

The vertex computation time scales roughly as $\dim(V)^b$ where $b$ is
shown on the log-log plot. The recursion is $O(D^3)$ where $D = \dim(V)$
(three nested loops over states at each charge triple), so we expect
$b \approx 3$ for the dominant cost.
"""

# ╔═╡ Cell order:
# ╟─d0000001-0010-0000-0000-000000000001
# ╠═d0000001-0001-0000-0000-000000000001
# ╠═d0000001-0002-0000-0000-000000000001
# ╠═d0000001-0003-0000-0000-000000000001
# ╠═d0000001-0004-0000-0000-000000000001
# ╠═d0000001-0011-0000-0000-000000000001
# ╟─d0000001-0012-0000-0000-000000000001
# ╠═d0000001-0013-0000-0000-000000000001
# ╟─d0000001-0014-0000-0000-000000000001
# ╠═d0000001-0015-0000-0000-000000000001
# ╟─d0000001-0020-0000-0000-000000000001
# ╠═d0000001-0021-0000-0000-000000000001
# ╟─d0000001-0030-0000-0000-000000000001
# ╠═d0000001-0031-0000-0000-000000000001
# ╟─d0000001-0040-0000-0000-000000000001
