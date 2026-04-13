#!/usr/bin/env julia
using CFTTruncation, TensorKit

for hm in [6.0, 8.0]
    println("=== h_max = $(Int(hm)) ===")
    cft = CompactBosonCFT(R=1.0, trunc=TruncationSpec(hm))
    println("dim(V) = ", dim(cft.basis_bond.V))

    # Warmup
    compute_vertex(cft, 1.0)

    # Timed (median of 3)
    times = Float64[]
    for _ in 1:3
        t = @elapsed compute_vertex(cft, 1.0)
        push!(times, t)
    end
    t_med = sort(times)[2]
    println("  compute_vertex: $(round(t_med; sigdigits=3))s (median of 3)")

    # Sub-step breakdown
    geom = CFTTruncation.compute_geometry(1.0, 20)
    neum = CFTTruncation.compute_neumann(geom, cft.m_max)
    times_raw = Float64[]
    for _ in 1:3
        t = @elapsed CFTTruncation._compute_vertex_raw(
            cft.basis_bond, cft.basis_phys, geom, neum, cft.J_bond, cft.J_phys, cft.R)
        push!(times_raw, t)
    end
    println("  Ward recursion: $(round(sort(times_raw)[2]; sigdigits=3))s")
    println()
end
