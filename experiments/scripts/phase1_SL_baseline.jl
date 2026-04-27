#!/usr/bin/env julia
# Qualitative baseline: T-MPS finite-N S(L) at moderate ℓ.
#
# Even though the constraint analysis predicts c_est ≪ 1 (window for
# clean c log L is empty — see memory/project_t_mps_window_empty.md),
# we still want to confirm the EE is non-trivial and qualitatively
# CFT-shaped before reaching for the cross MPO.
#
# Sanity flags per ℓ:
#   1. S(L; N) > 0 across the (L, N) grid.
#   2. S monotone in L at fixed N.
#   3. S monotone in N at fixed L.

using CFTTruncation
using TensorKit: fusiontrees
using LinearAlgebra
using Plots
using Printf

const R_VAL  = 1.0
const H_PHYS = 2.0
const H_BOND = 4.0
const ELLS   = [0.1, 0.5, 1.0]
const NS     = [8, 10]
const LMAX   = 4
const FIG_DIR = "/Users/kantaro/repositories/cft-truncation/docs/design/figures"

# ---------------------------------------------------------------- helpers

"""Pack the full vertex tensor into a flat A[s, αL, αR] indexed by
flattened (charge, level-state) pairs on each leg. Includes all
charge-conserving (n_T + n_L + n_R = 0) blocks. Charge ≠ 0 entries
that the recursion produces are now included; this is "option 2"."""
function extract_W(vd::CFTTruncation.VertexData)
    V = vd.vertex
    bond = vd.cft.basis_bond
    phys = vd.cft.basis_phys

    bond_charges = sort(collect(keys(bond.states)))
    phys_charges = sort(collect(keys(phys.states)))

    bond_off = Dict{Int,Int}(); cur = 0
    for n in bond_charges
        bond_off[n] = cur
        cur += length(bond.states[n])
    end
    D = cur

    phys_off = Dict{Int,Int}(); cur = 0
    for n in phys_charges
        phys_off[n] = cur
        cur += length(phys.states[n])
    end
    d_T = cur

    A = zeros(Float64, d_T, D, D)
    for (f1, f2) in fusiontrees(V)
        n_T = Int(f2.uncoupled[1].charge)
        n_L = Int(f2.uncoupled[2].charge)
        n_R = Int(f2.uncoupled[3].charge)
        haskey(phys.states, n_T) && haskey(bond.states, n_L) &&
            haskey(bond.states, n_R) || continue
        blk = V[f1, f2]
        oT = phys_off[n_T]; oL = bond_off[n_L]; oR = bond_off[n_R]
        for αT in 1:size(blk, 1), αL in 1:size(blk, 2), αR in 1:size(blk, 3)
            A[oT + αT, oL + αL, oR + αR] += blk[αT, αL, αR]
        end
    end
    A, D, d_T
end

function build_E(A::Array{Float64,3})
    d, D, _ = size(A)
    E = zeros(Float64, D*D, D*D)
    @inbounds for αL in 1:D, αLp in 1:D, αR in 1:D, αRp in 1:D
        s = 0.0
        for sT in 1:d
            s += A[sT, αL, αR] * A[sT, αLp, αRp]
        end
        E[αL + (αLp-1)*D, αR + (αRp-1)*D] = s
    end
    E
end

function rho_L_PBC(A::Array{Float64,3}, L::Int, N::Int)
    d, D, _ = size(A)
    @assert L ≥ 1 && L < N
    E = build_E(A)
    E_NmL = E^(N - L)
    Z_N = real(tr(E^N))
    Ndim = d^L
    M = zeros(Float64, Ndim, D, D)
    idx = zeros(Int, L)
    for s_flat in 1:Ndim
        k = s_flat - 1
        for l in 1:L
            idx[l] = (k % d) + 1
            k ÷= d
        end
        mat = Matrix{Float64}(I, D, D)
        for l in 1:L
            mat = mat * A[idx[l], :, :]
        end
        M[s_flat, :, :] = mat
    end
    Et = reshape(E_NmL, D, D, D, D)
    Etp = permutedims(Et, (3, 1, 2, 4))
    Mflat = reshape(M, Ndim, D * D)
    Etpflat = reshape(Etp, D * D, D * D)
    X = Mflat * Etpflat
    X3 = reshape(X, Ndim, D, D)
    Xperm = permutedims(X3, (1, 3, 2))
    ρ = reshape(Xperm, Ndim, D * D) * reshape(M, Ndim, D * D)'
    ρ ./= Z_N
    ρ
end

function ee_from_rho(ρ::Matrix{Float64})
    vals = eigvals(Hermitian((ρ + ρ') / 2))
    vals = sort(real.(vals); rev=true)
    trace = sum(vals)
    trace ≤ 0 && return NaN
    p = vals ./ trace
    p = p[p .> 1e-14]
    -sum(p .* log.(p))
end

cft_x(L::Int, N::Int) = log((N / π) * sin(π * L / N))

function fit_slope(results, Ns, Lmax)
    out = Tuple{Int, Float64}[]
    for N in Ns
        xs = Float64[]; ys = Float64[]
        for L in 1:min(Lmax, N - 1)
            S = get(results, (L, N), NaN)
            isnan(S) && continue
            push!(xs, cft_x(L, N)); push!(ys, S)
        end
        length(xs) < 2 && continue
        x̄, ȳ = sum(xs)/length(xs), sum(ys)/length(ys)
        slope = sum((xs .- x̄) .* (ys .- ȳ)) / sum((xs .- x̄).^2)
        push!(out, (N, slope))
    end
    out
end

# ---------------------------------------------------------------- sweep

function sweep_at(ell)
    cft = CompactBosonCFT(R = R_VAL, h_bond = H_BOND, h_phys = H_PHYS)
    vd  = compute_vertex(cft, ell; cache = :off, series_order = 20)
    A, D, d_T = extract_W(vd)
    results = Dict{Tuple{Int,Int}, Float64}()
    for N in NS, L in 1:min(LMAX, N - 1)
        results[(L, N)] = ee_from_rho(rho_L_PBC(A, L, N))
    end
    (; ell, D, d_T, results)
end

# Sanity flags.
function sanity(results, ns, lmax)
    flag_pos = true; flag_mono_L = true; flag_mono_N = true
    for N in ns, L in 1:min(lmax, N-1)
        S = results[(L, N)]
        S ≤ 0 && (flag_pos = false)
    end
    for N in ns
        prev = -Inf
        for L in 1:min(lmax, N-1)
            S = results[(L, N)]
            S < prev - 1e-12 && (flag_mono_L = false)
            prev = S
        end
    end
    for L in 1:lmax
        prev = -Inf
        for N in ns
            L < N || continue
            S = results[(L, N)]
            S < prev - 1e-12 && (flag_mono_N = false)
            prev = S
        end
    end
    (flag_pos, flag_mono_L, flag_mono_N)
end

# ---------------------------------------------------------------- plots

function plot_ell(sw, ell)
    p1 = plot(; xlabel = "L", ylabel = "S(L; N)",
              title = "ℓ = $ell — S vs L", legend = :topleft,
              size = (450, 360))
    for N in NS
        xs = Int[]; ys = Float64[]
        for L in 1:min(LMAX, N - 1)
            push!(xs, L); push!(ys, sw.results[(L, N)])
        end
        plot!(p1, xs, ys; marker = :o, label = "N=$N")
    end

    p2 = plot(; xlabel = "log[(N/π) sin(πL/N)]", ylabel = "S(L; N)",
              title = "ℓ = $ell — CFT axis", legend = :topleft,
              size = (450, 360))
    for N in NS
        xs = Float64[]; ys = Float64[]
        for L in 1:min(LMAX, N - 1)
            push!(xs, cft_x(L, N)); push!(ys, sw.results[(L, N)])
        end
        plot!(p2, xs, ys; marker = :o, label = "N=$N")
    end
    # Reference slope-1/3 line anchored to (L=2, N=12) if available.
    anchor = get(sw.results, (2, 12), nothing)
    if anchor !== nothing
        x0 = cft_x(2, 12); y0 = anchor
        xr = [-2.0, 1.5]
        plot!(p2, xr, y0 .+ (1/3) .* (xr .- x0);
              linestyle = :dash, color = :black, label = "slope 1/3 (c=1)")
    end

    plot(p1, p2; layout = (1, 2), size = (900, 360))
end

function summary_panel(sweeps)
    p = plot(; xlabel = "N (log scale)", ylabel = "S(L=Lmax; N)",
             title = "S(L=$LMAX; N) vs N for each ℓ",
             legend = :topleft, size = (560, 380), xscale = :log10)
    for sw in sweeps
        xs = Int[]; ys = Float64[]
        for N in NS
            L = min(LMAX, N - 1)
            haskey(sw.results, (L, N)) || continue
            push!(xs, N); push!(ys, sw.results[(L, N)])
        end
        plot!(p, xs, ys; marker = :o, label = "ℓ=$(sw.ell)", linewidth = 2)
    end
    p
end

# ---------------------------------------------------------------- run

println("=== T-MPS EE qualitative baseline (option 2: all charges) ===")
println("h_bond=$H_BOND, h_phys=$H_PHYS, R=$R_VAL, Ns=$NS, Lmax=$LMAX")
println()

mkpath(FIG_DIR)
sweeps = []
for ell in ELLS
    println("ℓ = $ell")
    sw = sweep_at(ell)
    push!(sweeps, sw)
    println("  D=$(sw.D), d_T=$(sw.d_T)")

    # Sanity.
    flag_pos, flag_mono_L, flag_mono_N = sanity(sw.results, NS, LMAX)
    println("  flags  S>0=$flag_pos, mono in L=$flag_mono_L, mono in N=$flag_mono_N")

    Smin = minimum(values(sw.results))
    Smax = maximum(values(sw.results))
    println(@sprintf("  S range  [%.4e, %.4e]", Smin, Smax))

    println("  c_est = 3 · slope(S vs cft_x), per N:")
    for (N, slope) in fit_slope(sw.results, NS, LMAX)
        println(@sprintf("    N=%-3d  c_est=%.4e", N, 3 * slope))
    end

    # Plot.
    plt = plot_ell(sw, ell)
    figpath = joinpath(FIG_DIR, "T_mps_ee_baseline_ell$(ell).png")
    savefig(plt, figpath)
    println("  saved $figpath")
    println()
end

# Summary panel.
plt = summary_panel(sweeps)
sumpath = joinpath(FIG_DIR, "T_mps_ee_baseline_summary.png")
savefig(plt, sumpath)
println("Summary plot saved: $sumpath")
