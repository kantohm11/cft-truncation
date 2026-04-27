#!/usr/bin/env julia
# Re-scan c_est for the charge-0 T-MPS at moderate-to-large ℓ
# (the previous Phase-1 memo only looked at ℓ ∈ {0.01, 0.1}).
# Mirrors notebook 10's run_sweep + fit_slope_per_N pipeline.

using CFTTruncation
using TensorKit: fusiontrees
using LinearAlgebra
using Printf

const R_VAL  = 1.0
const H_PHYS = 3.0    # d_T = 7
const H_BOND = 6.0
const Ns     = [6, 8, 10, 12, 16]
const Lmax   = 4

# Reuse the helpers from notebook 10 (re-coded inline for this script).
function extract_W(vd::CFTTruncation.VertexData)
    V = vd.vertex
    bond = vd.cft.basis_bond
    phys = vd.cft.basis_phys
    D = length(bond.states[0])
    d_T = length(phys.states[0])
    W = zeros(Float64, d_T, D, D)
    for (f1, f2) in fusiontrees(V)
        all(Int(f2.uncoupled[i].charge) == 0 for i in 1:3) || continue
        W .+= V[f1, f2]
    end
    W, D, d_T
end

function build_E(A::Array{Float64,3})
    d_T, D, _ = size(A)
    E = zeros(Float64, D*D, D*D)
    @inbounds for αL in 1:D, αLp in 1:D, αR in 1:D, αRp in 1:D
        s = 0.0
        for s_idx in 1:d_T
            s += A[s_idx, αL, αR] * A[s_idx, αLp, αRp]
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

function fit_slope(results::Dict{Tuple{Int,Int}, Float64}, Ns, Lmax)
    rows = Tuple{Int, Float64, Float64}[]
    for N in Ns
        xs = Float64[]; ys = Float64[]
        for L in 1:min(Lmax, N - 1)
            S = get(results, (L, N), NaN)
            isnan(S) && continue
            push!(xs, cft_x(L, N))
            push!(ys, S)
        end
        length(xs) < 2 && continue
        x̄, ȳ = sum(xs)/length(xs), sum(ys)/length(ys)
        slope = sum((xs .- x̄) .* (ys .- ȳ)) / sum((xs .- x̄).^2)
        intercept = ȳ - slope * x̄
        push!(rows, (N, slope, intercept))
    end
    rows
end

function sweep_at(ell::Float64)
    cft = CompactBosonCFT(R=R_VAL, h_bond=H_BOND, h_phys=H_PHYS)
    vd = compute_vertex(cft, ell; cache=:off, series_order=20)
    A, D, d_T = extract_W(vd)
    results = Dict{Tuple{Int,Int}, Float64}()
    for N in Ns, L in 1:min(Lmax, N - 1)
        ρ = rho_L_PBC(A, L, N)
        S = ee_from_rho(ρ)
        results[(L, N)] = S
    end
    (; ell, D, d_T, results)
end

# ---------------------------------------------------------------- run

ells = [0.01, 0.1, 0.5, 1.0, 2.0, 3.0]
println("=== c_est scan: c = 3 × slope(S vs cft_x) per N ===")
println("ℓ scan: $(ells); h_bond=$(H_BOND), h_phys=$(H_PHYS)")
println()

println(@sprintf("%6s | %8s | %s", "ℓ", "S(N/2,N)", join([@sprintf("%-22s", "N=$N: c_est") for N in Ns], "")))
println("-"^140)

for ell in ells
    sw = sweep_at(ell)
    s_half = haskey(sw.results, (Lmax, Ns[end])) ? sw.results[(Lmax, Ns[end])] : NaN
    slopes = fit_slope(sw.results, Ns, Lmax)
    c_per_N = Dict(N => 3*slope for (N, slope, _) in slopes)
    cest_strs = [@sprintf("%-22s", get(c_per_N, N, NaN)) for N in Ns]
    println(@sprintf("%6.2f | %8.5f | %s", ell, s_half,
        join(cest_strs, "")))
end
