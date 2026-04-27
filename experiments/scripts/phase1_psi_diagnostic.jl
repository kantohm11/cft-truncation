#!/usr/bin/env julia
# Phase-1 |ψ⟩ diagnostic: compute the per-site reduced density matrix
# ρ_phys of the charge-0-projected uniform T-MPS, look at its Schmidt
# spectrum, and decompose its dominant eigenvector |ψ⟩ over the
# charge-0 phys-basis. Also re-extract c_est at moderate-to-large ℓ.

using CFTTruncation
using TensorKit: fusiontrees
using LinearAlgebra
using Printf

const R_VAL    = 1.0
const H_BOND   = 6.0
const H_PHYS   = 3.0   # d_T = 7 in charge-0 sector

# ---------------------------------------------------------------- helpers

"""Charge-0 block A[s, αL, αR] of V_ℓ. Mirrors notebook 10's extract_W."""
function extract_A(vd::CFTTruncation.VertexData)
    V = vd.vertex
    bond = vd.cft.basis_bond
    phys = vd.cft.basis_phys
    D = length(bond.states[0])
    d_T = length(phys.states[0])
    A = zeros(Float64, d_T, D, D)
    for (f1, f2) in fusiontrees(V)
        all(Int(f2.uncoupled[i].charge) == 0 for i in 1:3) || continue
        A .+= V[f1, f2]
    end
    A
end

"""Transfer matrix E[(αL, αL'), (αR, αR')] = Σ_s A[s, αL, αR]·A[s, αL', αR']."""
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

"""Largest-eigenvalue right eigenvector of M (real-valued, post-symmetrize)."""
function dominant_right(M::Matrix{Float64})
    F = eigen(M)
    idx = argmax(abs.(F.values))
    λ = real(F.values[idx])
    v = real.(F.vectors[:, idx])
    λ, v
end

function dominant_left(M::Matrix{Float64})
    λ, v = dominant_right(Matrix(transpose(M)))
    λ, v
end

"""ρ_phys[s, s'] from dominant left/right E eigenvectors."""
function rho_phys(A::Array{Float64,3}, E::Matrix{Float64})
    d_T, D, _ = size(A)
    λ_R, v_R = dominant_right(E)
    λ_L, v_L = dominant_left(E)
    @assert isapprox(λ_R, λ_L; rtol=1e-8) "left/right eigvals disagree"

    # Reshape v_R, v_L from D² → (D, D).
    VR = reshape(v_R, D, D)        # VR[αR, αR']
    VL = reshape(v_L, D, D)        # VL[αL, αL']

    # Normalise: ⟨v_L | v_R⟩ in the bond Hilbert space (i.e. tr(VL · VR)).
    norm_LR = tr(VL * VR)
    VR ./= sqrt(abs(norm_LR))
    VL ./= sqrt(abs(norm_LR))

    # ρ[s, s'] = Σ VL[αL, αL'] · A[s, αL, αR] · A[s', αL', αR'] · VR[αR, αR'] / λ
    ρ = zeros(Float64, d_T, d_T)
    for s in 1:d_T, sp in 1:d_T
        # tr(VL · A[s] · VR · A[s']ᵀ) / λ
        ρ[s, sp] = tr(VL * A[s, :, :] * VR * transpose(A[sp, :, :])) / λ_R
    end
    # Hermitise (small numerical asymmetry).
    ρ = (ρ + transpose(ρ)) / 2
    ρ, λ_R
end

"""Dominant eigenvalue/vector of ρ_phys."""
function dominant_psi(ρ::Matrix{Float64})
    F = eigen(Symmetric(ρ))
    # eigenvalues sorted ascending; we want largest.
    μ = reverse(real.(F.values))
    V = F.vectors[:, end:-1:1]
    μ, V
end

"""Format the level/partition labels from basis_phys.states[0]."""
function fock_labels(phys::CFTTruncation.FockBasis, n::Int=0)
    states = phys.states[n]
    [@sprintf("[%s]", join(λ, ",")) for λ in states]
end

# ---------------------------------------------------------------- run

function run_at(ell::Float64)
    cft = CompactBosonCFT(R=R_VAL, h_bond=H_BOND, h_phys=H_PHYS)
    vd = compute_vertex(cft, ell; cache=:off, series_order=20)
    A = extract_A(vd)
    E = build_E(A)
    ρ, λ = rho_phys(A, E)
    μ, V = dominant_psi(ρ)
    ψ = V[:, 1]                         # normalized eigenvec
    vac_overlap = ψ[1]^2                # state s=1 is |0⟩ (empty partition first)
    (; ell, A_size=size(A), λ_E=λ,
       schmidt=μ[1:min(6,end)],
       psi=ψ,
       vac_overlap,
       phys_labels=fock_labels(vd.cft.basis_phys, 0))
end

println("=== |ψ⟩ diagnostic on charge-0 T-MPS at h_bond=$(H_BOND), h_phys=$(H_PHYS) ===\n")

ells = [0.01, 0.1, 0.5, 1.0, 2.0, 3.0]
results = [run_at(ell) for ell in ells]

# Summary table.
println(@sprintf("%6s | %14s %14s | %14s | %14s | %s",
    "ℓ", "λ_E", "1−λ_E", "Schmidt μ_1", "1−μ_1", "|⟨0|ψ⟩|²"))
println("-"^110)
for r in results
    println(@sprintf("%6.2f | %14.6e %14.3e | %14.6e | %14.3e | %.6f",
        r.ell, r.λ_E, 1 - r.λ_E, r.schmidt[1], 1 - r.schmidt[1], r.vac_overlap))
end

# Schmidt spectrum table.
println("\nSchmidt spectrum μ_k of ρ_phys:")
println(@sprintf("%6s | %s", "ℓ", join([@sprintf("%14s", "μ_$k") for k in 1:6], "")))
println("-"^100)
for r in results
    println(@sprintf("%6.2f | %s", r.ell,
        join([@sprintf("%14.6e", r.schmidt[k]) for k in 1:length(r.schmidt)], "")))
end

# |ψ⟩ decomposition over Fock basis.
println("\n|ψ⟩ Fock-basis composition (top components per ℓ):")
labels = results[1].phys_labels
for r in results
    println(@sprintf("\nℓ = %.2f", r.ell))
    coefs = [(abs(r.psi[k])^2, k) for k in 1:length(r.psi)]
    sort!(coefs; rev=true)
    for (w, k) in coefs[1:min(5, end)]
        w < 1e-10 && continue
        println(@sprintf("  |ψ⟩ has |c_%d|² = %.6e on %s", k, w, labels[k]))
    end
end
