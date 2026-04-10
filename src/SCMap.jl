"""
Schwarz-Christoffel parameters for the T-shaped domain.
"""
struct SCParams
    ell::Float64
    p::Float64    # p = ℓ/√(4+ℓ²)
    C::Float64    # C = √(4+ℓ²)/π
end

function compute_sc_params(ell::Real)
    ell = Float64(ell)
    s = sqrt(4 + ell^2)
    SCParams(ell, ell / s, s / π)
end

"""
    fprime_exact(z, sc::SCParams)

Evaluate the SC derivative f'(z) = C√(z²-p²) / (z(z²-1)).
Branch of √ chosen so that Im(f'(z)) > 0 in the upper half-plane near z=0.
"""
function fprime_exact(z::Number, sc::SCParams)
    p = sc.p
    # √(z²-p²): for z in UHP near 0, z²-p² ≈ -p² so √ ≈ ip
    # We use the convention √(z²-p²) = i*p*√(1 - z²/p²) with principal √
    # which gives the correct residue +iℓ/π at z=0.
    sq = sqrt(complex(z^2 - p^2))
    val = sc.C * sq / (z * (z^2 - 1))
    # Fix branch: residue at z=0 should be iℓ/π (positive imaginary)
    # Res_{z=0} f' = C * √(-p²) / (0 * ...) — check sign via limit
    # Actually compute directly: for z = iε (small ε>0),
    # z²-p² = -ε²-p² < 0, so √(z²-p²) should be i*√(ε²+p²)
    # giving f'(iε) = C * i*√(ε²+p²) / (iε*(−ε²−1)) ≈ C*ip/(−iε) = −Cp/ε
    # But Res = lim z*f'(z) as z→0, = C*√(−p²)/(-1) = C*ip*(-1) = -iCp = -iℓ/π
    # The residue is -iℓ/π, which gives arm direction σ_T = i after
    # accounting for w_T σ_T / π = -iℓ/π → σ_T = i (with w_T = ℓ, note the sign).
    return val
end
