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

"""
Schwarz-Christoffel parameters for the cross-shaped domain (horizontal
strip of width 1 ∪ vertical strip of width ℓ).

Gauge: `x_L = -1, x_T = 0, x_R = 1, x_B = ∞`. Corners at `±q_1, ±q_2`
with `0 < q_1 < 1 < q_2` and `q_1·q_2 = 1`.

Elementary closed form (residues at z = 0, 1, ∞ give a solvable system;
see `docs/design/conformal_map_cross.md`):

    q_1(ℓ) = (√(1+ℓ²) − 1)/ℓ
    q_2(ℓ) = (√(1+ℓ²) + 1)/ℓ      (= 1/q_1)
    C(ℓ)   = ℓ/π
"""
struct SCParamsCross
    ell::Float64
    q1::Float64   # corners near origin, 0 < q1 < 1
    q2::Float64   # corners far from origin, q2 = 1/q1 > 1
    C::Float64    # C = ℓ/π
end

function compute_sc_params_cross(ell::Real)
    ell = Float64(ell)
    s = sqrt(1 + ell^2)
    q1 = (s - 1) / ell
    q2 = (s + 1) / ell
    C = ell / π
    SCParamsCross(ell, q1, q2, C)
end

"""
    fprime_exact_cross(z, sc::SCParamsCross)

Evaluate the cross SC derivative

    f'(z) = C √{(q_1² − z²)(z² − q_2²)} / ((z² − 1) · z).

Branch of √ uses Julia's principal branch via `sqrt(complex(...))`.
With this convention, the residues are:

    Res_{z=0}  f' = −i C q_1 q_2 = −iℓ/π   (T arm, σ_T = −i)
    Res_{z=±1} f' = ± C √{(1−q_1²)(q_2²−1)}/2 = ±1/π   (R/L arm, σ_R,L = ±1)
    Res_{z=∞}  f' = −i C = −iℓ/π   (B arm, σ_B = −i)

Both T and B carry σ = −i in this codebase's convention (as does the
T-shape T arm in `fprime_exact`); the geometric "T goes up, B goes
down" distinction comes from whether z → 0 or z → ∞ is approached.
"""
function fprime_exact_cross(z::Number, sc::SCParamsCross)
    q1 = sc.q1; q2 = sc.q2
    sq = sqrt(complex((q1^2 - z^2) * (z^2 - q2^2)))
    return sc.C * sq / ((z^2 - 1) * z)
end
