"""
Local coordinate maps f_i, g_i for each arm of the T-shaped domain.
"""

struct ArmData
    label::Symbol
    x::Float64             # marked point on ℝ: -1, 1, or 0
    w::Float64             # arm width: 1 for L/R, ℓ for T
    σ::ComplexF64          # arm direction
    α::ComplexF64          # leading coefficient of f_i
    f_series::TruncLaurent{ComplexF64}   # f_i(x_i + ζ) as series in ζ, val=1
    g_series::TruncLaurent{ComplexF64}   # g_i(ξ) - x_i as series in ξ, val=1
    fprime_series::TruncLaurent{ComplexF64}  # f'(x_i + ζ) as Laurent series, val=-1
end

struct Geometry
    sc::SCParams
    arms::NamedTuple{(:L, :R, :T), NTuple{3, ArmData}}
    order::Int
end

"""
    compute_geometry(ℓ, order) -> Geometry

Compute SC map parameters, local coordinate maps, and their inverses
for the T-shaped domain with vertical arm width ℓ.
Series are computed to the given order.
"""
function compute_geometry(ℓ::Real, order::Int)
    sc = compute_sc_params(ℓ)
    arm_R = _compute_arm(sc, :R, order)
    arm_L = _compute_arm(sc, :L, order)
    arm_T = _compute_arm(sc, :T, order)
    Geometry(sc, (L=arm_L, R=arm_R, T=arm_T), order)
end

function _compute_arm(sc::SCParams, label::Symbol, order::Int)
    # σ_i is determined by residue of f' at x_i: w_i σ_i / π = Res_{z=x_i} f'.
    # At z=±1: Res = +1/π, so σ_L = σ_R = +1 (both arms have same log coefficient).
    # At z=0:  Res = -iℓ/π, so σ_T = -i.
    if label == :R
        x = 1.0; w = 1.0; σ = 1.0 + 0.0im
    elseif label == :L
        x = -1.0; w = 1.0; σ = 1.0 + 0.0im
    elseif label == :T
        x = 0.0; w = sc.ell; σ = 0.0 - 1.0im
    else
        error("Unknown arm label: $label")
    end

    # Step 1: Laurent-expand f'(z) around x
    fprime_laurent = _expand_fprime(sc, x, order)

    # Step 2: Integrate to get ρ(ζ) = regular part of ∫f'dz
    # f(x+ζ) = (wσ/π) log(ζ) + ρ(ζ)
    # So ρ'(ζ) = f'(x+ζ) - (wσ/π)/ζ = regular part of f'
    # ρ(ζ) = ρ₀ + Σ_{n≥1} a_{n-1}/n * ζⁿ  where a_n are regular coeffs of f'
    # ρ₀ is an integration constant determined by alignment
    residue = fprime_laurent[-1]
    fprime_reg = regular_part(fprime_laurent)
    prec = order + 1

    # ρ coefficients (indices 0, 1, 2, ...)
    # ρ^(n) = a^(n-1)/n for n ≥ 1, where a^(k) = fprime_reg[k]
    rho_coeffs = zeros(ComplexF64, prec)
    for n in 1:prec-1
        rho_coeffs[n + 1] = fprime_reg[n - 1] / n
    end

    # ρ₀: integration constant of the SC map f(z) = ∫ f'(z) dz + C.
    # Determined geometrically by matching f at the corner z = p from
    # different arm series (all must agree since f is a single function).
    rho_coeffs[1] = _compute_rho0(sc, label)

    # Step 3: Build f_i(z) = exp(-π σ* / w · f(z))
    # f(x+ζ) = (wσ/π) log(ζ) + ρ(ζ)
    # f_i(z) = exp(-πσ*/w · f(z)) where σ* is chosen so σ*σ = -1
    # σ* = -conj(σ)/|σ|² for our convention... actually σ*σ = -1 means σ* = -1/σ
    σ_star = -1.0 / σ

    # f_i(z) = exp(-πσ*/w · ((wσ/π)log(ζ) + ρ(ζ)))
    #        = exp(-σ*σ · log(ζ)) · exp(-πσ*/w · ρ(ζ))
    #        = ζ^(−σ*σ) · exp(-πσ*/w · ρ(ζ))
    #        = ζ · exp(-πσ*/w · ρ(ζ))    [since -σ*σ = -(-1/σ)σ = 1]
    #        = ζ · exp((-πσ*/w) · ρ(ζ))

    # The exponent: (-πσ*/w) · ρ(ζ) = (-πσ*/w) · (ρ₀ + ρ₁ζ + ρ₂ζ² + ...)
    # The ζ⁰ part gives α_factor = exp(-πσ*/w · ρ₀)
    # The ζ≥1 part gives exp_series input

    coeff_factor = -π * σ_star / w

    # Build the O(ζ) part of the exponent
    exp_input_coeffs = zeros(ComplexF64, prec - 1)
    for n in 1:prec-1
        exp_input_coeffs[n] = coeff_factor * rho_coeffs[n + 1]
    end
    exp_input = TruncLaurent(1, exp_input_coeffs, prec)
    exp_part = exp_series(exp_input)

    α_factor = exp(coeff_factor * rho_coeffs[1])

    # f_i(ζ) = α_factor · ζ · exp_part(ζ)
    # Multiply ζ · exp_part: shift valuation
    fi_coeffs = ComplexF64[α_factor * c for c in exp_part.coeffs]
    f_series = TruncLaurent(1, fi_coeffs, prec)

    α = f_series[1]

    # Step 4: Invert f_i to get g_i
    g_series = series_revert(f_series)

    ArmData(label, x, w, σ, α, f_series, g_series, fprime_laurent)
end

"""
    _compute_rho0(sc::SCParams, label::Symbol) -> ComplexF64

Compute the integration constant ρ₀ for arm `label`. The SC map
f(z) = ∫f'(z)dz has one free additive constant C. The ρ₀ for each arm
is chosen so that the reflex corners z = ±p land on the unit semicircle
|ξ_i| = 1, giving Neumann series R_conv = 1 for all arms.

Orientation convention: ρ₀ is chosen so that f_i maps UHP of z to the
upper semidisc of ξ (α_i real positive for all three arms).
This forces the adjacent reflex corner to ξ_R(p) = −1 for the R arm
and ξ_L(−p) = +1 for the L arm (the "mouth-at-corner" integration
constant ρ₀^R_num from singularity-subtracted numerical integration
already gives α > 0 real with no extra shift).

- R arm: ρ₀ = ρ₀^R_num (real, from numerical integration). α_R > 0 real,
  ξ_R(p) = −1.
- L arm: ρ₀ = ρ₀^R_num (same value). α_L > 0 real, ξ_L(−p) = +1.
- T arm: ρ₀ targets f(p) = 0 so that ξ_T(p) = +1, ξ_T(−p) = −1;
  α_T > 0 real.
"""
function _compute_rho0(sc::SCParams, label::Symbol)
    if label == :R
        return ComplexF64(_compute_rho0_R(sc))
    elseif label == :L
        return ComplexF64(_compute_rho0_R(sc))
    else
        return _compute_rho0_T(sc)
    end
end

"""
Compute ρ₀ for the R arm by numerical integration with singularity subtraction.

Integrates g(z) = f'(z) - (1/π)/(z-1) on [p, 1]. This function is smooth
(the pole of f' at z = 1 is subtracted analytically). Then:

    ρ₀^R = ∫_p^1 g(z) dz − (1/π) log(1−p)

This places the mouth at the R corner: Re(f(p)) = 0. Result is real.
"""
function _compute_rho0_R(sc::SCParams)
    p = sc.p
    C = sc.C
    ℓ = sc.ell

    function g(z)
        fp = -C * sqrt(z * z - p * p) / (z * (1 - z * z))
        fp - 1 / (π * (z - 1))
    end

    npts = 10000
    a = p; b = 1.0 - 1e-12
    h = (b - a) / npts
    s = g(a) + g(b)
    for i in 1:npts-1
        z = a + i * h
        s += (iseven(i) ? 2 : 4) * g(z)
    end
    integral = s * h / 3

    integral - log(1 - p) / π
end

"""
Compute ρ₀ for the T arm by placing corners at |ξ_T| = 1.

Convention: the two reflex corners z = ±p map to ξ_T = ±1
(endpoints of the upper semicircle in D⁺). Achieved by setting
f(p) = 0 from the T arm expansion, so ξ_T(p) = exp(iπ·0/ℓ) = 1.

This gives Neumann series R_conv = 1 for the T arm, matching the
natural convention already satisfied by the L/R arms.
"""
function _compute_rho0_T(sc::SCParams)
    p = sc.p
    ℓ = sc.ell

    fprime_T = _expand_fprime_at_T(sc.C, sc.p, 41)
    fprime_reg_T = regular_part(fprime_T)

    rho_T = zeros(ComplexF64, 41)
    for n in 1:40
        rho_T[n + 1] = fprime_reg_T[n - 1] / n
    end

    # f_T(p) with ρ₀^T = 0: Res_T · log(p) + Σ ρ_n · p^n
    f_T_at_p = (-im * ℓ / π) * log(Complex(p))
    zp = ComplexF64(1)
    for n in 1:40
        zp *= p
        f_T_at_p += rho_T[n + 1] * zp
    end

    # Target f(p) = 0 so that ξ_T(p) = exp(iπ/ℓ · 0) = +1
    -f_T_at_p
end

"""
Laurent-expand f'(z) = C√(z²-p²)/(z(z²-1)) around z = x₀.
Returns a TruncLaurent in ζ where z = x₀ + ζ.
"""
function _expand_fprime(sc::SCParams, x₀::Float64, order::Int)
    C = sc.C
    p = sc.p
    prec = order + 1

    if x₀ == 1.0
        _expand_fprime_at_R(C, p, prec)
    elseif x₀ == -1.0
        _expand_fprime_at_L(C, p, prec)
    elseif x₀ == 0.0
        _expand_fprime_at_T(C, p, prec)
    else
        error("Unsupported expansion point x₀ = $x₀")
    end
end

function _expand_fprime_at_R(C::Float64, p::Float64, prec::Int)
    # z = 1 + ζ
    # f'(z) = C√((1+ζ)²-p²) / ((1+ζ)((1+ζ)²-1))
    #       = C√(1-p² + 2ζ + ζ²) / ((1+ζ)·ζ·(2+ζ))

    # Numerator: √(1-p² + 2ζ + ζ²) = √(1-p²) · √(1 + (2ζ+ζ²)/(1-p²))
    a = 1 - p^2
    sqrt_a = sqrt(a)

    # Build u(ζ) = (2ζ + ζ²)/a as series with val=1
    u_coeffs = ComplexF64[2/a, 1/a]
    # Pad to prec
    append!(u_coeffs, zeros(ComplexF64, prec - 2))
    u = TruncLaurent(1, u_coeffs[1:prec-1], prec)

    # √(1+u) via binomial series: compute as exp(0.5 * log(1+u))
    # Better: iterative. (1+u)^{1/2} = Σ binom(1/2, k) u^k
    sqrt_series = _binomial_series(0.5, u, prec)
    # Multiply by √a: this is the numerator as a val=0 series
    num = TruncLaurent(0, ComplexF64[sqrt_a * c for c in sqrt_series.coeffs], prec)

    # Denominator: (1+ζ)·(2+ζ) = 2 + 3ζ + ζ²
    # 1/(1+ζ) = 1 - ζ + ζ² - ...
    inv1 = _geometric_series(-one(ComplexF64), prec)
    # 1/(2+ζ) = (1/2)·1/(1+ζ/2) = (1/2)(1 - ζ/2 + ζ²/4 - ...)
    inv2 = _geometric_series(ComplexF64(-1/2), prec)
    inv2 = TruncLaurent(0, ComplexF64[c/2 for c in inv2.coeffs], prec)

    # 1/((1+ζ)(2+ζ)) = inv1 * inv2
    denom_inv = inv1 * inv2

    # f'(z) = C · num · (1/ζ) · denom_inv
    # = C/ζ · (num * denom_inv)
    product = num * denom_inv
    # Shift by -1 (multiply by ζ⁻¹)
    result_coeffs = ComplexF64[C * c for c in product.coeffs]
    TruncLaurent(-1, result_coeffs, prec - 1)
end

function _expand_fprime_at_L(C::Float64, p::Float64, prec::Int)
    # z = -1 + ζ
    # By Z₂ symmetry f'(-z) = -f'(z) (since f is odd under z→-z up to constants)
    # Actually f'(z) = C√(z²-p²)/(z(z²-1)), so f'(-z) = C√(z²-p²)/((-z)(z²-1)) = -f'(z)
    # So expanding at -1+ζ: f'(-1+ζ) = -f'(1-ζ)
    # If the R expansion is F_R(ζ) = Σ a_n ζ^n, then F_L(ζ) = -F_R(-ζ) = -Σ a_n (-ζ)^n = Σ (-1)^(n+1) a_n ζ^n

    fp_R = _expand_fprime_at_R(C, p, prec)
    coeffs_L = ComplexF64[(-1)^(n+1) * fp_R.coeffs[n] for n in eachindex(fp_R.coeffs)]
    TruncLaurent(fp_R.val, coeffs_L, fp_R.prec)
end

function _expand_fprime_at_T(C::Float64, p::Float64, prec::Int)
    # z = ζ (x₀ = 0)
    # f'(ζ) = C√(ζ²-p²) / (ζ(ζ²-1))

    # √(ζ²-p²) = ip·√(1 - ζ²/p²) for the correct branch in UHP
    # 1/(ζ²-1) = -1/(1-ζ²) = -(1 + ζ² + ζ⁴ + ...) (geometric series in ζ²)

    # Build √(1 - ζ²/p²): let v = -ζ²/p², then (1+v)^{1/2}
    # v is even in ζ, so the result is even in ζ too
    # Build v as a series with val=2
    v_coeffs = zeros(ComplexF64, prec - 2)
    if length(v_coeffs) >= 1
        v_coeffs[1] = ComplexF64(-1/p^2)  # coefficient of ζ²
    end
    v = TruncLaurent(2, v_coeffs, prec)
    sqrt_series = _binomial_series(0.5, v, prec)
    # Multiply by ip: numerator = ip · sqrt_series
    num = TruncLaurent(0, ComplexF64[im * p * c for c in sqrt_series.coeffs], prec)

    # 1/(ζ(ζ²-1)) = (1/ζ) · (-1/(1-ζ²))
    # -1/(1-ζ²) = -(1 + ζ² + ζ⁴ + ...)
    geo_coeffs = zeros(ComplexF64, prec)
    for k in 1:prec
        # coefficient of ζ^(k-1): nonzero only for even powers
        if (k - 1) % 2 == 0
            geo_coeffs[k] = -one(ComplexF64)
        end
    end
    geo = TruncLaurent(0, geo_coeffs, prec)

    # Product: num * geo, then multiply by C/ζ
    product = num * geo
    result_coeffs = ComplexF64[C * c for c in product.coeffs]
    TruncLaurent(-1, result_coeffs, product.prec - 1)
end

"""
Compute (1 + u)^α as a power series, where u has valuation ≥ 1.
Uses the binomial series: (1+u)^α = Σ binom(α,k) u^k.
"""
function _binomial_series(α::Float64, u::TruncLaurent{ComplexF64}, prec::Int)
    # Horner or direct accumulation of binom(α,k) * u^k
    result = TruncLaurent(0, ComplexF64[one(ComplexF64)], prec)  # start with 1
    u_power = u  # u^1
    binom_coeff = ComplexF64(α)  # binom(α, 1) = α

    for k in 1:prec
        term = TruncLaurent(u_power.val, ComplexF64[binom_coeff * c for c in u_power.coeffs], prec)
        result = _add(result, term)
        # Update: u_power *= u, binom_coeff *= (α-k)/(k+1)
        u_power = _truncmul(u_power, u, prec)
        u_power.val >= prec && break
        binom_coeff *= (α - k) / (k + 1)
    end
    result
end

"""
Geometric series 1/(1 - r·ζ) = 1 + rζ + r²ζ² + ... as val=0 TruncLaurent.
Note: pass r such that the denominator is (1 + r·ζ), i.e., r includes the sign.
Actually this computes 1 + rζ + r²ζ² + ..., which is 1/(1 - rζ).
"""
function _geometric_series(r::ComplexF64, prec::Int)
    coeffs = zeros(ComplexF64, prec)
    coeffs[1] = one(ComplexF64)
    for k in 2:prec
        coeffs[k] = coeffs[k-1] * r
    end
    TruncLaurent(0, coeffs, prec)
end

# Re-use _add, _truncmul from TruncLaurent.jl (they are in the module scope)


# ============================================================================
# Cross-shape geometry (4 arms: L, R, T, B; see docs/design/conformal_map_cross.md §1.4)
# ============================================================================
#
# Conventions, matching the factored-branch fprime_exact_cross in SCMap.jl:
#   x_L = -1,  σ_L = -1,    w_L = 1
#   x_R = +1,  σ_R = +1,    w_R = 1
#   x_T = 0,   σ_T = -i,    w_T = ℓ
#   x_B = ∞,   σ_B = +i,    w_B = ℓ   (local coordinate u = 1/z)
#
# For the B arm we Laurent-expand g'(u) = f'(1/u) · (-1/u²) around u = 0.
# The returned ArmData stores x = Inf and the local series in u; downstream
# code that reads x numerically (NeumannCoefficients, PrimaryVertex) will
# need to special-case this in a later plan.

struct GeometryCross
    sc::SCParamsCross
    arms::NamedTuple{(:L, :R, :T, :B), NTuple{4, ArmData}}
    order::Int
end

"""
    compute_geometry_cross(ℓ, order) -> GeometryCross

Build the four-arm cross-shape geometry: SC parameters, local
coordinate maps `f_i`, their inverses `g_i`, and the Laurent
expansions of `f'` (or `g'` for B arm) around each arm preimage.

The ρ₀ for each arm is chosen so that f_i maps UHP of z to the upper
semidisc of ξ (α_i is real positive for L and T, real negative for B
to compensate the UHP↔LHP flip from u = 1/z; the R arm has σ_R = +1
so "α_R real positive" forces east corner to ξ = −1 instead of +1).
See `docs/design/conformal_map_cross.md` §1.4 for the parametrisation
and `docs/design/finite_entanglement_scaling.md` for the downstream
use.

Adjacent-corner ξ values (all on the unit circle):
  ξ_L(1−q₁) = +1,   ξ_R(q₁−1) = −1,
  ξ_T(±q₁) = ±1,    ξ_B(±q₁) = ∓1.
"""
function compute_geometry_cross(ℓ::Real, order::Int)
    sc = compute_sc_params_cross(ℓ)
    arm_L = _compute_arm_cross(sc, :L, order)
    arm_R = _compute_arm_cross(sc, :R, order)
    arm_T = _compute_arm_cross(sc, :T, order)
    arm_B = _compute_arm_cross(sc, :B, order)
    GeometryCross(sc, (L=arm_L, R=arm_R, T=arm_T, B=arm_B), order)
end

function _compute_arm_cross(sc::SCParamsCross, label::Symbol, order::Int)
    if label == :R
        x = 1.0; w = 1.0; σ = ComplexF64(1.0)
        fprime_laurent = _expand_fprime_cross_at_R(sc, order + 1)
    elseif label == :L
        x = -1.0; w = 1.0; σ = ComplexF64(-1.0)
        fprime_laurent = _expand_fprime_cross_at_L(sc, order + 1)
    elseif label == :T
        x = 0.0; w = sc.ell; σ = ComplexF64(0, -1)   # σ_T = −i
        fprime_laurent = _expand_fprime_cross_at_T(sc, order + 1)
    elseif label == :B
        # Special: local coord is u = 1/z, x stored as Inf.
        x = Inf; w = sc.ell; σ = ComplexF64(0, +1)    # σ_B = +i
        fprime_laurent = _expand_gprime_cross_at_B(sc, order + 1)
    else
        error("Unknown cross arm label: $label")
    end

    prec = order + 1
    residue = fprime_laurent[-1]
    fprime_reg = regular_part(fprime_laurent)

    rho_coeffs = zeros(ComplexF64, prec)
    for n in 1:prec-1
        rho_coeffs[n + 1] = fprime_reg[n - 1] / n
    end
    rho_coeffs[1] = _compute_rho0_cross(sc, label)

    σ_star = -1.0 / σ
    coeff_factor = -π * σ_star / w

    exp_input_coeffs = zeros(ComplexF64, prec - 1)
    for n in 1:prec-1
        exp_input_coeffs[n] = coeff_factor * rho_coeffs[n + 1]
    end
    exp_input = TruncLaurent(1, exp_input_coeffs, prec)
    exp_part = exp_series(exp_input)

    α_factor = exp(coeff_factor * rho_coeffs[1])

    fi_coeffs = ComplexF64[α_factor * c for c in exp_part.coeffs]
    f_series = TruncLaurent(1, fi_coeffs, prec)

    α = f_series[1]
    g_series = series_revert(f_series)

    ArmData(label, x, w, σ, α, f_series, g_series, fprime_laurent)
end

# ---------------------------------------------------------------------------
# ρ₀ per arm, via analytic series evaluation at the east-neighbour corner.
# ---------------------------------------------------------------------------
#
# For each arm, f(x_i + ζ) = (w σ/π) log(ζ) + ρ₀ + ρ₁ζ + ρ₂ζ² + ...
# ρ_n for n ≥ 1 are determined by the Laurent expansion of f' at x_i:
# ρ_n = (regular part of f' at ζ=0)[n-1] / n.
# ρ₀ is chosen to place ξ_i(east corner) on the unit circle at +1.
#
# For the T and B arms the east corner is at ζ_e = q_1 (positive real).
# For the L and R arms it's at ζ_e = ±(1-q_1).

"""
    _compute_rho0_cross(sc, label) -> ComplexF64

Compute ρ₀ for a cross arm by numerical integration of f' with the
log-singularity subtracted, then add the analytic log piece. For each
arm: ρ(ζ_e) = ρ₀ + ∫_0^{ζ_e} [f' − residue/ζ] dζ, target ρ(ζ_e) is
fixed by the ξ_i(east corner) convention. The integrand is smooth on
the integration interval (pole at the arm preimage subtracted, branch
points at the corners are finite — f' vanishes there).

Why not use the arm's own Laurent series? The east corner sits exactly
on the series' radius of convergence (|ζ_e| = R_conv by construction),
so partial-sum convergence is O(N^{−1/2}) — 2e−4 error even at N=41,
shrinking only to 3e−6 by N=641. Simpson on a pole-subtracted smooth
integrand gives machine precision in ~10⁴ points.
"""
function _compute_rho0_cross(sc::SCParamsCross, label::Symbol; npts::Int = 20000)
    q1 = sc.q1; ℓ = sc.ell

    # Each arm integrand has √-type branch-point behaviour at the
    # corner-endpoint of the integration interval (the pole endpoint is
    # smooth after pole subtraction). We regularise via a t² substitution
    # at the √-endpoint: Simpson then achieves full O(h⁴) convergence.
    if label == :R
        # Integrand ~ √(z − q1) at z = q1 (lower endpoint); pole at z=1 subtracted.
        target_reg = ComplexF64(-log(1 - q1) / π)
        g = z -> fprime_exact_cross(z, sc) - one(ComplexF64) / (π * (z - 1))
        integral = _simpson_sqrt_at_a(g, q1, 1.0, npts)
        return target_reg + integral
    elseif label == :L
        # Integrand ~ √(−q1 − z) at z = −q1 (upper endpoint); pole at z=−1 subtracted.
        target_reg = ComplexF64(log(1 - q1) / π)
        g = z -> fprime_exact_cross(z, sc) + one(ComplexF64) / (π * (z + 1))
        integral = _simpson_sqrt_at_b(g, -1.0, -q1, npts)
        return target_reg - integral
    elseif label == :T
        # Integrand ~ √(q1 − z) at z = q1 (upper endpoint); pole at z=0 subtracted.
        target_reg = (im * ℓ / π) * log(q1)
        g = z -> fprime_exact_cross(z, sc) + im * ℓ / (π * z)
        integral = _simpson_sqrt_at_b(g, 0.0, q1, npts)
        return target_reg - integral
    elseif label == :B
        # In u: g'(u) = f'(1/u)·(−1/u²); residue +iℓ/π at u=0 subtracted. √-endpoint at u=q1.
        target_reg = -(im * ℓ / π) * log(q1) - ℓ
        g = u -> begin
            gp = fprime_exact_cross(1 / u, sc) * (-1 / u^2)
            gp - im * ℓ / (π * u)
        end
        integral = _simpson_sqrt_at_b(g, 0.0, q1, npts)
        return target_reg - integral
    else
        error("Unknown cross arm label: $label")
    end
end

"""Composite Simpson's rule for ∫_a^b f(z) dz where f has √(z − a)
behaviour at the lower endpoint. Substitutes t = √(z − a), z = a + t²,
dz = 2t dt. Endpoint-regular integrand 2 t f(a + t²) is smooth, so
Simpson achieves O(h⁴). `ε_b` shrinks the upper limit to avoid a pole
at z = b; default 1e−12 is safe for pole-subtracted integrands."""
function _simpson_sqrt_at_a(f, a::Real, b::Real, n::Int; ε_b::Real = 1e-8)
    @assert iseven(n)
    T = sqrt(b - a - ε_b)
    h = T / n
    g(t) = 2 * t * f(complex(a + t^2))
    s = g(0.0) + g(T)
    @inbounds for i in 1:n-1
        s += (iseven(i) ? 2 : 4) * g(i * h)
    end
    s * h / 3
end

"""Composite Simpson's rule for ∫_a^b f(z) dz where f has √(b − z)
behaviour at the upper endpoint. Substitutes s = √(b − z), z = b − s².
`ε_a` offsets from the lower pole at z = a."""
function _simpson_sqrt_at_b(f, a::Real, b::Real, n::Int; ε_a::Real = 1e-8)
    @assert iseven(n)
    T = sqrt(b - a - ε_a)
    h = T / n
    g(t) = 2 * t * f(complex(b - t^2))
    s = g(0.0) + g(T)
    @inbounds for i in 1:n-1
        s += (iseven(i) ? 2 : 4) * g(i * h)
    end
    s * h / 3
end

# ---------------------------------------------------------------------------
# Laurent expansions of f'(z) (or g'(u) for B) around each arm preimage.
# All return a TruncLaurent with valuation -1 (the simple-pole term).
# ---------------------------------------------------------------------------
#
# Cross SC derivative (factored branch, from SCMap.jl):
#   f'(z) = C · (-i) · √(z-q1) · √(z+q1) · √(z-q2) · √(z+q2) / ((z²-1) · z)
#
# At z = 1 (R arm): the four √(z±q1,2) factors have bases
#   √(1-q1), √(1+q1), i·√(q2-1), √(1+q2)
# with product i·√((1-q1²)(q2²-1)) = i·2/ℓ  (residue identities).
# Times C·(-i) · 1/(2ζ) gives the pole +1/(πζ).

function _expand_fprime_cross_at_R(sc::SCParamsCross, prec::Int)
    C = sc.C; q1 = sc.q1; q2 = sc.q2

    # Each factor √(z - z_j) = (base) · √(1 + ζ/(1 - z_j)) at z = 1+ζ.
    base_1_mq1 = sqrt(1 - q1)
    base_1_pq1 = sqrt(1 + q1)
    base_1_mq2 = im * sqrt(q2 - 1)                # √(1 - q2) = +i·√(q2-1) via UHP branch
    base_1_pq2 = sqrt(1 + q2)

    u1 = TruncLaurent(1, ComplexF64[1 / (1 - q1)],  prec)
    u2 = TruncLaurent(1, ComplexF64[1 / (1 + q1)],  prec)
    u3 = TruncLaurent(1, ComplexF64[1 / (1 - q2)],  prec)
    u4 = TruncLaurent(1, ComplexF64[1 / (1 + q2)],  prec)

    s1 = _binomial_series(0.5, u1, prec)
    s2 = _binomial_series(0.5, u2, prec)
    s3 = _binomial_series(0.5, u3, prec)
    s4 = _binomial_series(0.5, u4, prec)

    overall_base = ComplexF64(base_1_mq1 * base_1_pq1 * base_1_mq2 * base_1_pq2)  # = i·2/ℓ
    prod_series = _truncmul(_truncmul(_truncmul(s1, s2, prec), s3, prec), s4, prec)
    prod_scaled = TruncLaurent(0,
        ComplexF64[overall_base * c for c in prod_series.coeffs], prec)

    # Denominator at z = 1+ζ: (z²-1)·z = ζ · (2+ζ) · (1+ζ).
    inv1 = _geometric_series(-one(ComplexF64), prec)                  # 1/(1+ζ)
    inv2_base = _geometric_series(-one(ComplexF64) / 2, prec)
    inv2 = TruncLaurent(0, ComplexF64[c / 2 for c in inv2_base.coeffs], prec)  # 1/(2+ζ)
    denom_inv = _truncmul(inv1, inv2, prec)                            # 1/((1+ζ)(2+ζ))

    product = _truncmul(prod_scaled, denom_inv, prec)
    final_factor = ComplexF64(C * (-im))
    result_coeffs = ComplexF64[final_factor * c for c in product.coeffs]
    TruncLaurent(-1, result_coeffs, prec - 1)
end

function _expand_fprime_cross_at_L(sc::SCParamsCross, prec::Int)
    # Direct expansion around z = -1 + ζ. The naïve Z₂ substitution
    # (used by the T-shape code) is NOT valid here: the factored-branch
    # sqrt has cuts on R from each branch point to -∞, and UHP-approach
    # principal values near z = +1 and z = -1 are on OPPOSITE sides of
    # sqrt(z - q_2)'s cut (which crosses both points), so Z₂ gets the
    # wrong sign at L.
    C = sc.C; q1 = sc.q1; q2 = sc.q2

    # At z = -1 + ζ, UHP-approach principal values of the four √:
    #   √(z - q1) = √(-(1+q1) + ζ) = i·√(1+q1)·(1 - ζ/(1+q1))^{1/2}
    #   √(z + q1) = √(-(1-q1) + ζ) = i·√(1-q1)·(1 - ζ/(1-q1))^{1/2}
    #   √(z - q2) = √(-(1+q2) + ζ) = i·√(1+q2)·(1 - ζ/(1+q2))^{1/2}
    #   √(z + q2) = √((q2-1) + ζ)  = +√(q2-1)·(1 + ζ/(q2-1))^{1/2}

    base1 = im * sqrt(1 + q1)
    base2 = im * sqrt(1 - q1)
    base3 = im * sqrt(1 + q2)
    base4 = sqrt(q2 - 1)

    u1 = TruncLaurent(1, ComplexF64[-1 / (1 + q1)], prec)
    u2 = TruncLaurent(1, ComplexF64[-1 / (1 - q1)], prec)
    u3 = TruncLaurent(1, ComplexF64[-1 / (1 + q2)], prec)
    u4 = TruncLaurent(1, ComplexF64[+1 / (q2 - 1)], prec)

    s1 = _binomial_series(0.5, u1, prec)
    s2 = _binomial_series(0.5, u2, prec)
    s3 = _binomial_series(0.5, u3, prec)
    s4 = _binomial_series(0.5, u4, prec)

    # Overall leading factor: i³·√((1+q1)(1-q1)(1+q2)(q2-1)) = -i·(2/ℓ).
    overall_base = ComplexF64(base1 * base2 * base3 * base4)
    prod_series = _truncmul(_truncmul(_truncmul(s1, s2, prec), s3, prec), s4, prec)
    prod_scaled = TruncLaurent(0,
        ComplexF64[overall_base * c for c in prod_series.coeffs], prec)

    # Denominator at z = -1+ζ: (z²-1)·z = ζ·(ζ-2)·(ζ-1).
    # 1/(ζ-2) = -(1/2)·1/(1-ζ/2);  1/(ζ-1) = -1/(1-ζ).
    inv_zm2_base = _geometric_series(one(ComplexF64) / 2, prec)       # 1/(1-ζ/2)
    inv_zm2 = TruncLaurent(0,
        ComplexF64[-c / 2 for c in inv_zm2_base.coeffs], prec)        # -(1/2)/(1-ζ/2)
    inv_zm1_base = _geometric_series(one(ComplexF64), prec)           # 1/(1-ζ)
    inv_zm1 = TruncLaurent(0,
        ComplexF64[-c for c in inv_zm1_base.coeffs], prec)            # -1/(1-ζ)
    denom_inv = _truncmul(inv_zm2, inv_zm1, prec)                      # 1/((ζ-2)(ζ-1))

    product = _truncmul(prod_scaled, denom_inv, prec)
    final_factor = ComplexF64(C * (-im))
    result_coeffs = ComplexF64[final_factor * c for c in product.coeffs]
    TruncLaurent(-1, result_coeffs, prec - 1)
end

function _expand_fprime_cross_at_T(sc::SCParamsCross, prec::Int)
    C = sc.C; q1 = sc.q1; q2 = sc.q2
    # At z = ζ, using q1 q2 = 1:
    #   √((ζ²-q1²)(ζ²-q2²)) = ±(1 - q2²ζ²)^{1/2} · (1 - q1²ζ²)^{1/2} · q1 q2,
    # with the overall sign fixed by the factored branch so that at ζ=0 the
    # product equals -1. Combined with C·(-i):
    #   f'_T(ζ) = (-i·C) · (1-q1²ζ²)^{1/2} · (1-q2²ζ²)^{1/2} / (ζ·(1-ζ²))
    # (sign tracking: denom ζ(ζ²-1) = -ζ(1-ζ²); factored sqrt leading = -1;
    # multiplying -i·C · (-1) / (-ζ(1-ζ²)) = -iC / (ζ(1-ζ²)).)

    v1_coeffs = zeros(ComplexF64, max(prec - 2, 0))
    if length(v1_coeffs) >= 1
        v1_coeffs[1] = ComplexF64(-q1^2)
    end
    v1 = TruncLaurent(2, v1_coeffs, prec)
    s1 = _binomial_series(0.5, v1, prec)

    v2_coeffs = zeros(ComplexF64, max(prec - 2, 0))
    if length(v2_coeffs) >= 1
        v2_coeffs[1] = ComplexF64(-q2^2)
    end
    v2 = TruncLaurent(2, v2_coeffs, prec)
    s2 = _binomial_series(0.5, v2, prec)

    num = _truncmul(s1, s2, prec)
    factor = ComplexF64(-im * C)
    num_scaled = TruncLaurent(0, ComplexF64[factor * c for c in num.coeffs], prec)

    # 1/(1 - ζ²): geometric series in ζ² (val=0, only even indices nonzero).
    geo_coeffs = zeros(ComplexF64, prec)
    for k in 1:prec
        if (k - 1) % 2 == 0
            geo_coeffs[k] = one(ComplexF64)
        end
    end
    geo = TruncLaurent(0, geo_coeffs, prec)

    product = _truncmul(num_scaled, geo, prec)
    TruncLaurent(-1, product.coeffs, product.prec - 1)
end

function _expand_gprime_cross_at_B(sc::SCParamsCross, prec::Int)
    C = sc.C; q1 = sc.q1; q2 = sc.q2
    # g'(u) = f'(1/u) · (-1/u²)
    #       = (+i C / u) · √((1-q1²u²)(1-q2²u²)) · 1/(1-u²)

    v1_coeffs = zeros(ComplexF64, max(prec - 2, 0))
    if length(v1_coeffs) >= 1
        v1_coeffs[1] = ComplexF64(-q1^2)
    end
    v1 = TruncLaurent(2, v1_coeffs, prec)
    s1 = _binomial_series(0.5, v1, prec)

    v2_coeffs = zeros(ComplexF64, max(prec - 2, 0))
    if length(v2_coeffs) >= 1
        v2_coeffs[1] = ComplexF64(-q2^2)
    end
    v2 = TruncLaurent(2, v2_coeffs, prec)
    s2 = _binomial_series(0.5, v2, prec)

    num = _truncmul(s1, s2, prec)
    factor = ComplexF64(im * C)                  # note: +i for B, -i for T
    num_scaled = TruncLaurent(0, ComplexF64[factor * c for c in num.coeffs], prec)

    # 1/(1 - u²): geometric in u².
    geo_coeffs = zeros(ComplexF64, prec)
    for k in 1:prec
        if (k - 1) % 2 == 0
            geo_coeffs[k] = one(ComplexF64)
        end
    end
    geo = TruncLaurent(0, geo_coeffs, prec)

    product = _truncmul(num_scaled, geo, prec)
    TruncLaurent(-1, product.coeffs, product.prec - 1)
end
