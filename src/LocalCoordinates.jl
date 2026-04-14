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
is determined by requiring all arm expansions to agree at the corners z = ±p.

Convention: mouth at corner. R corner at σ = 0, L corner at σ = ℓ.
For centered plots: post-shift by -ℓ/2.

- R arm: ρ₀ is real, from numerical integration targeting Re(f(p)) = 0 (mouth at corner).
- L arm: ρ₀ = ρ₀^R + i. The Z₂ gives ρ₀^L_Z2 = ρ₀^R + ℓ + i, but the
  L arm local coordinate uses ξ_L = exp(π(f - ℓ)) to shift its mouth to the
  L corner (σ = ℓ). The ℓ is absorbed, leaving ρ₀^L = ρ₀^R + i.
  Result: |α_L| = |α_R| (symmetric).
- T arm: ρ₀ is complex, from matching f_T(p) = f_R(p) = i.
"""
function _compute_rho0(sc::SCParams, label::Symbol)
    if label == :R
        return ComplexF64(_compute_rho0_R(sc))
    elseif label == :L
        # ξ_L = exp(π(f - ℓ)) shifts mouth to L corner.
        # Effective ρ₀^L = ρ₀^R + i (the ℓ from Z₂ is absorbed by the shift).
        return ComplexF64(_compute_rho0_R(sc)) + im
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
Compute ρ₀ for the T arm by matching at the R corner: f_T(p) = f_R(p) = i.

Evaluates f_T(p) from the T arm's f' series (with ρ₀^T = 0), then sets
ρ₀^T = i - f_T(p). Uses 40 series terms; converges well since |p| < 1.
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

    # f_R(p) = i (Re=0: mouth at corner, Im=1: log branch)
    im - f_T_at_p
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
