"""
Local coordinate maps f_i, g_i for each arm of the cross-shaped (4-arm) domain.

Split off from `LocalCoordinates.jl` (which retains the T-shape implementation);
shares `ArmData`, `_binomial_series`, `_geometric_series` and the helpers in
`TruncLaurent.jl`. Must be `include`d after `LocalCoordinates.jl`.
"""

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
