using Test
using CFTTruncation: compute_geometry_cross, compute_sc_params_cross, TruncLaurent
using CFTTruncation: evaluate, series_revert

"""Evaluate a TruncLaurent at ζ by direct summation (handles negative val)."""
function _eval_series(s::TruncLaurent, ζ)
    T = eltype(s.coeffs)
    result = zero(T) * one(complex(ζ))
    ζ_pow = complex(ζ)^s.val
    for c in s.coeffs
        result += c * ζ_pow
        ζ_pow *= ζ
    end
    result
end

@testset "LocalCoordinates cross" begin
    order = 30

    @testset "construction runs, fields populated" begin
        geom = compute_geometry_cross(1.0, order)
        for nm in (:L, :R, :T, :B)
            arm = getfield(geom.arms, nm)
            @test arm.label === nm
            @test isfinite(real(arm.α)) && isfinite(imag(arm.α))
            @test abs(arm.α) > 0
            @test arm.f_series.val == 1
        end
        @test geom.arms.L.x == -1.0
        @test geom.arms.R.x == +1.0
        @test geom.arms.T.x == 0.0
        @test isinf(geom.arms.B.x)
    end

    @testset "residues match w_i σ_i / π" begin
        for ℓ in (0.5, 1.0, 2.0)
            geom = compute_geometry_cross(ℓ, order)
            @test isapprox(geom.arms.R.fprime_series[-1], 1/π; atol=1e-10)
            @test isapprox(geom.arms.L.fprime_series[-1], -1/π; atol=1e-10)
            @test isapprox(geom.arms.T.fprime_series[-1], -1im * ℓ/π; atol=1e-10)
            @test isapprox(geom.arms.B.fprime_series[-1], +1im * ℓ/π; atol=1e-10)
        end
    end

    @testset "f_i ∘ g_i = id (round-trip)" begin
        for ℓ in (0.5, 1.0, 2.0)
            geom = compute_geometry_cross(ℓ, order)
            for nm in (:L, :R, :T, :B)
                arm = getfield(geom.arms, nm)
                for ξ in (0.3 + 0.2im, 0.1 - 0.4im, 0.5 + 0.3im)
                    # f_i(g_i(ξ)) = ξ, using library f_series and g_series.
                    # Tolerance is 2e-3 at order 30: series_revert precision
                    # is direction-dependent, and the ρ₀ sign convention
                    # reflects ξ → −ξ for R and B, putting the test points
                    # on the slower-converging side for those arms.
                    ζ = evaluate(arm.g_series, ξ)
                    ξ_back = _eval_series(arm.f_series, ζ)
                    @test isapprox(ξ_back, ξ; atol=2e-3)
                end
            end
        end
    end

    @testset "α leading coefficient consistency" begin
        for ℓ in (0.5, 1.0, 2.0)
            geom = compute_geometry_cross(ℓ, order)
            for nm in (:L, :R, :T, :B)
                arm = getfield(geom.arms, nm)
                @test arm.α ≈ arm.f_series[1]
            end
        end
    end

    @testset "horizontal Z₂: |α_L| = |α_R|" begin
        # In the factored-branch convention, σ_L = -1, σ_R = +1. The α's
        # share magnitude (and differ in overall sign).
        for ℓ in (0.5, 1.0, 2.0, 3.0)
            geom = compute_geometry_cross(ℓ, order)
            @test abs(geom.arms.L.α) ≈ abs(geom.arms.R.α)
        end
    end

    @testset "vertical Z₂-like: |α_T| = |α_B|" begin
        # T and B arms (σ = ∓i, both width ℓ) share |α|.
        for ℓ in (0.5, 1.0, 2.0, 3.0)
            geom = compute_geometry_cross(ℓ, order)
            @test abs(geom.arms.T.α) ≈ abs(geom.arms.B.α)
        end
    end

    @testset "R_conv = 1: east-corner ξ on unit circle" begin
        # Orientation convention: f_i maps UHP of z to the upper semidisc
        # of ξ on every arm. With that, ξ_L(east) = ξ_T(east) = +1 and
        # ξ_R(east) = ξ_B(east) = −1 (R has σ_R=+1 forcing the sign, and
        # B has the extra UHP↔LHP flip from u=1/z). Tolerance is relaxed
        # because we evaluate at the series convergence boundary.
        for ℓ in (0.5, 1.0, 2.0)
            geom = compute_geometry_cross(ℓ, order)
            q1 = geom.sc.q1
            # R arm: east corner at z = +q1, ζ = q1 - 1 (negative real)
            ξ_R = _eval_series(geom.arms.R.f_series, complex(q1 - 1.0))
            @test isapprox(ξ_R, -1.0; atol=5e-3)
            # L arm: east corner at z = -q1, ζ = 1 - q1 (positive real)
            ξ_L = _eval_series(geom.arms.L.f_series, complex(1.0 - q1))
            @test isapprox(ξ_L, 1.0; atol=5e-3)
            # T arm: east corner at z = +q1, ζ = q1 (positive real)
            ξ_T_east = _eval_series(geom.arms.T.f_series, complex(q1))
            @test isapprox(ξ_T_east, 1.0; atol=5e-3)
            # B arm: east corner at u = 1/q2 = q1 (since q1·q2=1)
            ξ_B_east = _eval_series(geom.arms.B.f_series, complex(q1))
            @test isapprox(ξ_B_east, -1.0; atol=5e-3)
        end
    end

    @testset "R_conv west corner: T at ξ = −1, B at ξ = +1" begin
        # T and B each see TWO adjacent corners within their R_conv.
        # T arm: ξ_T(±q1) = ±1. B arm (east at −1): by horizontal Z₂,
        # ξ_B(−q1) = +1. (L and R arms see only ONE corner within R_conv.)
        for ℓ in (0.5, 1.0, 2.0)
            geom = compute_geometry_cross(ℓ, order)
            q1 = geom.sc.q1
            ξ_T_west = _eval_series(geom.arms.T.f_series, complex(-q1))
            @test isapprox(ξ_T_west, -1.0; atol=5e-3)
            ξ_B_west = _eval_series(geom.arms.B.f_series, complex(-q1))
            @test isapprox(ξ_B_west, +1.0; atol=5e-3)
        end
    end
end
