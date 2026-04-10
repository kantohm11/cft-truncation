using Test
using CFTTruncation: TruncLaurent, valuation, series_precision, singular_part,
                     regular_part, exp_series, compose, series_revert, evaluate

@testset "TruncLaurent" begin

    @testset "1.1 Construction and access" begin
        # Series: 2ζ⁻¹ + 3 + ζ, truncated at O(ζ²)
        s = TruncLaurent(-1, [2.0, 3.0, 1.0], 2)
        @test valuation(s) == -1
        @test series_precision(s) == 2
        @test s[-1] == 2.0
        @test s[0] == 3.0
        @test s[1] == 1.0
        # Out of range returns zero
        @test s[-2] == 0.0
        @test s[2] == 0.0
    end

    @testset "1.2 Multiplication" begin
        # (1 + 2ζ)(3 + ζ) = 3 + 7ζ + 2ζ²
        a = TruncLaurent(0, [1.0, 2.0], 3)
        b = TruncLaurent(0, [3.0, 1.0], 3)
        c = a * b
        @test c[0] ≈ 3.0
        @test c[1] ≈ 7.0
        @test c[2] ≈ 2.0

        # Laurent × Laurent: (ζ⁻¹ + 1)(2ζ⁻¹ - 1) = 2ζ⁻² + ζ⁻¹ - 1
        a = TruncLaurent(-1, [1.0, 1.0], 2)
        b = TruncLaurent(-1, [2.0, -1.0], 2)
        c = a * b
        @test c[-2] ≈ 2.0
        @test c[-1] ≈ 1.0
        @test c[0] ≈ -1.0
    end

    @testset "1.3 Inversion" begin
        # (1 + ζ)⁻¹ = 1 - ζ + ζ² - ζ³ + ...
        a = TruncLaurent(0, [1.0, 1.0, 0.0, 0.0, 0.0], 5)
        b = inv(a)
        @test b[0] ≈ 1.0
        @test b[1] ≈ -1.0
        @test b[2] ≈ 1.0
        @test b[3] ≈ -1.0
        # round-trip
        c = a * b
        @test c[0] ≈ 1.0
        for k in 1:4
            @test abs(c[k]) < 1e-14
        end

        # (2 + 3ζ)⁻¹ = (1/2)(1 - 3ζ/2 + 9ζ²/4 - ...)
        a = TruncLaurent(0, [2.0, 3.0, 0.0, 0.0], 4)
        b = inv(a)
        @test b[0] ≈ 1/2
        @test b[1] ≈ -3/4
        @test b[2] ≈ 9/8
    end

    @testset "1.4 Exponentiation" begin
        # exp(ζ) = 1 + ζ + ζ²/2 + ζ³/6 + ...
        a = TruncLaurent(1, [1.0], 6)
        b = exp_series(a)
        @test b[0] ≈ 1.0
        @test b[1] ≈ 1.0
        @test b[2] ≈ 1/2
        @test b[3] ≈ 1/6
        @test b[4] ≈ 1/24

        # exp(2ζ + ζ²) = 1 + 2ζ + 3ζ² + ...
        a = TruncLaurent(1, [2.0, 1.0], 5)
        b = exp_series(a)
        @test b[0] ≈ 1.0
        @test b[1] ≈ 2.0
        @test b[2] ≈ 3.0
    end

    @testset "1.5 Composition" begin
        # f(g(ξ)) where g(ξ) = 2ξ + ξ², f(ζ) = 1 + 3ζ + ζ²
        # f(g(ξ)) = 1 + 6ξ + 7ξ² + ...
        f = TruncLaurent(0, [1.0, 3.0, 1.0], 4)
        g = TruncLaurent(1, [2.0, 1.0], 4)
        h = compose(f, g)
        @test h[0] ≈ 1.0
        @test h[1] ≈ 6.0
        @test h[2] ≈ 7.0
    end

    @testset "1.6 Series reversion" begin
        # f(ζ) = 2ζ + ζ², g = f⁻¹, f(g(ξ)) = ξ
        f = TruncLaurent(1, [2.0, 1.0, 0.0, 0.0], 5)
        g = series_revert(f)
        @test g[1] ≈ 1/2
        @test g[2] ≈ -1/8

        # round-trip
        h = compose(f, g)
        @test h[1] ≈ 1.0
        for k in 2:4
            @test abs(h[k]) < 1e-13
        end
    end

    @testset "1.7 Singular / regular part" begin
        s = TruncLaurent(-2, [1.0, 3.0, 5.0, 2.0, 7.0], 3)
        # = ζ⁻² + 3ζ⁻¹ + 5 + 2ζ + 7ζ²
        sp = singular_part(s)
        @test sp[-2] ≈ 1.0
        @test sp[-1] ≈ 3.0
        @test valuation(sp) == -2
        @test series_precision(sp) == 0

        sr = regular_part(s)
        @test sr[0] ≈ 5.0
        @test sr[1] ≈ 2.0
        @test sr[2] ≈ 7.0
    end

    @testset "1.8 BigFloat cross-check" begin
        a = TruncLaurent(0, BigFloat[1, 1, 0, 0, 0], 5)
        b = inv(a)
        @test Float64(b[3]) ≈ -1.0
    end

    @testset "Evaluate" begin
        # f(ζ) = 2 + 3ζ, f(0.5) = 3.5
        f = TruncLaurent(0, [2.0, 3.0], 3)
        @test evaluate(f, 0.5) ≈ 3.5

        # Laurent: f(ζ) = ζ⁻¹ + 1, f(2) = 1.5
        f = TruncLaurent(-1, [1.0, 1.0], 2)
        @test evaluate(f, 2.0) ≈ 1.5
    end

end
