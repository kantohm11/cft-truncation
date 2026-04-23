using Test
using CFTTruncation: compute_sc_params_cross, fprime_exact_cross

@testset "SCMap cross" begin

    @testset "closed-form values" begin
        # ℓ = 1: q_1 = √2 − 1, q_2 = √2 + 1, C = 1/π.
        sc = compute_sc_params_cross(1.0)
        @test sc.q1 ≈ sqrt(2) - 1
        @test sc.q2 ≈ sqrt(2) + 1
        @test sc.C ≈ 1/π

        # ℓ = 2: q_1 = (√5 − 1)/2, q_2 = (√5 + 1)/2 = golden ratio, C = 2/π.
        sc = compute_sc_params_cross(2.0)
        @test sc.q1 ≈ (sqrt(5) - 1)/2
        @test sc.q2 ≈ (sqrt(5) + 1)/2
        @test sc.C ≈ 2/π

        # ℓ = 0.5: check the formula directly.
        sc = compute_sc_params_cross(0.5)
        ℓ = 0.5
        @test sc.q1 ≈ (sqrt(1 + ℓ^2) - 1) / ℓ
        @test sc.q2 ≈ (sqrt(1 + ℓ^2) + 1) / ℓ
        @test sc.C ≈ ℓ/π
    end

    @testset "algebraic invariants" begin
        for ℓ in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            sc = compute_sc_params_cross(ℓ)
            # q_1 · q_2 = 1 is forced by the T-arm residue + C = ℓ/π.
            @test sc.q1 * sc.q2 ≈ 1.0
            # C = ℓ/π (set by the B-arm residue).
            @test sc.C ≈ ℓ/π
            # Ordering: 0 < q_1 < 1 < q_2.
            @test 0 < sc.q1 < 1
            @test sc.q2 > 1
        end
    end

    @testset "limits" begin
        # ℓ → 0: vertical arms pinch, q_1 → 0 and q_2 → ∞.
        sc = compute_sc_params_cross(1e-6)
        @test sc.q1 < 1e-6
        @test sc.q2 > 1e6

        # ℓ → ∞: horizontal arms pinch; both q_1 and q_2 approach 1
        # (the ±1 = arm preimages; corners collapse onto them).
        # Leading correction: q_1 ≈ 1 − 1/ℓ, q_2 ≈ 1 + 1/ℓ.
        sc = compute_sc_params_cross(1e6)
        @test sc.q1 > 1 - 2e-6
        @test sc.q2 < 1 + 2e-6

        # Monotonicity of q_2 − q_1 = 2/ℓ.
        for ℓ in [0.1, 1.0, 10.0]
            sc = compute_sc_params_cross(ℓ)
            @test sc.q2 - sc.q1 ≈ 2/ℓ
        end
    end

    @testset "f'(z) evaluation and residues" begin
        sc = compute_sc_params_cross(1.0)
        z = 0.3 + 0.2im
        fp = fprime_exact_cross(z, sc)
        @test fp isa ComplexF64
        @test isfinite(fp)

        # Residue at z = 1 (R arm): Res = +1/π. (The principal-branch sqrt
        # of the numerator is real-positive at z = 1 + iε when approached
        # from UHP.)
        δ = 1e-8
        res_R = fprime_exact_cross(1.0 + δ*im, sc) * (δ*im)
        @test abs(res_R - 1/π) < 1e-4

        # Residue at z = −1 (L arm): also +1/π in this codebase's
        # convention (σ_L = σ_R = +1 — see LocalCoordinates.jl for the
        # T-shape analogue). The naïve z → −z Z_2 would give −1/π, but
        # Z_2 relates UHP values to LHP values, so UHP-approach residues
        # at ±1 come out with the same sign.
        res_L = fprime_exact_cross(-1.0 + δ*im, sc) * (δ*im)
        @test abs(res_L - 1/π) < 1e-4

        # Residue at z = 0 (T arm, w_T σ_T / π = −iℓ/π with σ_T = −i, w_T = ℓ).
        # At ℓ = 1, w_T = 1, so residue = −i/π.
        res_T = fprime_exact_cross(δ*im, sc) * (δ*im)
        @test abs(res_T - (-1im/π)) < 1e-4
    end

    @testset "residues match arm widths across ℓ" begin
        δ = 1e-9
        for ℓ in [0.1, 0.5, 1.0, 2.0, 5.0]
            sc = compute_sc_params_cross(ℓ)
            # R arm (width 1, direction +1): residue = 1/π
            res_R = fprime_exact_cross(1.0 + δ*im, sc) * (δ*im)
            @test abs(res_R - 1/π) < 1e-4
            # T arm (width ℓ, direction −i): residue = −iℓ/π
            res_T = fprime_exact_cross(δ*im, sc) * (δ*im)
            @test abs(res_T - (-1im*ℓ/π)) < 1e-4
            # B arm: check via large imaginary z.
            # For z = iR (R large), f'(iR) ∼ −iC/z_at_∞-limit.
            # Integral around large semicircle gives residue at ∞.
            # Equivalently: substitute u = 1/z, evaluate −u² f'(1/u) at u → 0.
            R_big = 1e8
            val = fprime_exact_cross(R_big * 1im, sc)
            # f'(iR) ≈ −C/R → residue at ∞ = −iC = −iℓ/π
            # Using lim z·f'(z) as z → ∞ gives the NEGATIVE of Res_∞; here
            # z f'(z) at z = iR: iR · (−C/R) = −iC = −iℓ/π. Matches.
            res_B = (1im * R_big) * val
            @test abs(res_B - (-1im*ℓ/π)) < 1e-4
        end
    end

    @testset "horizontal Z_2 symmetry: f'(−z) = −f'(z)" begin
        # Formula is f'(z) = C·√((q_1² − z²)(z² − q_2²)) / ((z² − 1) z).
        # Under z → −z: numerator invariant (even in z), denominator → −(z²−1)z,
        # so f'(−z) = −f'(z).
        for ℓ in [0.5, 1.0, 2.0]
            sc = compute_sc_params_cross(ℓ)
            for z in [0.3 + 0.4im, 0.7 + 0.2im, 0.1 + 1.5im]
                @test fprime_exact_cross(-z, sc) ≈ -fprime_exact_cross(z, sc)
            end
        end
    end
end
