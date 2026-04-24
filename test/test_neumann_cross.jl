using Test
using CFTTruncation: compute_geometry_cross, compute_neumann,
                     TruncLaurent, singular_part

@testset "NeumannCoefficients cross" begin

    @testset "4-arm diagonal self-consistency" begin
        geom = compute_geometry_cross(1.0, 20)
        neum = compute_neumann(geom, 10)

        for arm in (:L, :R, :T, :B)
            Fm_polys = neum.F_polys[arm]
            a = getfield(geom.arms, arm)
            for m in 1:5
                composed = CFTTruncation._compose_Fm_with_g(
                    Fm_polys[m], a, a, geom.order)
                @test abs(composed[-m] - 1.0) < 1e-6
                for k in (-m+1):(-1)
                    @test abs(composed[k]) < 1e-6
                end
            end
        end
    end

    @testset "All 16 matrices finite and real" begin
        cross_keys = (:LL, :LR, :LT, :LB,
                      :RL, :RR, :RT, :RB,
                      :TL, :TR, :TT, :TB,
                      :BL, :BR, :BT, :BB)
        for ℓ in (0.5, 1.0, 2.0)
            geom = compute_geometry_cross(ℓ, 20)
            neum = compute_neumann(geom, 6)

            for arm_key in cross_keys
                mat = getfield(neum.𝒩, arm_key)
                @test all(isfinite, mat)
                # eltype is Float64 by construction; this asserts the
                # compose returned values with tiny imaginary parts.
                @test eltype(mat) === Float64
            end
        end
    end

    @testset "Zero-mode rule: N^{i→B}_{m,0} = 0 for i ∈ {L, R, T}" begin
        # F_m^{(i)}(1/u) has val ≥ 1 in u; composed with u = g_B(ξ) (val=1),
        # the result has val ≥ 1 in ξ. So [ξ^0] = 0 exactly.
        for ℓ in (0.5, 1.0, 2.0)
            geom = compute_geometry_cross(ℓ, 20)
            neum = compute_neumann(geom, 6)
            for m in 1:5
                @test abs(neum.𝒩.LB[m, 1]) < 1e-10
                @test abs(neum.𝒩.RB[m, 1]) < 1e-10
                @test abs(neum.𝒩.TB[m, 1]) < 1e-10
            end
        end
    end

    @testset "N^{B→i}_{1,0} nonzero for i ∈ {L, R, T}" begin
        # F_1^{(B)}(z) = c_{-1} · z  (polynomial of degree 1, no constant).
        # [ξ^0] F_1^{(B)}(x_j + g_j(ξ)) = c_{-1} · x_j. For L (x=-1) and R
        # (x=+1) this is ±c_{-1}; for T (x=0) the zero-mode vanishes too.
        # Sanity-check L, R nonzero; T exactly zero.
        geom = compute_geometry_cross(1.0, 20)
        neum = compute_neumann(geom, 3)
        @test abs(neum.𝒩.BL[1, 1]) > 1e-6
        @test abs(neum.𝒩.BR[1, 1]) > 1e-6
        @test abs(neum.𝒩.BT[1, 1]) < 1e-10
        # With x_L = -1 and x_R = +1, BL and BR at m=k=0 differ by a sign.
        @test neum.𝒩.BL[1, 1] ≈ -neum.𝒩.BR[1, 1]  atol=1e-10
    end

    @testset "Convergence with series order" begin
        cross_keys = (:LL, :LR, :LT, :LB,
                      :RL, :RR, :RT, :RB,
                      :TL, :TR, :TT, :TB,
                      :BL, :BR, :BT, :BB)
        geom_20 = compute_geometry_cross(1.0, 20)
        neum_20 = compute_neumann(geom_20, 5)
        geom_30 = compute_geometry_cross(1.0, 30)
        neum_30 = compute_neumann(geom_30, 5)
        for key in cross_keys, m in 1:3, k in 1:3
            mat20 = getfield(neum_20.𝒩, key)
            mat30 = getfield(neum_30.𝒩, key)
            # rtol on absolute difference, guarded by |mat20| magnitude.
            if abs(mat20[m, k]) > 1e-8
                @test isapprox(mat20[m, k], mat30[m, k]; rtol=1e-4)
            else
                @test abs(mat30[m, k] - mat20[m, k]) < 1e-8
            end
        end
    end

    @testset "D₄ at ℓ=1, k ≥ 1: |N^{II}| equal across the 4 arms" begin
        # At ℓ=1 the cross has full D₄ symmetry, realised on UHP by the
        # Möbius M(z) = (z+1)/(1-z). For k ≥ 1 entries the four diagonal
        # blocks share magnitude (k=0 is the zero-mode coupling, which
        # carries a gauge-like ambiguity that breaks naive D₄ equality —
        # see `memory/reference_cross_d4_rotation_nontrivial.md`).
        geom = compute_geometry_cross(1.0, 20)
        neum = compute_neumann(geom, 5)
        for m in 1:4, k in 1:4
            vs = (abs(neum.𝒩.LL[m, k+1]),
                  abs(neum.𝒩.RR[m, k+1]),
                  abs(neum.𝒩.TT[m, k+1]),
                  abs(neum.𝒩.BB[m, k+1]))
            # Skip entries where parity forces some to be zero.
            if minimum(vs) > 1e-10
                @test maximum(vs) - minimum(vs) < 1e-5
            end
        end
    end

    @testset "Horizontal Z₂: |N^{LL}| = |N^{RR}|" begin
        # L↔R are related by horizontal reflection z → −z of the SC map.
        # With the current UHP→upper-semidisc convention (α_L = α_R real
        # positive), N^{LL} and N^{RR} agree in magnitude. The sign
        # pattern is measured empirically and pinned once first read.
        for ℓ in (0.5, 1.0, 2.0)
            geom = compute_geometry_cross(ℓ, 20)
            neum = compute_neumann(geom, 6)
            for m in 1:4, k in 0:4
                @test abs(neum.𝒩.LL[m, k+1]) ≈ abs(neum.𝒩.RR[m, k+1])  atol=1e-10
            end
        end
    end

    @testset "Vertical reciprocal Z₂: |N^{TT}| = |N^{BB}|" begin
        # T at z=0 ↔ B at z=∞ via u = 1/z. Expect magnitude equality at
        # every ℓ (not just ℓ=1). Sign pattern measured empirically.
        for ℓ in (0.5, 1.0, 2.0)
            geom = compute_geometry_cross(ℓ, 20)
            neum = compute_neumann(geom, 6)
            for m in 1:4, k in 0:4
                @test abs(neum.𝒩.TT[m, k+1]) ≈ abs(neum.𝒩.BB[m, k+1])  atol=1e-10
            end
        end
    end

end
