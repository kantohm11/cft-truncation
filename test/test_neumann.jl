using Test
using CFTTruncation: compute_geometry, compute_neumann, compose,
                     TruncLaurent, singular_part

@testset "NeumannCoefficients" begin

    @testset "4.1 Diagonal self-consistency" begin
        geom = compute_geometry(1.0, 15)
        neum = compute_neumann(geom, 10)

        # For the diagonal case (i == j), the Neumann coefficients
        # N^{ii}_{m,k} for k ≥ 0 are the regular part of F_m(g_i(ξ)).
        # The intermediate powers ξ^{-m+1},...,ξ^{-1} must be zero
        # by construction. We verify this via the internal compose.
        for arm in (:L, :R, :T)
            Fm_polys = neum.F_polys[arm]
            a = getfield(geom.arms, arm)
            for m in 1:5
                # Use the internal _compose_Fm_with_g
                composed = CFTTruncation._compose_Fm_with_g(
                    Fm_polys[m], a, a, geom.order)
                # ξ^{-m} coefficient should be 1
                @test composed[-m] ≈ 1.0 atol=1e-10
                # Intermediate powers should be zero
                for k in (-m+1):(-1)
                    @test abs(composed[k]) < 1e-10
                end
            end
        end
    end

    @testset "4.2 Z₂ symmetry" begin
        # TODO: Fix Z₂ symmetry test after verifying σ conventions
        # in the L-arm expansion. The Neumann coefficients should satisfy
        # N^{LL} = N^{RR} and N^{LR} = N^{RL} after proper accounting
        # of the arm direction signs.
        @test_skip "Z₂ symmetry deferred"
    end

    @testset "4.3 Decay of off-diagonal coefficients" begin
        # TODO: Verify decay properties once Z₂ conventions are settled.
        # The Neumann coefficients should decay for large m or k
        # at a rate controlled by |x_i - x_j|.
        @test_skip "Decay test deferred"
    end

    @testset "4.4 Convergence with series order" begin
        geom_10 = compute_geometry(1.0, 12)
        neum_10 = compute_neumann(geom_10, 5)

        geom_20 = compute_geometry(1.0, 22)
        neum_20 = compute_neumann(geom_20, 5)

        for m in 1:3, k in 1:3
            @test neum_10.𝒩.LT[m, k] ≈ neum_20.𝒩.LT[m, k] rtol=1e-6
        end
    end

end
