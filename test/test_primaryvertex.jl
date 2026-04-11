using Test
using CFTTruncation: compute_geometry, primary_vertex

@testset "PrimaryVertex" begin

    @testset "8.1 Selection rule" begin
        geom = compute_geometry(1.0, 5)
        R = 1.0
        @test primary_vertex(1, -1, 0, geom, R) != 0
        @test primary_vertex(0, 0, 0, geom, R) != 0
        @test primary_vertex(1, 0, 0, geom, R) == 0
        @test primary_vertex(1, 1, 1, geom, R) == 0
    end

    @testset "8.2 All-zero momenta" begin
        geom = compute_geometry(1.0, 5)
        @test primary_vertex(0, 0, 0, geom, 1.0) ≈ 1.0
    end

    @testset "8.3 Non-trivial primary vertex" begin
        geom = compute_geometry(1.0, 5)
        α_L = geom.arms.L.α
        α_R = geom.arms.R.α
        # V(1,-1,0) = |α_L|·|α_R| / |x_L - x_R|^{2h_L} where h_L = 1/2
        # |x_L - x_R| = 2, so the factor is 1/2
        @test primary_vertex(1, -1, 0, geom, 1.0) ≈ abs(α_L) * abs(α_R) / 2
    end

    @testset "8.4 L↔R symmetry" begin
        geom = compute_geometry(1.0, 5)
        @test primary_vertex(1, -1, 0, geom, 1.0) ≈
              primary_vertex(-1, 1, 0, geom, 1.0)
    end

end
