using Test
using CFTTruncation: CompactBosonCFT, TruncationSpec,
                     compute_geometry_cross, compute_neumann,
                     compute_vertex, charge_block,
                     primary_vertex
using TensorKit
using LinearAlgebra: norm

@testset "Recursion cross" begin

    # Small CFT for fast tests. h_bond = h_phys = 1.0 gives a handful of
    # non-trivial levels, enough to exercise the Ward recursion.
    cft = CompactBosonCFT(; R=1.0, h_bond=1.0, h_phys=1.0)
    geom = compute_geometry_cross(1.0, 20)
    neum = compute_neumann(geom, cft.m_max)
    vd = compute_vertex(cft, geom, neum; ell=1.0)

    @testset "VertexData for cross has expected fields" begin
        @test vd.cft === cft
        @test vd.geom === geom
        @test vd.neumann === neum
        @test vd.ell == 1.0
    end

    @testset "Level-0 entries match primary_vertex" begin
        # For each charge quadruple with Σn = 0 that has both T and B in
        # phys sectors and L, R in bond sectors, the (α=1,1,1,1) entry (all
        # vacuum states) is the primary vertex.
        for (nL, nR, nT, nB) in ((0, 0, 0, 0), (1, -1, 0, 0), (0, 0, 1, -1),
                                 (1, 0, -1, 0), (1, 0, 0, -1),
                                 (1, 1, -1, -1), (1, -1, 1, -1))
            @assert nL + nR + nT + nB == 0
            blk = charge_block(vd, nL, nR, nT, nB)
            length(blk) == 0 && continue
            expected = primary_vertex(nL, nR, nT, nB, geom, cft.R)
            # blk axes: [αL, αR, αT, αB]
            @test blk[1, 1, 1, 1] ≈ expected  rtol=1e-10
        end
    end

    @testset "Non-conserving charges give empty block" begin
        # The vertex lives in the charge-0 sector; asking for Σn ≠ 0 returns
        # either a structurally-empty block or zeros.
        blk = charge_block(vd, 1, 0, 0, 0)
        @test length(blk) == 0 || all(iszero, blk)
    end

    @testset "All vertex entries are finite" begin
        V = vd.vertex
        for (f1, f2) in TensorKit.fusiontrees(V)
            blk = V[f1, f2]
            @test all(isfinite, Array(blk))
        end
    end

    @testset "Horizontal Z₂: L↔R swap preserves block norm" begin
        # Per-entry the swap involves a sign pattern from the Ward identity
        # (J_0 on the L/R arm changes sign with charge), mirroring the
        # T-shape test 10.2. The block norms must match.
        for (nL, nR, nT, nB) in ((1, -1, 0, 0), (2, -1, -1, 0), (1, -1, 1, -1))
            @assert nL + nR + nT + nB == 0
            blk_LR = charge_block(vd, nL, nR, nT, nB)
            blk_RL = charge_block(vd, nR, nL, nT, nB)
            (length(blk_LR) == 0 || length(blk_RL) == 0) && continue
            @test norm(blk_LR) ≈ norm(blk_RL)  atol=1e-10
        end
    end

    @testset "Vertex TensorMap has expected leg structure" begin
        V = vd.vertex
        # ℂ ← V_phys ⊗ V_phys ⊗ V_bond ⊗ V_bond  (legs: T, B, L, R)
        @test numout(V) == 0
        @test numin(V) == 4
    end

end
