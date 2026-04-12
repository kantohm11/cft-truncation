using Test
using CFTTruncation: compute_vertex, charge_block, TruncationSpec, primary_vertex,
                     CompactBosonCFT, vertex_sweep
using TensorKit

@testset "Recursion" begin

    # Shared CFT for most tests
    cft2 = CompactBosonCFT(R=1.0, trunc=TruncationSpec(2.0))

    @testset "9.1 Level 0 matches primary vertex" begin
        vd = compute_vertex(cft2, 1.0)

        # V(|0;0⟩, |0;0⟩, |0;0⟩) = 1
        B = charge_block(vd, 0, 0, 0)
        @test B[1, 1, 1] ≈ 1.0

        # Another primary triple: (1,-1,0)
        B2 = charge_block(vd, 1, -1, 0)
        @test B2[1, 1, 1] ≈ primary_vertex(1, -1, 0, vd.geom, 1.0)
    end

    @testset "9.2 V(J₋₁|0⟩, |0⟩, |0⟩) = 0" begin
        vd = compute_vertex(cft2, 1.0)
        B = charge_block(vd, 0, 0, 0)
        @test abs(B[2, 1, 1]) < 1e-12
    end

    @testset "9.3 V(J₋₁|0⟩, J₋₁|0⟩, |0⟩) = -N^{LR}_{1,1}" begin
        vd = compute_vertex(cft2, 1.0)
        B = charge_block(vd, 0, 0, 0)
        N_LR_11 = vd.neumann.𝒩.LR[1, 2]
        @test B[2, 2, 1] ≈ -N_LR_11 atol=1e-12
    end

    @testset "9.4 V(J₋₁|1⟩, |-1⟩, |0⟩) Ward identity" begin
        vd = compute_vertex(cft2, 1.0)
        R = 1.0
        V_prim = primary_vertex(1, -1, 0, vd.geom, R)
        N_LL_10 = vd.neumann.𝒩.LL[1, 1]
        N_LR_10 = vd.neumann.𝒩.LR[1, 1]
        expected = -(N_LL_10 / R - N_LR_10 / R) * V_prim

        B = charge_block(vd, 1, -1, 0)
        @test B[2, 1, 1] ≈ expected atol=1e-12
    end

    @testset "9.5 Tensor structure" begin
        vd = compute_vertex(cft2, 1.0)
        @test codomain(vd.vertex) == one(vd.cft.basis_phys.V)
        @test domain(vd.vertex) == vd.cft.basis_phys.V ⊗ vd.cft.basis_bond.V ⊗ vd.cft.basis_bond.V
    end

    @testset "9.6 Norm grows with truncation" begin
        cft1 = CompactBosonCFT(R=1.0, trunc=TruncationSpec(1.0))
        vd1 = compute_vertex(cft1, 1.0)
        vd2 = compute_vertex(cft2, 1.0)
        @test norm(vd1.vertex) > 0
        @test norm(vd2.vertex) > norm(vd1.vertex)
    end

    @testset "10.1 ell-dependence of primary vertex" begin
        vals = Float64[]
        for ell in [0.5, 1.0, 1.5, 2.0]
            vd = compute_vertex(cft2, ell)
            B = charge_block(vd, 1, -1, 0)
            push!(vals, B[1, 1, 1])
        end
        @test all(isfinite, vals)
        @test all(!iszero, vals)
    end

    @testset "10.2 Z₂ symmetry of vertex (L ↔ R)" begin
        for ell in [0.5, 1.0, 2.0]
            vd = compute_vertex(cft2, ell)
            B1 = charge_block(vd, 1, -1, 0)
            B2 = charge_block(vd, -1, 1, 0)
            @test norm(B1) ≈ norm(B2) atol=1e-10
        end
    end

    @testset "Sweep: build CFT once, compute across ℓ" begin
        cft = CompactBosonCFT(R=1.0, trunc=TruncationSpec(2.0))
        ells = [0.5, 1.0, 1.5, 2.0]
        vds = [compute_vertex(cft, ell) for ell in ells]
        for (i, ell) in enumerate(ells)
            vd = vds[i]
            @test vd.ell == ell
            @test vd.cft === cft
            @test vd.cft.R == 1.0
            @test dim(vd.cft.basis_bond.V) == dim(cft.basis_bond.V)
        end
    end

    @testset "vertex_sweep convenience" begin
        cft = CompactBosonCFT(R=1.0, trunc=TruncationSpec(2.0))
        vds = vertex_sweep(cft, [0.5, 1.0, 1.5])
        @test length(vds) == 3
        @test [vd.ell for vd in vds] == [0.5, 1.0, 1.5]
        @test all(vd.cft === cft for vd in vds)
    end

    @testset "Vertex values consistent across ℓ sweep" begin
        cft = CompactBosonCFT(R=1.0, trunc=TruncationSpec(2.0))
        for ell in [0.5, 1.0, 1.5, 2.0]
            vd1 = compute_vertex(cft, ell)
            vd2 = compute_vertex(cft, ell)
            for n_L in -2:2, n_R in -2:2
                n_T = -(n_L + n_R)
                B1 = charge_block(vd1, n_L, n_R, n_T)
                B2 = charge_block(vd2, n_L, n_R, n_T)
                @test size(B1) == size(B2)
                isempty(B1) && continue
                @test B1 ≈ B2 atol=1e-14
            end
        end
    end

end
