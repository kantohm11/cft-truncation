using Test
using CFTTruncation: build_fock_basis, build_bpz_form, build_bpz_map
using TensorKit

@testset "BPZ" begin

    @testset "7.0 BPZ bilinear form η : V⊗V → ℂ" begin
        basis = build_fock_basis(1.0, 3.0)
        η_form = build_bpz_form(basis)
        # Codomain is trivial, domain is V ⊗ V
        @test codomain(η_form) == one(basis.V)
        @test domain(η_form) == basis.V ⊗ basis.V
    end

    @testset "7.1 Conjugation map: diagonal with (-1)^N" begin
        basis = build_fock_basis(1.0, 3.0)
        η = build_bpz_map(basis)

        for (f₁, f₂) in fusiontrees(η)
            n = Int(f₂.uncoupled[1].charge)
            blk = η[f₁, f₂]
            d = size(blk, 1)
            for α in 1:d
                @test blk[α, α] ≈ (-1.0)^basis.levels[n][α]
            end
            for α in 1:d, β in 1:d
                α == β && continue
                @test abs(blk[α, β]) < 1e-15
            end
        end
    end

    @testset "7.2 Specific values at n = 0" begin
        basis = build_fock_basis(1.0, 3.0)
        η = build_bpz_map(basis)
        blk = block(η, U1Irrep(0))
        @test blk[1,1] ≈  1.0   # level 0
        @test blk[2,2] ≈ -1.0   # level 1
        @test blk[3,3] ≈  1.0   # level 2
        @test blk[4,4] ≈  1.0   # level 2
        @test blk[5,5] ≈ -1.0   # level 3
        @test blk[6,6] ≈ -1.0   # level 3
        @test blk[7,7] ≈ -1.0   # level 3
    end

    @testset "7.3 Involution: η² = id" begin
        basis = build_fock_basis(1.0, 3.0)
        η = build_bpz_map(basis)
        # η is V ← V (endomorphism), so η*η should be identity
        η2 = η * η
        @test η2 ≈ id(basis.V) atol=1e-14
    end

end
