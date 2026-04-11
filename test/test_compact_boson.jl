using Test
using CFTTruncation: CompactBosonCFT, TruncationSpec
using TensorKit: dim

@testset "CompactBosonCFT" begin

    @testset "Constructor with TruncationSpec" begin
        cft = CompactBosonCFT(R=1.0, trunc=TruncationSpec(2.0))
        @test cft.R == 1.0
        @test cft.trunc.h_bond == 2.0
        @test cft.trunc.h_phys == 2.0
        @test cft.m_max == 4   # max(2,2) + 2

        # basis dims should match a freshly built FockBasis(R, h_max)
        @test dim(cft.basis_bond.V) > 0
        @test dim(cft.basis_phys.V) > 0
    end

    @testset "Constructor with h_bond, h_phys" begin
        cft = CompactBosonCFT(R=1.0, h_bond=2.0, h_phys=3.0)
        @test cft.trunc.h_bond == 2.0
        @test cft.trunc.h_phys == 3.0
        @test cft.m_max == 5   # max(2,3) + 2

        # bond and phys bases differ when truncations differ
        @test dim(cft.basis_bond.V) != dim(cft.basis_phys.V)
    end

    @testset "Two CFTs with same args are equivalent" begin
        cft1 = CompactBosonCFT(R=1.0, trunc=TruncationSpec(2.0))
        cft2 = CompactBosonCFT(R=1.0, trunc=TruncationSpec(2.0))

        # Field-by-field structural equivalence
        @test cft1.R == cft2.R
        @test cft1.m_max == cft2.m_max
        @test dim(cft1.basis_bond.V) == dim(cft2.basis_bond.V)
        @test dim(cft1.basis_phys.V) == dim(cft2.basis_phys.V)

        # J matrices should match (same n keys, same per-sector matrices)
        @test keys(cft1.J_bond) == keys(cft2.J_bond)
        for n in keys(cft1.J_bond)
            for k in 1:length(cft1.J_bond[n])
                @test cft1.J_bond[n][k] == cft2.J_bond[n][k]
            end
        end
    end

    @testset "Constructor errors" begin
        # Missing both trunc and h_bond/h_phys
        @test_throws ErrorException CompactBosonCFT(R=1.0)
        # Only one of h_bond/h_phys given
        @test_throws ErrorException CompactBosonCFT(R=1.0, h_bond=2.0)
    end

    @testset "BPZ form is built and has trivial codomain" begin
        cft = CompactBosonCFT(R=1.0, trunc=TruncationSpec(2.0))
        @test codomain(cft.bpz_bond_form) == one(cft.basis_bond.V)
        @test domain(cft.bpz_bond_form) == cft.basis_bond.V ⊗ cft.basis_bond.V
    end

end
