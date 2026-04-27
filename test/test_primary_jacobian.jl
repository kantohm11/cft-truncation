using Test
using CFTTruncation
using CFTTruncation: compute_geometry, compute_neumann, compute_vertex,
                     uhp_3pt_correlator, primary_jacobian_factor,
                     primary_vertex_from_Δ, primary_vertex,
                     CompactBosonCFT, charge_block

@testset "Primary-vertex Jacobian convention check" begin
    # The U(1) current J(z) is a Virasoro primary of scaling dimension Δ=1,
    # lying entirely in the u(1) vacuum module. By Virasoro-primary
    # universality, the Jacobian rule that `primary_vertex` implements for
    # charged primaries (with Δ = 2 h_bulk = (n/R)²) should equally well
    # predict V(J·, J·, vac) when `primary_vertex_from_Δ` is called with
    # Δ = 1 for each J insertion and Δ = 0 for the identity.
    #
    # The recursion computes V(J·, J·, vac) using only the Neumann
    # coefficients and `primary_vertex(0, 0, 0) = 1` — convention-independent
    # at Δ = 0. So the recursion's value is taken as the empirical truth.
    #
    # The convention `Jac_i = (1/α_i)^{Δ_i}` was read off from the recursion
    # and the boundary 2-pt of J (see experiments/scripts/jacobian_jjvac_check.jl):
    # V·(α_i α_j) = c_J / |x_i − x_j|² with c_J = +1 holds to machine
    # precision across ℓ. The three cyclic checks below — J·J·vac, J·vac·J,
    # vac·J·J — verify this through the factored helper.

    cft = CompactBosonCFT(R = 1.0, h_bond = 6.0, h_phys = 3.0)

    function find_partition_index(states, λ)
        idx = findfirst(p -> p == λ, states)
        idx === nothing && error("Partition $λ not found")
        idx
    end

    for ell in (0.5, 1.0, 2.0)
        vd = compute_vertex(cft, ell; cache = :off, series_order = 20)

        # Indices in the charge-0 sector for vacuum |∅⟩ and J_{−1}|0⟩ ([1]).
        αL_v = find_partition_index(cft.basis_bond.states[0], Int[])
        αL_J = find_partition_index(cft.basis_bond.states[0], [1])
        αR_v = αL_v
        αR_J = αL_J
        αT_v = find_partition_index(cft.basis_phys.states[0], Int[])
        αT_J = find_partition_index(cft.basis_phys.states[0], [1])

        blk = charge_block(vd, 0, 0, 0)   # blk[αL, αR, αT]
        V_jjv_recursion = blk[αL_J, αR_J, αT_v]
        V_jvj_recursion = blk[αL_J, αR_v, αT_J]
        V_vjj_recursion = blk[αL_v, αR_J, αT_J]

        # Predicted via the factored helper: J insertions have scaling
        # dimension Δ = 1 (chiral primary, no doubling), identity Δ = 0.
        V_jjv_pred = primary_vertex_from_Δ(1.0, 1.0, 0.0, vd.geom; C = 1.0)
        V_jvj_pred = primary_vertex_from_Δ(1.0, 0.0, 1.0, vd.geom; C = 1.0)
        V_vjj_pred = primary_vertex_from_Δ(0.0, 1.0, 1.0, vd.geom; C = 1.0)

        @test isapprox(V_jjv_pred, V_jjv_recursion; rtol = 1e-10)
        @test isapprox(V_jvj_pred, V_jvj_recursion; rtol = 1e-10)
        @test isapprox(V_vjj_pred, V_vjj_recursion; rtol = 1e-10)
    end

end
