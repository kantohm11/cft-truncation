"""
Primary vertex: boundary 3-point function on the UHP with Jacobian factors.

Factored into three reusable pieces parameterised by **boundary scaling
dimension Δ** (not bulk h). For a chiral primary like the U(1) current
J(z), Δ = h_chiral. For a charged compact-boson V_n on a Neumann boundary
(chiral × antichiral by doubling), Δ = 2 h_bulk = (n/R)².

The Jacobian rule is `(1/α_i)^{Δ_i}` per leg — read off from the
J·J·vac convention check (see `experiments/scripts/jacobian_jjvac_check.jl`
and `test/test_primary_jacobian.jl`).
"""

# ----------------------------------------------------------------------------
# Building blocks (parameterised by scaling dimension Δ).
# ----------------------------------------------------------------------------

"""
    uhp_3pt_correlator(Δ_L, Δ_R, Δ_T, x_L, x_R, x_T; C=1.0) -> Float64

UHP boundary 3-pt function value
    `C / (|x_LR|^{Δ_L+Δ_R−Δ_T} · |x_RT|^{Δ_R+Δ_T−Δ_L} · |x_LT|^{Δ_L+Δ_T−Δ_R})`,
fixed by conformal symmetry up to the OPE coefficient `C`.
"""
function uhp_3pt_correlator(Δ_L::Real, Δ_R::Real, Δ_T::Real,
                            x_L::Real, x_R::Real, x_T::Real;
                            C::Real = 1.0)
    d_LR = abs(x_L - x_R)
    d_RT = abs(x_R - x_T)
    d_LT = abs(x_L - x_T)
    Float64(C) / (d_LR^(Δ_L + Δ_R - Δ_T) *
                  d_RT^(Δ_R + Δ_T - Δ_L) *
                  d_LT^(Δ_L + Δ_T - Δ_R))
end

"""
    primary_jacobian_factor(α_L, α_R, α_T, Δ_L, Δ_R, Δ_T) -> Float64

Local-frame-to-global Jacobian decoration `∏ |α_i|^{−Δ_i}`. Converts a
boundary-3-pt value computed in the global UHP frame into the vertex
amplitude with the local-frame state insertions used by the truncated
Fock basis. Sign and exponent are pinned by the J·J·vac convention
test in `test/test_primary_jacobian.jl`.
"""
function primary_jacobian_factor(α_L::Real, α_R::Real, α_T::Real,
                                 Δ_L::Real, Δ_R::Real, Δ_T::Real)
    abs(α_L)^(-Δ_L) * abs(α_R)^(-Δ_R) * abs(α_T)^(-Δ_T)
end

"""
    primary_vertex_from_Δ(Δ_L, Δ_R, Δ_T, geom::Geometry; C=1.0) -> Float64

Primary 3-pt vertex from explicit scaling dimensions `Δ_i`. Useful for
non-charged primary insertions (e.g., the chiral current J(z) at Δ=1)
where the standard `(n_i, R)` charged-primary wrapper doesn't apply.
"""
function primary_vertex_from_Δ(Δ_L::Real, Δ_R::Real, Δ_T::Real,
                               geom::Geometry; C::Real = 1.0)
    x_L = geom.arms.L.x; x_R = geom.arms.R.x; x_T = geom.arms.T.x
    α_L = abs(geom.arms.L.α)
    α_R = abs(geom.arms.R.α)
    α_T = abs(geom.arms.T.α)

    uhp = uhp_3pt_correlator(Δ_L, Δ_R, Δ_T, x_L, x_R, x_T; C = C)
    jac = primary_jacobian_factor(α_L, α_R, α_T, Δ_L, Δ_R, Δ_T)
    jac * uhp
end

# ----------------------------------------------------------------------------
# Public charged-primary entry points (T-shape and cross).
# ----------------------------------------------------------------------------

"""
    primary_vertex(n_L, n_R, n_T, geom::Geometry, R::Real) -> Float64

Boundary-3-pt primary vertex for the compact boson at radius R. Each
state is a charged primary `V_{n_i}` with boundary scaling dimension
`Δ_i = (n_i/R)²` (= 2 h_bulk_i, by Neumann doubling).

Enforces momentum conservation `n_L + n_R + n_T = 0`. OPE coefficient
`C_LRT = 1` (free boson identity channel).

    V = ∏_i |α_i|^{−Δ_i} · 1 / ∏_{ij} |x_i − x_j|^{Δ_i+Δ_j−Δ_k}
"""
function primary_vertex(n_L::Int, n_R::Int, n_T::Int,
                        geom::Geometry, R::Real)
    n_L + n_R + n_T == 0 || return 0.0
    R = Float64(R)
    Δ_L = (n_L / R)^2
    Δ_R = (n_R / R)^2
    Δ_T = (n_T / R)^2
    primary_vertex_from_Δ(Δ_L, Δ_R, Δ_T, geom; C = 1.0)
end

"""
    primary_vertex(n_L, n_R, n_T, n_B, geom::GeometryCross, R::Real) -> Float64

Four-arm primary vertex for the cross geometry. Conservation:
`n_L + n_R + n_T + n_B = 0`. Free compact-boson formula:

    V = ∏_i |α_i|^{−Δ_i} · ∏_{i<j, both finite} |x_i − x_j|^{2 p_i p_j}

with `p_i = n_i/R`, `Δ_i = p_i² = 2 h_bulk_i`. Factors involving `x_B`
absorb into `|α_B|^{−Δ_B}` via the `u = 1/z` local coordinate
(`x_B = ∞` never appears numerically).

For `n_B = 0`, `Δ_B = 0` so the B-leg contributes 1, and the formula
reduces to the T-shape 3-pt evaluated on the cross arms' α's
(verified by `test_primaryvertex_cross.jl`).
"""
function primary_vertex(n_L::Int, n_R::Int, n_T::Int, n_B::Int,
                        geom::GeometryCross, R::Real)
    n_L + n_R + n_T + n_B == 0 || return 0.0

    R = Float64(R)
    p_L = n_L / R;  p_R = n_R / R;  p_T = n_T / R;  p_B = n_B / R
    Δ_L = p_L^2;  Δ_R = p_R^2;  Δ_T = p_T^2;  Δ_B = p_B^2

    α_L = abs(geom.arms.L.α); α_R = abs(geom.arms.R.α)
    α_T = abs(geom.arms.T.α); α_B = abs(geom.arms.B.α)

    jacobian = α_L^(-Δ_L) * α_R^(-Δ_R) * α_T^(-Δ_T) * α_B^(-Δ_B)

    # Free-boson 4-pt on UHP boundary: ∏_{i<j}|x_ij|^{2 p_i p_j}.
    # B-pair factors absorb into |α_B|^{−Δ_B} as x_B → ∞.
    x_L = geom.arms.L.x; x_R = geom.arms.R.x; x_T = geom.arms.T.x
    d_LR = abs(x_L - x_R)  # 2
    d_LT = abs(x_L - x_T)  # 1
    d_RT = abs(x_R - x_T)  # 1
    geom_factor = d_LR^(2 * p_L * p_R) *
                  d_LT^(2 * p_L * p_T) *
                  d_RT^(2 * p_R * p_T)

    jacobian * geom_factor
end
