"""
Primary vertex: boundary 3-point function on the UHP with Jacobian factors.

Factored into three reusable pieces so that test vectors with non-charged
primary insertions (e.g., the U(1) current `J(z)` at chiral weight 1) can
be predicted with the same machinery and compared to the Ward-recursion
output. See `test/test_primary_jacobian.jl` for the convention check.
"""

# ----------------------------------------------------------------------------
# Building blocks (preserve current numerical behaviour exactly).
# ----------------------------------------------------------------------------

"""
    uhp_3pt_correlator(h_L, h_R, h_T, x_L, x_R, x_T; C=1.0) -> Float64

UHP boundary-3-pt-function value `C / (|x_LR|^{h_L+h_R-h_T} ·
|x_RT|^{h_R+h_T-h_L} · |x_LT|^{h_L+h_T-h_R})`. The exponents use the
weights `h_i` directly (the chiral pattern), with `C` the OPE coefficient
(default 1 for the compact-boson identity channel).
"""
function uhp_3pt_correlator(h_L::Real, h_R::Real, h_T::Real,
                            x_L::Real, x_R::Real, x_T::Real;
                            C::Real = 1.0)
    d_LR = abs(x_L - x_R)
    d_RT = abs(x_R - x_T)
    d_LT = abs(x_L - x_T)
    Float64(C) / (d_LR^(h_L + h_R - h_T) *
                  d_RT^(h_R + h_T - h_L) *
                  d_LT^(h_L + h_T - h_R))
end

"""
    primary_jacobian_factor(α_L, α_R, α_T, h_L, h_R, h_T) -> Float64

The local-frame-to-global Jacobian decoration `∏ |α_i|^{2 h_i}` applied
to a UHP boundary correlator to produce the vertex amplitude. This is
the **current convention**; the sign and magnitude of the exponent are
the subject of an ongoing convention check (see
`test/test_primary_jacobian.jl`). Independent of how this is interpreted
physically, the formula is fixed here so that downstream code keeps
producing identical numbers to before this refactor.
"""
function primary_jacobian_factor(α_L::Real, α_R::Real, α_T::Real,
                                 h_L::Real, h_R::Real, h_T::Real)
    abs(α_L)^(2 * h_L) * abs(α_R)^(2 * h_R) * abs(α_T)^(2 * h_T)
end

"""
    primary_vertex_from_h(h_L, h_R, h_T, geom::Geometry; C=1.0) -> Float64

Primary 3-pt vertex from explicit conformal weights. Combines
`uhp_3pt_correlator` and `primary_jacobian_factor` using the geometry's
α's and x's. Useful for non-charged primary insertions (e.g., chiral
currents `J(z)` at h=1) where the standard `(n_i, R)`-driven wrapper
doesn't directly apply.
"""
function primary_vertex_from_h(h_L::Real, h_R::Real, h_T::Real,
                               geom::Geometry; C::Real = 1.0)
    x_L = geom.arms.L.x; x_R = geom.arms.R.x; x_T = geom.arms.T.x
    α_L = abs(geom.arms.L.α)
    α_R = abs(geom.arms.R.α)
    α_T = abs(geom.arms.T.α)

    uhp = uhp_3pt_correlator(h_L, h_R, h_T, x_L, x_R, x_T; C = C)
    jac = primary_jacobian_factor(α_L, α_R, α_T, h_L, h_R, h_T)
    jac * uhp
end

# ----------------------------------------------------------------------------
# Public charged-primary entry points (T-shape and cross).
# ----------------------------------------------------------------------------

"""
    primary_vertex(n_L, n_R, n_T, geom::Geometry, R::Real) -> Float64

Evaluate the primary 3-point vertex V_ℓ^{prim}(h_L, h_R, h_T) for the
compact boson at radius R. Each state is a primary with momentum n_i,
and conformal weight h_i = (n_i/R)²/2.

Enforces momentum conservation n_L + n_R + n_T = 0. For the compact
boson, the OPE coefficient C_LRT = 1 (free boson).

Formula: V = ∏_i |α_i|^{2h_i} · 1 / (|x_L-x_R|^{h_L+h_R-h_T}
                                     · |x_R-x_T|^{h_R+h_T-h_L}
                                     · |x_L-x_T|^{h_L+h_T-h_R})
"""
function primary_vertex(n_L::Int, n_R::Int, n_T::Int, geom::Geometry, R::Real)
    n_L + n_R + n_T == 0 || return 0.0

    R = Float64(R)
    h_L = (n_L / R)^2 / 2
    h_R = (n_R / R)^2 / 2
    h_T = (n_T / R)^2 / 2

    primary_vertex_from_h(h_L, h_R, h_T, geom; C = 1.0)
end

"""
    primary_vertex(n_L, n_R, n_T, n_B, geom::GeometryCross, R::Real) -> Float64

Four-arm primary vertex for the cross geometry. Conservation:
n_L + n_R + n_T + n_B = 0. Formula (free compact boson, boundary
convention matching the T-shape):

    V = ∏_i |α_i|^{2h_i} · ∏_{i<j, both finite} |x_i − x_j|^{p_i p_j}

with p_i = n_i / R, h_i = p_i² / 2. Factors involving x_B would be
|x_B|^{p_B·Σ_{finite} p_i} = |x_B|^{−p_B²} as x_B → ∞; this divergence
is absorbed into the definition of α_B via the u = 1/z local coord
(|α_B|^{2h_B} already encodes the correct asymptotic normalization).

For n_B = 0 the formula reduces to the T-shape answer evaluated on
the cross arms' α's.
"""
function primary_vertex(n_L::Int, n_R::Int, n_T::Int, n_B::Int,
                        geom::GeometryCross, R::Real)
    n_L + n_R + n_T + n_B == 0 || return 0.0

    R = Float64(R)
    p_L = n_L / R;  p_R = n_R / R;  p_T = n_T / R;  p_B = n_B / R
    h_L = p_L^2 / 2;  h_R = p_R^2 / 2;  h_T = p_T^2 / 2;  h_B = p_B^2 / 2

    α_L = abs(geom.arms.L.α); α_R = abs(geom.arms.R.α)
    α_T = abs(geom.arms.T.α); α_B = abs(geom.arms.B.α)

    jacobian = α_L^(2h_L) * α_R^(2h_R) * α_T^(2h_T) * α_B^(2h_B)

    # Finite-pair distances (B at infinity contributes only via |α_B|^{2h_B}).
    x_L = geom.arms.L.x; x_R = geom.arms.R.x; x_T = geom.arms.T.x
    d_LR = abs(x_L - x_R)  # 2
    d_LT = abs(x_L - x_T)  # 1
    d_RT = abs(x_R - x_T)  # 1

    # Free-boson correlator: ∏_{i<j} |x_i − x_j|^{p_i·p_j} (on 3 finite pairs).
    geom_factor = d_LR^(p_L * p_R) * d_LT^(p_L * p_T) * d_RT^(p_R * p_T)

    jacobian * geom_factor
end
