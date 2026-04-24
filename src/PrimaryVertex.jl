"""
Primary vertex: boundary 3-point function on the UHP with Jacobian factors.
"""

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
    # Momentum conservation
    n_L + n_R + n_T == 0 || return 0.0

    R = Float64(R)
    h_L = (n_L / R)^2 / 2
    h_R = (n_R / R)^2 / 2
    h_T = (n_T / R)^2 / 2

    α_L = abs(geom.arms.L.α)
    α_R = abs(geom.arms.R.α)
    α_T = abs(geom.arms.T.α)

    x_L = geom.arms.L.x  # -1
    x_R = geom.arms.R.x  #  1
    x_T = geom.arms.T.x  #  0

    jacobian = α_L^(2 * h_L) * α_R^(2 * h_R) * α_T^(2 * h_T)

    # Boundary 3-pt: C / (|x_LR|^{h_L+h_R-h_T} · |x_RT|^{h_R+h_T-h_L} · |x_LT|^{h_L+h_T-h_R})
    # For compact boson C = 1
    d_LR = abs(x_L - x_R)  # 2
    d_RT = abs(x_R - x_T)  # 1
    d_LT = abs(x_L - x_T)  # 1

    denom = d_LR^(h_L + h_R - h_T) * d_RT^(h_R + h_T - h_L) * d_LT^(h_L + h_T - h_R)

    jacobian / denom
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
