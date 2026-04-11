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
