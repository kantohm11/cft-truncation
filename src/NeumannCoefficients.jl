"""
Neumann coefficients N^{i→j}_{m,k} from the functions F_m^{(i)}.
"""

struct NeumannData
    𝒩::NamedTuple{(:LL,:LR,:LT,:RL,:RR,:RT,:TL,:TR,:TT), NTuple{9, Matrix{Float64}}}
    F_polys::NamedTuple{(:L,:R,:T), NTuple{3, Vector{TruncLaurent{ComplexF64}}}}
end

"""
    compute_neumann(geom::Geometry, m_max::Int) -> NeumannData

Compute Neumann coefficients for modes m = 1, ..., m_max.
N^{i→j}_{m,k} is stored as 𝒩.IJ[m, k+1] (1-indexed, k starts at 0).
"""
function compute_neumann(geom::Geometry, m_max::Int)
    arms = (:L, :R, :T)

    # Step 1: Compute F_m polynomials for each arm
    F_polys_dict = Dict{Symbol, Vector{TruncLaurent{ComplexF64}}}()
    for arm in arms
        F_polys_dict[arm] = _compute_F_polys(getfield(geom.arms, arm), m_max)
    end

    # Step 2: Compute Neumann coefficients by composing F_m with g_j
    k_max = m_max  # number of k values: k = 0, 1, ..., k_max-1
    neum_dict = Dict{Symbol, Matrix{Float64}}()

    for i in arms, j in arms
        key = Symbol(i, j)
        mat = zeros(Float64, m_max, k_max)
        a_i = getfield(geom.arms, i)
        a_j = getfield(geom.arms, j)
        for m in 1:m_max
            # F_m^{(i)}(z) is a polynomial in (z - x_i)^{-1} of degree m.
            # To get F_m^{(i)}(g_j(ξ)), substitute z = x_j + g_j_series(ξ),
            # so (z - x_i) = (x_j - x_i) + g_j_series(ξ).
            composed = _compose_Fm_with_g(F_polys_dict[i][m], a_i, a_j, geom.order)
            for k in 0:k_max-1
                mat[m, k+1] = real(composed[k])
            end
        end
        neum_dict[key] = mat
    end

    𝒩 = (
        LL=neum_dict[:LL], LR=neum_dict[:LR], LT=neum_dict[:LT],
        RL=neum_dict[:RL], RR=neum_dict[:RR], RT=neum_dict[:RT],
        TL=neum_dict[:TL], TR=neum_dict[:TR], TT=neum_dict[:TT],
    )
    F_polys = (
        L=F_polys_dict[:L], R=F_polys_dict[:R], T=F_polys_dict[:T],
    )
    NeumannData(𝒩, F_polys)
end

"""
Compute F_m^{(i)}(z) = (f_i(z))^{-m} |_{singular part} for m = 1,...,m_max.

For a finite arm (x_i ∈ ℝ) each F_m is a polynomial in (z − x_i)^{−1}
of degree m. For the cross B arm (x_B = ∞, local coord u = 1/z) the
same computation lives in u, and the "singular part" is Σ c_n u^n for
n = −m,...,−1; reinterpreted in z this is a polynomial in z of
degree m (no constant term).
"""
function _compute_F_polys(a::ArmData, m_max::Int)
    fi = a.f_series  # f_i(local) with val=1

    # f_i^{-1} as a Laurent series (val=-1, prec = fi.prec - 1)
    fi_inv = inv(fi)

    polys = Vector{TruncLaurent{ComplexF64}}(undef, m_max)

    # f_i^{-m}: iteratively multiply by f_i^{-1}
    current_inv_power = fi_inv  # f_i^{-1}, val = -1
    for m in 1:m_max
        polys[m] = singular_part(current_inv_power)
        if m < m_max
            current_inv_power = _truncmul(current_inv_power, fi_inv, fi.prec)
        end
    end

    polys
end

"""
Compose F_m^{(i)} (a polynomial in (z-x_i)^{-1} of degree m) with z = x_j + g_j(ξ).

For j == i: (z - x_i) = g_i(ξ), and F_m composed with g_i gives ξ^{-m} + O(ξ^0).
For j ≠ i: (z - x_i) = (x_j - x_i) + g_j(ξ), which is a regular series starting at O(ξ^0).
"""
function _compose_Fm_with_g(Fm::TruncLaurent{ComplexF64}, arm_i::ArmData, arm_j::ArmData, order::Int)
    prec = order
    if arm_i.label == arm_j.label
        # Same arm: F_m is a polynomial in ζ^{-1} (or u^{-1} for B arm)
        # and the inner variable is g_j(ξ). F_m(g_j(ξ)) = Σ c_n (g_j(ξ))^n.
        g = arm_j.g_series
        result = TruncLaurent(0, zeros(ComplexF64, prec), prec)
        ginv = inv(g)
        for n in 1:length(Fm.coeffs)
            power = Fm.val + n - 1
            neg_power = -power
            g_to_power = _power_of(ginv, neg_power, prec)
            coeff = Fm.coeffs[n]
            if coeff != zero(ComplexF64)
                scaled = TruncLaurent(g_to_power.val,
                    ComplexF64[coeff * c for c in g_to_power.coeffs], prec)
                result = _add(result, scaled)
            end
        end
        return result

    elseif arm_i.label == :B
        # B source, finite target. F_m^{(B)} is stored as a Laurent in u
        # with val = -m. In z (via u = 1/z) it is a polynomial of degree m
        # with no constant term:  F_m^{(B)}(z) = Σ_{n=-m}^{-1} c_n z^{|n|}.
        # Evaluate at z = x_j + g_j(ξ), a val=0 series.
        g = arm_j.g_series
        z_coeffs = zeros(ComplexF64, prec)
        z_coeffs[1] = ComplexF64(arm_j.x)
        for k in 1:min(length(g.coeffs), prec - 1)
            z_coeffs[k + 1] = g.coeffs[k]
        end
        z_series = TruncLaurent(0, z_coeffs, prec)
        result = TruncLaurent(0, zeros(ComplexF64, prec), prec)
        for n in 1:length(Fm.coeffs)
            power = Fm.val + n - 1        # -m, -m+1, ..., -1
            exponent = -power             # m, m-1, ..., 1
            coeff = Fm.coeffs[n]
            if coeff != zero(ComplexF64)
                z_to_exp = _power_of(z_series, exponent, prec)
                scaled = TruncLaurent(z_to_exp.val,
                    ComplexF64[coeff * c for c in z_to_exp.coeffs], prec)
                result = _add(result, scaled)
            end
        end
        return result

    elseif arm_j.label == :B
        # Finite source, B target. F_m^{(i)}(z) = Σ c_n (z − x_i)^n for
        # n = −m,…,−1, evaluated at z = 1/u with u = g_B(ξ).
        # (z − x_i)^{-|n|} = [u/(1 − x_i u)]^{|n|} = u^{|n|} · (1 − x_i u)^{-|n|}.
        # So the result has val ≥ 1 in u, and after u = g_B(ξ) also val ≥ 1 in ξ.
        g_B = arm_j.g_series            # val=1
        x_i = ComplexF64(arm_i.x)

        # base(ξ) = 1 − x_i · g_B(ξ), val=0 with constant term 1.
        base_coeffs = zeros(ComplexF64, prec)
        base_coeffs[1] = one(ComplexF64)
        for k in 1:min(length(g_B.coeffs), prec - 1)
            base_coeffs[k + 1] = -x_i * g_B.coeffs[k]
        end
        base = TruncLaurent(0, base_coeffs, prec)
        base_inv = inv(base)

        result = TruncLaurent(0, zeros(ComplexF64, prec), prec)
        for n in 1:length(Fm.coeffs)
            power = Fm.val + n - 1        # -m, ..., -1
            exponent = -power             # m, ..., 1
            coeff = Fm.coeffs[n]
            coeff == zero(ComplexF64) && continue
            g_pow = _power_of(g_B, exponent, prec)
            binv_pow = _power_of(base_inv, exponent, prec)
            term = _truncmul(g_pow, binv_pow, prec)
            scaled = TruncLaurent(term.val,
                ComplexF64[coeff * c for c in term.coeffs], prec)
            result = _add(result, scaled)
        end
        return result

    else
        # Two distinct finite arms.
        Δx = ComplexF64(arm_j.x - arm_i.x)
        g = arm_j.g_series
        w_coeffs = zeros(ComplexF64, prec)
        w_coeffs[1] = Δx
        for k in 1:min(length(g.coeffs), prec - 1)
            w_coeffs[k + 1] = g.coeffs[k]
        end
        w = TruncLaurent(0, w_coeffs, prec)
        winv = inv(w)
        result = TruncLaurent(0, zeros(ComplexF64, prec), prec)
        for n in 1:length(Fm.coeffs)
            power = Fm.val + n - 1
            neg_power = -power
            w_to_neg_power = _power_of(winv, neg_power, prec)
            coeff = Fm.coeffs[n]
            if coeff != zero(ComplexF64)
                scaled = TruncLaurent(w_to_neg_power.val,
                    ComplexF64[coeff * c for c in w_to_neg_power.coeffs], prec)
                result = _add(result, scaled)
            end
        end
        return result
    end
end

function _power_of(s::TruncLaurent{T}, n::Int, prec::Int) where T
    n == 0 && return TruncLaurent(0, T[one(T)], prec)
    n == 1 && return s
    result = s
    for _ in 2:n
        result = _truncmul(result, s, prec)
    end
    result
end


# ============================================================================
# Cross-shape (4-arm) Neumann coefficients.
# ============================================================================

const _CROSS_ARM_KEYS = (:LL, :LR, :LT, :LB,
                         :RL, :RR, :RT, :RB,
                         :TL, :TR, :TT, :TB,
                         :BL, :BR, :BT, :BB)

struct NeumannDataCross
    𝒩::NamedTuple{_CROSS_ARM_KEYS, NTuple{16, Matrix{Float64}}}
    F_polys::NamedTuple{(:L,:R,:T,:B), NTuple{4, Vector{TruncLaurent{ComplexF64}}}}
end

"""
    compute_neumann(geom::GeometryCross, m_max::Int) -> NeumannDataCross

Compute Neumann coefficients for the cross geometry's 4 arms, yielding
a 4×4 = 16-matrix family N^{i→j}_{m,k} (i, j ∈ {L, R, T, B}). The B arm
is handled via its u = 1/z local coordinate: F_m^{(B)} lives as a
Laurent in u with val = −m, and off-diagonal entries touching B use the
1/u coordinate change — see `_compose_Fm_with_g` for the four dispatch
cases.

Returns a `NeumannDataCross` with `𝒩.IJ[m, k+1]` giving N^{i→j}_{m,k},
and `F_polys.I[m]` the F_m singular polynomial for arm i.
"""
function compute_neumann(geom::GeometryCross, m_max::Int)
    arms = (:L, :R, :T, :B)

    F_polys_dict = Dict{Symbol, Vector{TruncLaurent{ComplexF64}}}()
    for arm in arms
        F_polys_dict[arm] = _compute_F_polys(getfield(geom.arms, arm), m_max)
    end

    k_max = m_max
    neum_dict = Dict{Symbol, Matrix{Float64}}()

    for i in arms, j in arms
        key = Symbol(i, j)
        mat = zeros(Float64, m_max, k_max)
        a_i = getfield(geom.arms, i)
        a_j = getfield(geom.arms, j)
        for m in 1:m_max
            composed = _compose_Fm_with_g(F_polys_dict[i][m], a_i, a_j, geom.order)
            for k in 0:k_max-1
                mat[m, k+1] = real(composed[k])
            end
        end
        neum_dict[key] = mat
    end

    𝒩 = NamedTuple{_CROSS_ARM_KEYS}(ntuple(i -> neum_dict[_CROSS_ARM_KEYS[i]], 16))
    F_polys = (L=F_polys_dict[:L], R=F_polys_dict[:R],
               T=F_polys_dict[:T], B=F_polys_dict[:B])
    NeumannDataCross(𝒩, F_polys)
end
