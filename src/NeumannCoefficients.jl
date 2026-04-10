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
        F_polys_dict[arm] = _compute_F_polys(geom, arm, m_max)
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
Each F_m is a polynomial in (z - x_i)^{-1} of degree m.
"""
function _compute_F_polys(geom::Geometry, arm::Symbol, m_max::Int)
    a = getfield(geom.arms, arm)
    fi = a.f_series  # f_i(ζ) with val=1

    # f_i^{-1} as a Laurent series
    fi_inv = _invert_val1_series(fi)

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
        # Same arm: (z - x_i) = g_i(ξ) with val=1
        # F_m is a polynomial in ζ^{-1}, and ζ = g_i(ξ)
        # Compose directly: F_m(g(ξ))
        # F_m has val=-m, prec=0. We need to extend its prec for compose to work.
        # Actually, F_m is a finite polynomial Σ_{n=-m}^{-1} c_n ζ^n
        # F_m(g(ξ)) = Σ c_n (g(ξ))^n
        # g has val=1, so g^{-n} has val=-n. Compute each power separately.
        g = arm_j.g_series
        result = TruncLaurent(0, zeros(ComplexF64, prec), prec)
        ginv = _invert_val1_series(g)
        ginv_power = TruncLaurent(0, ComplexF64[one(ComplexF64)], prec)  # g^0 = 1
        for n in 1:length(Fm.coeffs)
            power = Fm.val + n - 1  # the exponent: -m, -m+1, ..., -1
            neg_power = -power  # positive: m, m-1, ..., 1
            # g^power = ginv^{neg_power}
            g_to_power = _power_of(ginv, neg_power, prec)
            coeff = Fm.coeffs[n]
            if coeff != zero(ComplexF64)
                scaled = TruncLaurent(g_to_power.val,
                    ComplexF64[coeff * c for c in g_to_power.coeffs], prec)
                result = _add(result, scaled)
            end
        end
        return result
    else
        # Different arm: (z - x_i) = (x_j - x_i) + g_j(ξ)
        # This is a series in ξ starting at O(ξ^0) with constant term (x_j - x_i)
        Δx = ComplexF64(arm_j.x - arm_i.x)
        g = arm_j.g_series  # val=1
        # Build w(ξ) = Δx + g(ξ) as a val=0 series
        w_coeffs = zeros(ComplexF64, prec)
        w_coeffs[1] = Δx
        for k in 1:min(length(g.coeffs), prec - 1)
            w_coeffs[k + 1] = g.coeffs[k]
        end
        w = TruncLaurent(0, w_coeffs, prec)
        # F_m(w) = Σ c_n w^n for n = -m,...,-1
        # w has val=0 and nonzero constant term, so w^{-1} = inv(w)
        winv = inv(w)
        result = TruncLaurent(0, zeros(ComplexF64, prec), prec)
        winv_power = TruncLaurent(0, ComplexF64[one(ComplexF64)], prec)
        for n in 1:length(Fm.coeffs)
            power = Fm.val + n - 1  # -m, -m+1, ..., -1
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

"""
Invert a series f with valuation 1: f(ζ) = αζ(1 + h(ζ)) → f^{-1} = (1/α)ζ^{-1}(1+h)^{-1}
"""
function _invert_val1_series(f::TruncLaurent{T}) where T
    α = f.coeffs[1]
    prec = f.prec
    # Build (1 + h) where f = α·ζ·(1+h), so (1+h) coefficients are f.coeffs[k]/α
    n = length(f.coeffs)
    one_plus_h_coeffs = zeros(T, n)
    for k in 1:n
        one_plus_h_coeffs[k] = f.coeffs[k] / α
    end
    one_plus_h = TruncLaurent(0, one_plus_h_coeffs, prec)
    inv_one_plus_h = inv(one_plus_h)
    # f^{-1} = (1/α) · ζ^{-1} · inv(1+h)
    out_coeffs = T[c / α for c in inv_one_plus_h.coeffs]
    TruncLaurent(-1, out_coeffs, prec - 1)
end
