"""
    TruncLaurent{T}

Truncated Laurent series: Σ_{n=val}^{prec-1} coeffs[n-val+1] * ζ^n + O(ζ^prec).
"""
struct TruncLaurent{T<:Number}
    val::Int           # valuation: exponent of the leading term
    coeffs::Vector{T}  # coefficients[k] = coefficient of ζ^(val+k-1)
    prec::Int          # precision: series is O(ζ^prec)
end

valuation(s::TruncLaurent) = s.val
series_precision(s::TruncLaurent) = s.prec

function Base.getindex(s::TruncLaurent{T}, n::Int) where T
    i = n - s.val + 1
    if i < 1 || i > length(s.coeffs) || n >= s.prec
        return zero(T)
    end
    return s.coeffs[i]
end

# --- Arithmetic ---

function Base.:*(a::TruncLaurent{T}, b::TruncLaurent{S}) where {T,S}
    R = promote_type(T, S)
    prec = min(a.prec, b.prec)
    new_val = a.val + b.val
    nterms = prec - new_val
    nterms <= 0 && return TruncLaurent(new_val, R[], prec)
    coeffs = zeros(R, nterms)
    for i in eachindex(a.coeffs), j in eachindex(b.coeffs)
        k = i + j - 1  # index into result
        k > nterms && continue
        coeffs[k] += a.coeffs[i] * b.coeffs[j]
    end
    TruncLaurent(new_val, coeffs, prec)
end

"""
    inv(a::TruncLaurent{T}) -> TruncLaurent{T}

Multiplicative inverse of a truncated Laurent series. Handles any integer
valuation: if `a` has valuation `v` and leading coefficient `α`, the result
has valuation `-v` and leading coefficient `1/α`.

For `v = 0` this is the standard power-series inverse via the recurrence
`b[n] = -(1/a[0]) Σ_{j=1}^n a[j] b[n-j]`.
For `v ≠ 0` we factor out `α·ζ^v`, invert the `(1 + h)` part, and multiply
back by `(1/α)·ζ^{-v}`.
"""
function Base.inv(a::TruncLaurent{T}) where T
    isempty(a.coeffs) && error("inv of empty series")
    a.coeffs[1] == zero(T) && error("inv requires nonzero leading coefficient")

    if a.val == 0
        return _inv_val0(a)
    else
        # Factor a = α · ζ^v · (1 + h), invert the (1 + h) part,
        # then shift by ζ^{-v}. The existing call sites that this
        # subsumes (`_series_inv_val1`, `_invert_val1_series`) reduce
        # the precision by `abs(v)` — we preserve that convention.
        α = a.coeffs[1]
        one_plus_h_coeffs = T[c / α for c in a.coeffs]
        one_plus_h = TruncLaurent(0, one_plus_h_coeffs, a.prec)
        inv_oph = _inv_val0(one_plus_h)
        out_coeffs = T[c / α for c in inv_oph.coeffs]
        return TruncLaurent(-a.val, out_coeffs, a.prec - abs(a.val))
    end
end

# Helper: inv specialised to val=0 series (the recurrence).
function _inv_val0(a::TruncLaurent{T}) where T
    @assert a.val == 0
    prec = a.prec
    n = prec
    b = zeros(T, n)
    a0inv = one(T) / a.coeffs[1]
    b[1] = a0inv
    for k in 2:n
        s = zero(T)
        for j in 2:min(k, length(a.coeffs))
            s += a.coeffs[j] * b[k - j + 1]
        end
        b[k] = -a0inv * s
    end
    TruncLaurent(0, b, prec)
end

# --- exp_series: requires valuation ≥ 1 ---

function exp_series(a::TruncLaurent{T}) where T
    a.val < 1 && error("exp_series requires valuation ≥ 1, got $(a.val)")
    prec = a.prec
    n = prec  # output runs from ζ^0 to ζ^(prec-1)
    e = zeros(T, n)
    e[1] = one(T)
    # Recurrence: (k) * e[k+1] = Σ_{j=1}^{k} j * a_j * e[k+1-j]
    # where a_j = a[j] (coefficient of ζ^j, stored at index j-val+1 of a)
    for k in 1:n-1
        s = zero(T)
        for j in 1:k
            aj = (j >= a.val && j < a.prec) ? a[j] : zero(T)
            aj == zero(T) && continue
            s += T(j) * aj * e[k - j + 1]
        end
        e[k + 1] = s / T(k)
    end
    TruncLaurent(0, e, prec)
end

# --- Composition: f(g(ξ)), requires valuation(g) ≥ 1 ---

function compose(f::TruncLaurent{T}, g::TruncLaurent{S}) where {T,S}
    g.val < 1 && error("compose requires inner series valuation ≥ 1, got $(g.val)")
    R = promote_type(T, S)
    prec = min(f.prec * g.val, g.prec)  # conservative precision
    # Horner's method: f = c0 + c1*ζ + c2*ζ² + ...
    # Start from highest order and work down
    n = length(f.coeffs)
    # Result accumulator — start with the highest coefficient
    if n == 0
        return TruncLaurent(0, R[], prec)
    end

    # Build result using Horner: result = f[last] * g + f[last-1], etc.
    # But f may have negative powers, so we need a different approach.
    # General approach: compute g^k for each k and accumulate.

    # Powers of g: g^0 = 1, g^1 = g, g^2 = g*g, ...
    # For negative powers of g when f has val < 0, we need g^(-1) etc.
    # But compose is only valid when val(g) ≥ 1 and val(f) ≥ 0 typically.
    # Actually f can have any valuation as long as g starts at ζ^1.
    # f(g) with f having val < 0 requires g^(-k) which is a Laurent series.

    # For our use case, f typically has val ≥ 0.
    # Let's handle general case: f = Σ c_n ζ^n for n from f.val to f.val+len-1
    # f(g) = Σ c_n g^n

    # Start with g_power = g^(f.val)
    # We need inv(g) if f.val < 0. Now that Base.inv handles val=1,
    # we can use it directly.
    if f.val < 0
        ginv = inv(g)
        g_power = _series_power_val1(ginv, -f.val, prec)
    elseif f.val == 0
        g_power = TruncLaurent(0, R[one(R)], prec)
    else
        g_power = _series_power_val1(g, f.val, prec)
    end

    result = TruncLaurent(0, zeros(R, prec), prec)
    for i in 1:n
        coeff = R(f.coeffs[i])
        if coeff != zero(R)
            result = _add(result, _scale(g_power, coeff))
        end
        if i < n
            result, g_power = result, _truncmul(g_power, g, prec)
        end
    end
    result
end

function _series_power_val1(g::TruncLaurent, n::Int, prec::Int)
    n == 0 && return TruncLaurent(0, [one(eltype(g.coeffs))], prec)
    result = g
    for _ in 2:n
        result = _truncmul(result, g, prec)
    end
    result
end

function _truncmul(a::TruncLaurent{T}, b::TruncLaurent{S}, prec::Int) where {T,S}
    R = promote_type(T, S)
    new_val = a.val + b.val
    nterms = prec - new_val
    nterms <= 0 && return TruncLaurent(new_val, R[], prec)
    coeffs = zeros(R, nterms)
    for i in eachindex(a.coeffs), j in eachindex(b.coeffs)
        k = i + j - 1
        k > nterms && continue
        coeffs[k] += a.coeffs[i] * b.coeffs[j]
    end
    TruncLaurent(new_val, coeffs, prec)
end

function _add(a::TruncLaurent{T}, b::TruncLaurent{S}) where {T,S}
    R = promote_type(T, S)
    prec = min(a.prec, b.prec)
    new_val = min(a.val, b.val)
    nterms = prec - new_val
    nterms <= 0 && return TruncLaurent(new_val, R[], prec)
    coeffs = zeros(R, nterms)
    for k in 1:nterms
        n = new_val + k - 1
        coeffs[k] = R(a[n]) + R(b[n])
    end
    TruncLaurent(new_val, coeffs, prec)
end

function _scale(a::TruncLaurent{T}, c::S) where {T,S}
    R = promote_type(T, S)
    TruncLaurent(a.val, R[x * c for x in a.coeffs], a.prec)
end

# --- Series reversion: given f with val=1, find g such that f(g(ξ)) = ξ ---

function series_revert(f::TruncLaurent{T}) where T
    f.val != 1 && error("series_revert requires valuation 1, got $(f.val)")
    f.coeffs[1] == zero(T) && error("series_revert requires nonzero leading coefficient")
    prec = f.prec
    n = prec - 1  # number of output coefficients (from ζ^1 to ζ^(prec-1))
    n <= 0 && return TruncLaurent(1, T[], prec)

    # g(ξ) = d₁ξ + d₂ξ² + ..., where d₁ = 1/f₁
    d = zeros(T, n)
    f1 = f.coeffs[1]
    d[1] = one(T) / f1

    # Iterative: f(g(ξ)) = ξ
    # Build g order by order using the relation that the k-th coefficient
    # of f(g(ξ)) must be δ_{k,1}.
    # Use: f(g) = f₁*g + f₂*g² + ... and expand g = d₁ξ + d₂ξ² + ...
    # At order k: f₁*d_k + [terms involving d₁..d_{k-1}] = 0 for k ≥ 2

    for k in 2:n
        # Compute coefficient of ξ^k in f(g(ξ)) without the f₁*d_k term
        # This requires computing g^m up to ξ^k for m = 1..min(k, len(f))
        # and accumulating f_m * [ξ^k in g^m]

        # Build g so far (up to d_{k-1})
        g_partial = TruncLaurent(1, d[1:k-1], k + 1)

        # Compute f(g_partial) up to ξ^k
        s = zero(T)
        g_pow = g_partial  # g^1
        for m in 1:min(k, length(f.coeffs))
            fm = f.coeffs[m]
            if fm != zero(T) && m > 1
                # coefficient of ξ^k in f_m * g^m
                s += fm * _coeff_of_power(g_partial, m, k)
            end
        end
        # f₁*d_k + s = 0  (for k ≥ 2, since [ξ^k in f(g)] = 0)
        d[k] = -s / f1
    end

    TruncLaurent(1, d, prec)
end

# Coefficient of ξ^k in g^m, where g has valuation 1
function _coeff_of_power(g::TruncLaurent{T}, m::Int, k::Int) where T
    # g^m has valuation m, so if k < m, coefficient is 0
    k < m && return zero(T)
    pow = g
    for _ in 2:m
        pow = _truncmul(pow, g, k + 1)
    end
    return pow[k]
end

# --- Singular / regular part ---

function singular_part(s::TruncLaurent{T}) where T
    if s.val >= 0
        return TruncLaurent(0, T[], 0)
    end
    n_sing = min(-s.val, length(s.coeffs))
    TruncLaurent(s.val, s.coeffs[1:n_sing], 0)
end

function regular_part(s::TruncLaurent{T}) where T
    if s.val >= 0
        return s
    end
    offset = -s.val + 1  # index of the ζ^0 coefficient
    if offset > length(s.coeffs)
        return TruncLaurent(0, T[], s.prec)
    end
    TruncLaurent(0, s.coeffs[offset:end], s.prec)
end

# --- Evaluate at a point ---

function evaluate(s::TruncLaurent{T}, z) where T
    result = zero(promote_type(T, typeof(z)))
    for i in eachindex(s.coeffs)
        n = s.val + i - 1
        result += s.coeffs[i] * z^n
    end
    result
end
