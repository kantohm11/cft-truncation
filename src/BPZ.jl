"""
BPZ bilinear form and conjugation map.

For a quasi-primary of weight h, the BPZ map sends modes as:
    bpz(φ_m) = (-1)^{m-h} φ_{-m}

For the U(1) current J (weight h=1): bpz(J_{-k}) = (-1)^{k+1} J_k.
So odd modes (k odd) get no sign, even modes (k even) get -1.

The BPZ sign for a Fock state |λ⟩ = ∏ J_{-k_i} |0⟩ is:
    ∏_i (-1)^{k_i + 1} = (-1)^{N + #parts}
where N = level = Σk_i and #parts = length(λ).
"""

"""
    _bpz_sign(basis::FockBasis, n::Int, α::Int) -> Float64

BPZ sign for the α-th state in sector n.
Returns ∏_i (-1)^{k_i + 1} for partition λ = [k_1, k_2, ...].
"""
function _bpz_sign(basis::FockBasis, n::Int, α::Int)
    λ = basis.states[n][α]
    isempty(λ) && return 1.0
    s = 1.0
    for k in λ
        iseven(k) && (s = -s)
    end
    s
end

"""
    build_bpz_form(basis::FockBasis) -> TensorMap

Build the BPZ bilinear form η : V ⊗ V → ℂ.
Diagonal in the normalised basis with entries ∏(-1)^{k_i+1}.
"""
function build_bpz_form(basis::FockBasis)
    V = basis.V
    η = zeros(Float64, one(V), V ⊗ V)
    for (f₁, f₂) in fusiontrees(η)
        n = Int(f₂.uncoupled[1].charge)
        haskey(basis.levels, n) || continue
        blk = η[f₁, f₂]
        d = size(blk, 1)
        for α in 1:min(d, length(basis.levels[n]))
            blk[α, α] = _bpz_sign(basis, n, α)
        end
        η[f₁, f₂] = blk
    end
    η
end

"""
    build_bpz_map(basis::FockBasis) -> TensorMap

Build the BPZ conjugation map η : V → V' (maps kets to dual vectors).
Diagonal with entries ∏(-1)^{k_i+1}. The codomain is V' (the dual space),
reflecting that BPZ conjugation sends states to covectors.

For U(1) grading, V' has charge sectors conjugated (charge n → -n), so
the block at (V' charge -n, V charge n) is the BPZ diagonal.
"""
function build_bpz_map(basis::FockBasis)
    V = basis.V
    η = zeros(Float64, V', V)
    for (f₁, f₂) in fusiontrees(η)
        n = Int(f₂.uncoupled[1].charge)
        haskey(basis.levels, n) || continue
        blk = η[f₁, f₂]
        d = size(blk, 1)
        for α in 1:min(d, length(basis.levels[n]))
            blk[α, α] = _bpz_sign(basis, n, α)
        end
        η[f₁, f₂] = blk
    end
    η
end
