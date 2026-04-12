"""
BPZ bilinear form and conjugation map.
"""

"""
    build_bpz_form(basis::FockBasis) -> TensorMap

Build the BPZ bilinear form η : V ⊗ V → ℂ.
Diagonal in the normalised basis with entries (-1)^level.
"""
function build_bpz_form(basis::FockBasis)
    V = basis.V
    η = zeros(Float64, one(V), V ⊗ V)
    for (f₁, f₂) in fusiontrees(η)
        # Domain is V ⊗ V; sector labels are (n, -n) with n+(-n)=0
        n = Int(f₂.uncoupled[1].charge)
        haskey(basis.levels, n) || continue
        blk = η[f₁, f₂]
        d = size(blk, 1)
        for α in 1:min(d, length(basis.levels[n]))
            blk[α, α] = (-1.0)^basis.levels[n][α]
        end
        η[f₁, f₂] = blk
    end
    η
end

"""
    build_bpz_map(basis::FockBasis) -> TensorMap

Build the BPZ conjugation map η : V → V' (maps kets to dual vectors).
Diagonal with entries (-1)^level. The codomain is V' (the dual space),
reflecting that BPZ conjugation sends states to covectors.

For U(1) grading, V' has charge sectors conjugated (charge n → -n), so
the block at (V' charge -n, V charge n) is the (-1)^level diagonal.
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
            blk[α, α] = (-1.0)^basis.levels[n][α]
        end
        η[f₁, f₂] = blk
    end
    η
end
