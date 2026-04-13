#!/usr/bin/env julia
# Compare Dict key strategies for the recursion's memoization table.
# Benchmark: insert N entries, then do 10N random lookups.

using Random
Random.seed!(42)

N = 2_000_000  # approximate size at h_max=8

# Generate random keys (simulating (n_T, n_L, n_R, αT, αL, αR))
keys_6tuple = [(rand(-3:3), rand(-3:3), rand(-3:3),
                rand(1:50), rand(1:50), rand(1:50)) for _ in 1:N]
values = randn(N)

# Strategy 1: Dict{NTuple{6,Int}, Float64} (current)
println("Strategy 1: NTuple{6,Int} key")
d1 = Dict{NTuple{6,Int}, Float64}()
@time for (i, k) in enumerate(keys_6tuple)
    d1[k] = values[i]
end
lookup_keys = [keys_6tuple[rand(1:N)] for _ in 1:10N]
s = Ref(0.0)
@time for k in lookup_keys
    s[] += get(d1, k, 0.0)
end
println("  $(length(d1)) entries, sum=$(s[])")

# Strategy 2: Dict{Int64, Float64} with packed key
println("\nStrategy 2: packed Int64 key")
function pack_key(n_T, n_L, n_R, aT, aL, aR)
    # Pack 6 small ints into one Int64 (each fits in ~10 bits)
    ((Int64(n_T + 10) << 50) | (Int64(n_L + 10) << 40) |
     (Int64(n_R + 10) << 30) | (Int64(aT) << 20) |
     (Int64(aL) << 10) | Int64(aR))
end
d2 = Dict{Int64, Float64}()
@time for (i, k) in enumerate(keys_6tuple)
    d2[pack_key(k...)] = values[i]
end
lookup_packed = [pack_key(k...) for k in lookup_keys]
s2 = Ref(0.0)
@time for k in lookup_packed
    s2[] += get(d2, k, 0.0)
end
println("  $(length(d2)) entries, sum=$(s2[])")

# Strategy 3: Dense Vector with flat indexing
println("\nStrategy 3: Dense Vector (flat index)")
# Simulate: 7 sectors (-3..3), 50 states each
# triple_id = (n_T+3)*7*7 + (n_L+3)*7 + (n_R+3)
# flat = triple_id * 50*50*50 + (aT-1)*50*50 + (aL-1)*50 + aR
max_n = 3; max_a = 50
n_triples = (2*max_n+1)^3
total_size = n_triples * max_a^3
println("  Allocating $(total_size) entries ($(round(total_size*8/1e6; digits=1)) MB)")
dense = zeros(Float64, total_size)
function flat_idx(n_T, n_L, n_R, aT, aL, aR)
    tid = (n_T + max_n) * (2*max_n+1)^2 + (n_L + max_n) * (2*max_n+1) + (n_R + max_n)
    tid * max_a^3 + (aT-1) * max_a^2 + (aL-1) * max_a + aR
end
@time for (i, k) in enumerate(keys_6tuple)
    dense[flat_idx(k...) + 1] = values[i]
end
lookup_flat = [flat_idx(k...) + 1 for k in lookup_keys]
s3 = Ref(0.0)
@time for idx in lookup_flat
    s3[] += dense[idx]
end
println("  sum=$(s3[])")

println("\nDone.")
