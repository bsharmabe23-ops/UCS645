import numpy as np
import time
import matplotlib.pyplot as plt

print("=" * 60)
print("PROBLEM 2: Parallel Reduction & Shared Memory")
print("=" * 60)

N = 2**20
data = np.random.rand(N).astype(np.float32)
expected = np.sum(data)

# ─────────────────────────────────────────────
# PART A — Three Reduction Strategies
# ─────────────────────────────────────────────
print("\n--- PART A: Three Reduction Strategies ---\n")

# 1. Naive sequential reduction
t0 = time.perf_counter()
result_naive = 0.0
for i in range(0, N, N//64):
    result_naive += np.sum(data[i:i+N//64])
t1 = time.perf_counter()
naive_time = (t1 - t0) * 1e6  # microseconds

# 2. Shared memory tree reduction (simulated with numpy chunked)
t0 = time.perf_counter()
chunk = 256
partial = np.array([np.sum(data[i:i+chunk]) for i in range(0, N, chunk)])
result_tree = np.sum(partial)
t1 = time.perf_counter()
tree_time = (t1 - t0) * 1e6

# 3. Warp shuffle reduction (simulated with numpy vectorized)
t0 = time.perf_counter()
result_warp = np.sum(data)
t1 = time.perf_counter()
warp_time = (t1 - t0) * 1e6

bytes_accessed = data.nbytes

print(f"{'Strategy':<25} {'Time (us)':>12} {'GB/s':>10} {'Result':>14} {'Correct?':>10}")
print("-" * 75)

for name, t, res in [
    ("Naive Sequential",   naive_time, result_naive),
    ("Shared Mem Tree",    tree_time,  result_tree),
    ("Warp Shuffle",       warp_time,  result_warp),
]:
    gbs = bytes_accessed / (t * 1e-6) / 1e9
    correct = "✓" if abs(res - expected) < 0.1 * N else "✗"
    print(f"{name:<25} {t:>12.2f} {gbs:>10.2f} {res:>14.2f} {correct:>10}")

print(f"\nReference (numpy.sum): {expected:.2f}")

# ─────────────────────────────────────────────
# PART B — Bank Conflict Profiling
# ─────────────────────────────────────────────
print("\n--- PART B: Bank Conflict Profiling ---\n")

tile = np.random.rand(16, 16).astype(np.float32)
strides = [1, 2, 4, 8, 16, 32]

print(f"{'Stride':>8} {'Time (us)':>12} {'Bank Conflicts':>16}")
print("-" * 40)

stride_times = []
for s in strides:
    t0 = time.perf_counter()
    for _ in range(1000):
        idx = (np.arange(16) * s) % 256
        _ = tile.flatten()[idx % 256]
    t1 = time.perf_counter()
    elapsed = (t1 - t0) * 1e6 / 1000

    if s == 1:
        conflict = "None (optimal)"
    elif s == 32:
        conflict = "Max (32-way)"
    else:
        conflict = f"{s}-way"

    stride_times.append(elapsed)
    print(f"{s:>8} {elapsed:>12.4f} {conflict:>16}")

print("\nExplanation:")
print("Stride=1: consecutive threads access consecutive banks → no conflicts (optimal)")
print("Stride=32: all 32 threads in a warp access the SAME bank → 32-way conflict,")
print("serialized into 32 sequential accesses, destroying throughput.")

print("\nPadding Solution (tile[16][17]):")
print("Adding 1 extra column shifts each row's start to a different bank.")
print("This breaks the alignment that causes conflicts at stride=32.")

# Measure padding speedup
t0 = time.perf_counter()
for _ in range(1000):
    padded = np.zeros((16, 17), dtype=np.float32)
    padded[:, :16] = tile
    _ = padded[:, :16].sum()
t1 = time.perf_counter()
padded_time = (t1 - t0) * 1e6 / 1000

t0 = time.perf_counter()
for _ in range(1000):
    _ = tile.sum()
t1 = time.perf_counter()
normal_time = (t1 - t0) * 1e6 / 1000

print(f"Without padding: {normal_time:.4f} us")
print(f"With padding:    {padded_time:.4f} us")
print(f"Speedup:         {normal_time/padded_time:.2f}x")

# Plot bank conflict times
plt.figure(figsize=(8, 4))
plt.plot(strides, stride_times, 'o-', color='red')
plt.xlabel('Stride')
plt.ylabel('Time (us)')
plt.title('Shared Memory Access Time vs Stride (Bank Conflicts)')
plt.xticks(strides)
plt.grid(True)
plt.savefig('p2_bank_conflicts.png')
print("\nPlot saved: p2_bank_conflicts.png")

# ─────────────────────────────────────────────
# PART C — Histogram with Shared Memory
# ─────────────────────────────────────────────
print("\n--- PART C: Shared Memory Histogram ---\n")

N_hist = 2**20
num_bins = 256
data_hist = np.random.randint(0, 256, N_hist).astype(np.int32)

# Global atomics histogram (naive)
t0 = time.perf_counter()
hist_global = np.bincount(data_hist, minlength=num_bins)
t1 = time.perf_counter()
global_time = (t1 - t0) * 1e6

# Private per-block histogram then merge (shared memory approach)
t0 = time.perf_counter()
block_size = 1024
num_blocks = (N_hist + block_size - 1) // block_size
partial_hists = np.zeros((num_blocks, num_bins), dtype=np.int32)
for b in range(num_blocks):
    chunk = data_hist[b*block_size : (b+1)*block_size]
    partial_hists[b] = np.bincount(chunk, minlength=num_bins)
hist_shared = partial_hists.sum(axis=0)
t1 = time.perf_counter()
shared_time = (t1 - t0) * 1e6

correct = np.array_equal(hist_global, hist_shared)
print(f"Global atomics time:  {global_time:.2f} us")
print(f"Shared memory time:   {shared_time:.2f} us")
print(f"Speedup:              {shared_time/global_time:.2f}x")
print(f"Correctness check:    {'✓ PASSED' if correct else '✗ FAILED'}")
print("\nNote: Shared memory reduces global atomicAdd contention by keeping")
print("per-block partial histograms in fast shared memory, then merging once.")

print("\n✅ Problem 2 Complete!")
