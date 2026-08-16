"""
Streaming & Sketching Algorithms:
1. Bloom Filter
2. Flajolet-Martin (FM)
3. HyperLogLog (HLL)
4. Misra-Gries
5. Count-Min Sketch
6. Count Sketch
7. AMS Sketch (F2 moment)
8. Hash Kernels (Feature Hashing)
Dependencies: numpy, matplotlib, hashlib
"""

import numpy as np
import matplotlib.pyplot as plt
import hashlib


# ==========================================
# Universal Hashing Utilities
# ==========================================
def _hash_murmur_like(x, seed, mask=0xFFFFFFFF):
    """Numerically stable integer hash using 32-bit bitwise mixing."""
    if isinstance(x, str):
        val = int(hashlib.md5((x + str(seed)).encode()).hexdigest()[:8], 16)
    else:
        val = int(x) ^ (seed * 0x5bd1e995)
    val = ((val >> 16) ^ val) * 0x45d9f3b
    val = ((val >> 16) ^ val) * 0x45d9f3b
    val = (val >> 16) ^ val
    return val & mask


def _count_trailing_zeros(v):
    if v == 0:
        return 32
    return (v & -v).bit_length() - 1


# ==========================================
# 1. Bloom Filter
# ==========================================
class BloomFilter:
    def __init__(self, size=5000, num_hashes=4):
        self.size = size
        self.k = num_hashes
        self.bit_array = np.zeros(size, dtype=bool)

    def add(self, item):
        for seed in range(self.k):
            idx = _hash_murmur_like(item, seed) % self.size
            self.bit_array[idx] = True

    def contains(self, item):
        for seed in range(self.k):
            idx = _hash_murmur_like(item, seed) % self.size
            if not self.bit_array[idx]:
                return False
        return True


# ==========================================
# 2. Flajolet-Martin (FM) Cardinality
# ==========================================
class FlajoletMartin:
    def __init__(self, num_hashes=16):
        self.num_hashes = num_hashes
        self.max_trailing_zeros = np.zeros(num_hashes, dtype=int)

    def process(self, item):
        for i in range(self.num_hashes):
            h = _hash_murmur_like(item, seed=i)
            tz = _count_trailing_zeros(h)
            self.max_trailing_zeros[i] = max(self.max_trailing_zeros[i], tz)

    def estimate(self):
        phi = 0.77351  # Correction constant
        estimates = (2.0 ** self.max_trailing_zeros) / phi
        return np.median(estimates)


# ==========================================
# 3. HyperLogLog (HLL)
# ==========================================
class HyperLogLog:
    def __init__(self, p=6):  # m = 2^p registers
        self.p = p
        self.m = 1 << p
        self.registers = np.zeros(self.m, dtype=int)
        
        # Alpha correction factor
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1.0 + 1.079 / self.m)

    def process(self, item):
        h = _hash_murmur_like(item, seed=42)
        idx = h >> (32 - self.p)  # First p bits define register index
        w = (h << self.p) & 0xFFFFFFFF  # Remaining bits
        leading_zeros = 1 + _count_trailing_zeros(w)
        self.registers[idx] = max(self.registers[idx], leading_zeros)

    def estimate(self):
        # Harmonic mean
        indicator = np.sum(2.0 ** (-self.registers))
        raw_estimate = self.alpha * (self.m ** 2) / indicator
        
        # Small range correction
        if raw_estimate <= 2.5 * self.m:
            zeros = np.count_nonzero(self.registers == 0)
            if zeros != 0:
                raw_estimate = self.m * np.log(self.m / zeros)
        return raw_estimate


# ==========================================
# 4. Misra-Gries (Heavy Hitters)
# ==========================================
class MisraGries:
    def __init__(self, k=10):
        self.k = k
        self.counters = {}

    def process(self, item):
        if item in self.counters:
            self.counters[item] += 1
        elif len(self.counters) < self.k - 1:
            self.counters[item] = 1
        else:
            # Decrement all
            to_remove = []
            for key in list(self.counters.keys()):
                self.counters[key] -= 1
                if self.counters[key] == 0:
                    to_remove.append(key)
            for key in to_remove:
                del self.counters[key]

    def estimate(self, item):
        return self.counters.get(item, 0)


# ==========================================
# 5. Count-Min Sketch
# ==========================================
class CountMinSketch:
    def __init__(self, width=200, depth=5):
        self.w = width
        self.d = depth
        self.table = np.zeros((depth, width), dtype=int)

    def process(self, item, count=1):
        for row in range(self.d):
            col = _hash_murmur_like(item, seed=row) % self.w
            self.table[row, col] += count

    def estimate(self, item):
        return min(self.table[row, _hash_murmur_like(item, seed=row) % self.w] for row in range(self.d))


# ==========================================
# 6. Count Sketch
# ==========================================
class CountSketch:
    def __init__(self, width=200, depth=5):
        self.w = width
        self.d = depth
        self.table = np.zeros((depth, width), dtype=float)

    def process(self, item, count=1):
        for row in range(self.d):
            col = _hash_murmur_like(item, seed=row) % self.w
            sign = 1 if (_hash_murmur_like(item, seed=row + 100) % 2) == 1 else -1
            self.table[row, col] += sign * count

    def estimate(self, item):
        estimates = []
        for row in range(self.d):
            col = _hash_murmur_like(item, seed=row) % self.w
            sign = 1 if (_hash_murmur_like(item, seed=row + 100) % 2) == 1 else -1
            estimates.append(sign * self.table[row, col])
        return np.median(estimates)


# ==========================================
# 7. AMS Sketch (Second Moment F2)
# ==========================================
class AMSSketch:
    def __init__(self, num_groups=7, group_size=15):
        self.d1 = num_groups
        self.d2 = group_size
        self.sketches = np.zeros((num_groups, group_size), dtype=float)

    def process(self, item, count=1):
        for i in range(self.d1):
            for j in range(self.d2):
                seed = i * self.d2 + j
                sign = 1 if (_hash_murmur_like(item, seed=seed) % 2 == 1) else -1
                self.sketches[i, j] += sign * count

    def estimate_f2(self):
        # Average within rows, median across rows
        row_estimates = np.mean(self.sketches**2, axis=1)
        return np.median(row_estimates)


# ==========================================
# 8. Hash Kernels (Feature Hashing)
# ==========================================
class HashKernel:
    def __init__(self, n_features=64):
        self.n_features = n_features

    def transform(self, sparse_vector):
        """Hashes an arbitrary-length feature dictionary into fixed dimension."""
        transformed = np.zeros(self.n_features, dtype=float)
        for feat_idx, val in sparse_vector.items():
            h_idx = _hash_murmur_like(feat_idx, seed=1) % self.n_features
            sign = 1 if (_hash_murmur_like(feat_idx, seed=2) % 2 == 1) else -1
            transformed[h_idx] += sign * val
        return transformed


# ==========================================
# Benchmark & Visualizations
# ==========================================
def run_streaming_benchmarks():
    rng = np.random.default_rng(42)

    # 1. Bloom Filter False Positive Rate Experiment
    capacities = np.linspace(100, 2000, 10, dtype=int)
    empirical_fpr = []
    theoretical_fpr = []
    bf_size, bf_k = 4000, 4

    for n in capacities:
        bf = BloomFilter(size=bf_size, num_hashes=bf_k)
        inserted_items = [f"item_{i}" for i in range(n)]
        for it in inserted_items:
            bf.add(it)
            
        test_negatives = [f"test_{i}" for i in range(1000)]
        fps = sum(1 for it in test_negatives if bf.contains(it))
        empirical_fpr.append(fps / 1000.0)
        
        # Formula: (1 - e^(-kn/m))^k
        p_theo = (1.0 - np.exp(-bf_k * n / bf_size)) ** bf_k
        theoretical_fpr.append(p_theo)

    # 2. Cardinality Estimation: Exact vs FM vs HLL
    stream_sizes = [500, 1000, 2500, 5000, 10000]
    fm_estimates, hll_estimates = [], []
    
    for sz in stream_sizes:
        fm = FlajoletMartin(num_hashes=32)
        hll = HyperLogLog(p=7)
        for i in range(sz):
            item_str = f"stream_val_{i}"
            fm.process(item_str)
            hll.process(item_str)
        fm_estimates.append(fm.estimate())
        hll_estimates.append(hll.estimate())

    # 3. Frequency Estimation Error on Zipfian Stream
    zipf_data = rng.zipf(a=1.5, size=20000)
    unique_items, true_counts = np.unique(zipf_data, return_counts=True)
    
    cms = CountMinSketch(width=100, depth=5)
    cs = CountSketch(width=100, depth=5)
    mg = MisraGries(k=50)
    
    for val in zipf_data:
        cms.process(int(val))
        cs.process(int(val))
        mg.process(int(val))

    top_items = unique_items[np.argsort(-true_counts)[:20]]
    actual_freqs = [np.sum(zipf_data == x) for x in top_items]
    cms_errs = [abs(cms.estimate(int(x)) - actual_freqs[i]) for i, x in enumerate(top_items)]
    cs_errs = [abs(cs.estimate(int(x)) - actual_freqs[i]) for i, x in enumerate(top_items)]
    mg_errs = [abs(mg.estimate(int(x)) - actual_freqs[i]) for i, x in enumerate(top_items)]

    # --- Plotting Visualizations ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Bloom Filter FPR
    ax1 = axes[0]
    ax1.plot(capacities, empirical_fpr, 'o-', color='#1f77b4', label='Empirical FPR')
    ax1.plot(capacities, theoretical_fpr, '--', color='#d62728', label='Theoretical FPR')
    ax1.set_title("Bloom Filter: False Positive Rate vs Load", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Number of Inserted Items ($n$)")
    ax1.set_ylabel("False Positive Rate")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Plot 2: Cardinality Tracking
    ax2 = axes[1]
    ax2.plot(stream_sizes, stream_sizes, 'k--', label='Exact Distinct Count')
    ax2.plot(stream_sizes, fm_estimates, 's-', color='#2ca02c', label='Flajolet-Martin')
    ax2.plot(stream_sizes, hll_estimates, '^-', color='#ff7f0e', label='HyperLogLog')
    ax2.set_title("Distinct Element Counting Accuracy", fontsize=11, fontweight='bold')
    ax2.set_xlabel("True Cardinality")
    ax2.set_ylabel("Estimated Cardinality")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    # Plot 3: Frequency Sketch Errors
    ax3 = axes[2]
    x_indices = np.arange(len(top_items))
    width = 0.25
    ax3.bar(x_indices - width, cms_errs, width=width, label='Count-Min', color='#1f77b4')
    ax3.bar(x_indices, cs_errs, width=width, label='Count Sketch', color='#ff7f0e')
    ax3.bar(x_indices + width, mg_errs, width=width, label='Misra-Gries', color='#2ca02c')
    ax3.set_title("Absolute Error on Top 20 Heavy Hitters", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Item Rank")
    ax3.set_ylabel("Absolute Frequency Error $|f - \hat{f}|$")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend()

    plt.tight_layout()
    plt.show()

    # 4. AMS and Hash Kernel Demonstrations
    exact_f2 = np.sum(true_counts**2)
    ams = AMSSketch(num_groups=9, group_size=25)
    for v in zipf_data:
        ams.process(int(v))
    estimated_f2 = ams.estimate_f2()
    print(f"[AMS Sketch] Exact F2 Moment: {exact_f2:,.0f} | Estimated F2: {estimated_f2:,.0f} (Relative Err: {abs(exact_f2 - estimated_f2)/exact_f2:.2%})")

    # Hash Kernel Inner Product Preservation Test
    hk = HashKernel(n_features=128)
    vec_a = {10: 2.0, 55: 4.0, 1024: 1.5, 9000: 3.0}
    vec_b = {10: 1.0, 55: 2.0, 2048: 5.0, 9000: 2.0}
    
    true_dot = sum(vec_a[k] * vec_b.get(k, 0.0) for k in vec_a)
    h_a = hk.transform(vec_a)
    h_b = hk.transform(vec_b)
    estimated_dot = np.dot(h_a, h_b)
    print(f"[Hash Kernel] True Dot Product: {true_dot:.2f} | Hashed Feature Dot Product: {estimated_dot:.2f}")


if __name__ == "__main__":
    run_streaming_benchmarks()
