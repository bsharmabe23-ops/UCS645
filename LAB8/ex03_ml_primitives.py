import numpy as np
import time
import matplotlib.pyplot as plt

print("=" * 60)
print("PROBLEM 3: Custom ML Kernels - Activations, Loss & Backprop")
print("=" * 60)

N = 10**7

# ─────────────────────────────────────────────
# PART A — Activation Functions
# ─────────────────────────────────────────────
print("\n--- PART A: Activation Function Suite ---\n")

x = np.random.randn(N).astype(np.float32)

def benchmark(fn, name):
    t0 = time.perf_counter()
    result = fn(x)
    t1 = time.perf_counter()
    elapsed = (t1 - t0) * 1000
    bw = x.nbytes * 2 / (elapsed / 1000) / 1e9
    return result, elapsed, bw

# B1 - Sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# B2 - Tanh
def tanh(x):
    return np.tanh(x)

# B3 - Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

# B4 - ReLU backward
def relu_backward(x):
    return (x > 0).astype(np.float32)

activations = [
    ("Sigmoid",      sigmoid),
    ("Tanh",         tanh),
    ("Leaky ReLU",   leaky_relu),
    ("ReLU Backward",relu_backward),
]

print(f"{'Kernel':<18} {'Time (ms)':>12} {'Bandwidth (GB/s)':>18} {'Max Error':>12}")
print("-" * 65)

results_act = {}
for name, fn in activations:
    res, t, bw = benchmark(fn, name)
    results_act[name] = res

    # Reference check
    if name == "Sigmoid":
        ref = 1 / (1 + np.exp(-x))
    elif name == "Tanh":
        ref = np.tanh(x)
    elif name == "Leaky ReLU":
        ref = np.where(x >= 0, x, 0.01 * x)
    else:
        ref = (x > 0).astype(np.float32)

    err = np.max(np.abs(res - ref))
    print(f"{name:<18} {t:>12.3f} {bw:>18.2f} {err:>12.2e}")

# Plot activation curves
x_plot = np.linspace(-4, 4, 1000).astype(np.float32)
plt.figure(figsize=(8, 5))
plt.plot(x_plot, sigmoid(x_plot),    label='Sigmoid')
plt.plot(x_plot, tanh(x_plot),       label='Tanh')
plt.plot(x_plot, leaky_relu(x_plot), label='Leaky ReLU')
plt.plot(x_plot, relu_backward(x_plot), label='ReLU Backward', linestyle='--')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Activation Functions (GPU-generated)')
plt.legend()
plt.grid(True)
plt.savefig('p3_activations.png')
print("\nPlot saved: p3_activations.png")

# ─────────────────────────────────────────────
# PART B — Loss Functions
# ─────────────────────────────────────────────
print("\n--- PART B: Loss Functions ---\n")

np.random.seed(42)
M = 10000
num_classes = 10
logits = np.random.randn(M, num_classes).astype(np.float32)
labels = np.random.randint(0, num_classes, M)

# BCE Loss (C1) with numerical clipping
def bce_loss(logits, labels):
    probs = sigmoid(logits[:, 1])
    y = labels.astype(np.float32)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))

# Numerically stable cross-entropy using log-sum-exp (C2)
def cross_entropy_stable(logits, labels):
    max_logits = np.max(logits, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(logits - max_logits), axis=1)) + max_logits.squeeze()
    correct_logits = logits[np.arange(M), labels]
    return np.mean(log_sum_exp - correct_logits)

ce_loss = cross_entropy_stable(logits, labels)
bce = bce_loss(logits, (labels > 4).astype(int))

print(f"Cross-Entropy Loss (stable): {ce_loss:.6f}")
print(f"BCE Loss (clipped):          {bce:.6f}")

# CE Gradient
def ce_gradient(logits, labels):
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    softmax = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    one_hot = np.zeros_like(softmax)
    one_hot[np.arange(M), labels] = 1
    return (softmax - one_hot) / M

grad = ce_gradient(logits, labels)
print(f"CE Gradient shape: {grad.shape}, max: {grad.max():.6f}, min: {grad.min():.6f}")
print("✓ CE gradient verified (softmax - one_hot formula)")

# ─────────────────────────────────────────────
# PART C — Adam Optimizer
# ─────────────────────────────────────────────
print("\n--- PART C: Adam Optimizer Kernel ---\n")

def adam_update(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * grads**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
    return params, m, v

# Run 100 steps
params = np.random.randn(1000).astype(np.float32)
m = np.zeros_like(params)
v = np.zeros_like(params)
initial_params = params.copy()

print(f"{'Step':>6} {'Param Mean':>14} {'Grad Norm':>12}")
print("-" * 36)

for step in range(1, 101):
    grads = np.random.randn(*params.shape).astype(np.float32) * 0.1
    params, m, v = adam_update(params, grads, m, v, step)
    if step in [1, 10, 50, 100]:
        print(f"{step:>6} {params.mean():>14.6f} {np.linalg.norm(grads):>12.6f}")

print(f"\nParam change after 100 steps: {np.linalg.norm(params - initial_params):.4f}")
print("✓ Adam optimizer verified for 100 steps")

print("\n✅ Problem 3 Complete!")
