import numpy as np
import time
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    TORCH = True
except ImportError:
    TORCH = False
    print("PyTorch not found, using numpy simulation")

print("=" * 60)
print("PROBLEM 5: Full MNIST CNN Training")
print("=" * 60)

if not TORCH:
    print("Install PyTorch: pip3 install torch torchvision --break-system-packages")
    exit()

device = torch.device('cpu')
print(f"Device: {device}")

# ─────────────────────────────────────────────
# PART A — Model Implementation
# ─────────────────────────────────────────────
print("\n--- PART A: Model Implementation ---\n")

class MnistCNN(nn.Module):
    def __init__(self, use_bn=False, use_dropout=False):
        super().__init__()
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        # B1 - Conv layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        # BatchNorm layers
        self.bn1 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(64) if use_bn else nn.Identity()

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # B2 - FC layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()

    def forward(self, x):
        # B3 - Forward pass
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 28→14
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 14→7
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def build_optimizer(model, opt_type='adam'):
    if opt_type == 'adam':
        return optim.Adam(model.parameters(), lr=0.001)
    else:
        return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def build_scheduler(optimizer, sched_type='none'):
    if sched_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    return None

# Load MNIST
print("Loading MNIST dataset...")
transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])
try:
    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)
except:
    print("Could not download MNIST. Using random data for demo.")
    train_ds = [(torch.randn(1,28,28), np.random.randint(10)) for _ in range(1000)]
    test_ds  = [(torch.randn(1,28,28), np.random.randint(10)) for _ in range(200)]

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=256)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss/total, correct/total*100

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            correct += (out.argmax(1) == y).sum().item()
            total += X.size(0)
    return correct/total*100

criterion = nn.CrossEntropyLoss()
model = MnistCNN()
optimizer = build_optimizer(model, 'adam')

print(f"\nTraining baseline model (5 epochs)...")
print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc%':>12} {'Test Acc%':>12} {'Time(s)':>10}")
print("-" * 55)

train_losses, test_accs = [], []
for epoch in range(1, 6):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
    te_acc = evaluate(model, test_loader)
    elapsed = time.time() - t0
    train_losses.append(tr_loss)
    test_accs.append(te_acc)
    print(f"{epoch:>6} {tr_loss:>12.4f} {tr_acc:>12.2f} {te_acc:>12.2f} {elapsed:>10.1f}")

# ─────────────────────────────────────────────
# PART B — Ablation Study
# ─────────────────────────────────────────────
print("\n--- PART B: Ablation Study ---\n")

configs = [
    ("Baseline (Adam)",          MnistCNN(),                   'adam', 'none'),
    ("+ BatchNorm",              MnistCNN(use_bn=True),        'adam', 'none'),
    ("+ Dropout",                MnistCNN(use_dropout=True),   'adam', 'none'),
    ("SGD+Momentum+Cosine",      MnistCNN(),                   'sgd',  'cosine'),
]

print(f"{'Config':<28} {'Test Acc%':>10} {'Time(s)':>10}")
print("-" * 52)

ablation_results = []
for cfg_name, mdl, opt_t, sched_t in configs:
    opt = build_optimizer(mdl, opt_t)
    sched = build_scheduler(opt, sched_t)
    t0 = time.time()
    for ep in range(1, 4):  # 3 epochs for speed
        train_epoch(mdl, train_loader, opt, criterion)
        if sched: sched.step()
    acc = evaluate(mdl, test_loader)
    elapsed = time.time() - t0
    ablation_results.append((cfg_name, acc, elapsed))
    print(f"{cfg_name:<28} {acc:>10.2f} {elapsed:>10.1f}")

best = max(ablation_results, key=lambda x: x[1])
print(f"\nBest config: {best[0]} with {best[1]:.2f}% accuracy")
print("""
Discussion: BatchNorm stabilizes training by normalizing layer inputs,
reducing internal covariate shift and allowing higher learning rates.
Dropout prevents overfitting by randomly zeroing activations, acting
as an ensemble of smaller networks. SGD with momentum and CosineAnnealingLR
often achieves better final accuracy than Adam for CNNs on simple datasets
because the LR schedule helps escape local minima. For MNIST specifically,
BatchNorm gives the most consistent improvement as the dataset is clean
and regularization via Dropout may be less critical.
""")

# ─────────────────────────────────────────────
# PART C — Data Augmentation
# ─────────────────────────────────────────────
print("\n--- PART C: Data Augmentation ---\n")

aug_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, shear=10),
    transforms.RandomErasing(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

try:
    aug_ds = datasets.MNIST('./data', train=True, download=False, transform=aug_transform)
    aug_loader = DataLoader(aug_ds, batch_size=256, shuffle=True)

    model_aug = MnistCNN()
    opt_aug = build_optimizer(model_aug, 'adam')

    print("Training with augmentation (3 epochs)...")
    for ep in range(1, 4):
        train_epoch(model_aug, aug_loader, opt_aug, criterion)
    aug_acc = evaluate(model_aug, test_loader)
    base_acc = test_accs[-1]

    print(f"Without augmentation: {base_acc:.2f}%")
    print(f"With augmentation:    {aug_acc:.2f}%")
    print(f"Augmentation helps:   {'Yes' if aug_acc > base_acc else 'Marginal for MNIST'}")
    print("Note: MNIST is simple enough that augmentation has limited effect,")
    print("but for complex datasets augmentation typically improves generalization.")
except Exception as e:
    print(f"Augmentation demo skipped: {e}")

# Plot training curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, 6), train_losses, 'o-')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Training Loss'); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, 6), test_accs, 's-', color='green')
plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy'); plt.grid(True)

plt.tight_layout()
plt.savefig('p5_training.png')
print("\nTraining plot saved: p5_training.png")

print("\n✅ Problem 5 Complete!")
print("\n🎉 ALL 5 PROBLEMS DONE!")
