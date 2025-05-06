# ============================================================
# 🛠️  Cell 0 — Install / Imports
# ============================================================
# !pip -q install torch torchvision --upgrade
# !pip -q install cma

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import numpy as np, random, math, cma, copy, time

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Dataset (10 k train, 5 k val, 10 k test)
# ------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor()])
full_train = datasets.MNIST(root=".", train=True, download=True, transform=transform)
# Dataset (10 k train, 50 k val)
train_set, val_set = random_split(
    full_train,
    [10_000, len(full_train) - 10_000],  # 10 000 + 50 000 = 60 000
    generator=torch.Generator().manual_seed(SEED)
)

test_set = datasets.MNIST(root=".", train=False, download=True, transform=transform)

BATCH = 128
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False, drop_last=False)
test_loader = DataLoader(test_set, batch_size=BATCH, shuffle=False, drop_last=False)


# ------------------------------------------------------------
# Model & helpers
# ------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(5 * 5 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def accuracy(model, loader):
    model.eval()
    corr = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            corr += (preds == y).sum().item()
    return corr / len(loader.dataset)


# 🔄 Parameter (de)‑flattening for CMA‑ES ----------------------
def model_to_vec(net):
    return torch.cat([p.data.view(-1) for p in net.parameters()]).cpu().numpy()


def vec_to_model(net, vec):
    idx = 0
    for p in net.parameters():
        numel = p.numel()
        p.data.copy_(torch.tensor(vec[idx:idx + numel]).view_as(p).to(device))
        idx += numel


# ============================================================
# 🧪 Cell 1 — Plain Adam baseline
# ============================================================
E_BASE = 3  # baseline epochs
LR_BASE = 1e-3

baseline = SmallCNN().to(device)
opt = torch.optim.Adam(baseline.parameters(), lr=LR_BASE)

for epoch in range(E_BASE):
    baseline.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(baseline(x), y)
        loss.backward()
        opt.step()
    print(
        f"Epoch {epoch + 1}/{E_BASE}  |  Train acc: {accuracy(baseline, train_loader):.3f}  |  Val acc: {accuracy(baseline, val_loader):.3f}")

print("➡️  Baseline test accuracy:", accuracy(baseline, test_loader))

# ============================================================
# 🧬 Cell 2 — Hybrid Adam + CMA‑ES
# ============================================================
μ = 4  # parents
λ = 8  # offspring
k = 20  # Adam steps inside each generation
G = 3  # generations
LR_INNER = 1e-3
SIGMA0 = 0.05  # initial CMA step‑size

# --- prepare a template network + flattening helpers --------
template_net = SmallCNN().to(device)
dim = len(model_to_vec(template_net))


def fitness(vec):
    """Return *negative* validation accuracy after k Adam steps"""
    net = copy.deepcopy(template_net)
    vec_to_model(net, vec)
    opt = torch.optim.Adam(net.parameters(), lr=LR_INNER)
    net.train()
    train_iter = iter(train_loader)
    for _ in range(k):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(net(x), y)
        loss.backward()
        opt.step()
    # Negative because CMA‑ES minimizes
    return -accuracy(net, val_loader)


# --- Initialize CMA‑ES --------------------------------------
es = cma.CMAEvolutionStrategy(dim * [0.0], SIGMA0,
                              {'popsize': λ, 'seed': SEED})

best_val_acc = 0
best_vec = None

for gen in range(G):
    # ask for a batch of candidate parameter vectors
    solutions = es.ask()
    scores = [fitness(v) for v in solutions]  # negative val acc
    es.tell(solutions, scores)
    es.disp()

    # record best
    gen_best_idx = int(np.argmin(scores))
    gen_best_val = -scores[gen_best_idx]
    if gen_best_val > best_val_acc:
        best_val_acc = gen_best_val
        best_vec = solutions[gen_best_idx]
    print(f"Gen {gen + 1}/{G} — best val acc: {gen_best_val:.3f}")

# --- Evaluate the best individual on the test set -----------
hybrid_model = SmallCNN().to(device)
vec_to_model(hybrid_model, best_vec)
print("➡️  Hybrid test accuracy:", accuracy(hybrid_model, test_loader))
