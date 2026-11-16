# Same network and settings as bench.cpp: 784 -> 128 (ReLU) -> 10, SGD lr 0.1, batch 64.
import sys
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)     # nnlib is single-threaded, so PyTorch gets 1 thread too
torch.manual_seed(0)
data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"


def load_images(path):
    raw = open(path, "rb").read()
    n = int.from_bytes(raw[4:8], "big")
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, 784).astype(np.float32) / 255


def load_labels(path):
    raw = open(path, "rb").read()
    return np.frombuffer(raw, dtype=np.uint8, offset=8).astype(np.int64)


train_x = torch.from_numpy(load_images(f"{data_dir}/train-images-idx3-ubyte"))
train_y = torch.from_numpy(load_labels(f"{data_dir}/train-labels-idx1-ubyte"))
test_x = torch.from_numpy(load_images(f"{data_dir}/t10k-images-idx3-ubyte"))
test_y = torch.from_numpy(load_labels(f"{data_dir}/t10k-labels-idx1-ubyte"))

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
# same starting weights recipe as nnlib: He for the ReLU layer, Xavier for the last, biases 0
nn.init.kaiming_normal_(model[0].weight, nonlinearity="relu")
nn.init.zeros_(model[0].bias)
nn.init.xavier_uniform_(model[2].weight)
nn.init.zeros_(model[2].bias)

opt = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()
epochs, batch = 5, 64
n = train_x.shape[0]

# ---- 1. training speed ----
for epoch in range(1, epochs + 1):
    perm = torch.randperm(n)
    t0 = time.perf_counter()
    total_loss = 0.0
    batches = 0
    for start in range(0, n - batch + 1, batch):
        idx = perm[start:start + batch]
        x, y = train_x[idx], train_y[idx]
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        batches += 1
    secs = time.perf_counter() - t0
    with torch.no_grad():
        acc = (model(test_x).argmax(1) == test_y).float().mean().item() * 100
    print(f"pytorch epoch {epoch}  time {secs:.4f} s  loss {total_loss / batches:.4f}  test acc {acc:.4f}%")

# ---- 2. batch-1 inference latency ----
singles = [test_x[i:i + 1] for i in range(1000)]
with torch.inference_mode():
    for i in range(200):                       # warm-up
        model(singles[i % 1000])
    runs = 10000
    us = []
    for i in range(runs):
        t0 = time.perf_counter()
        model(singles[i % 1000])
        us.append((time.perf_counter() - t0) * 1e6)
us.sort()
print(f"pytorch batch-1 latency:  p50 {us[runs // 2]:.4f} us   p99 {us[runs * 99 // 100]:.4f} us")
