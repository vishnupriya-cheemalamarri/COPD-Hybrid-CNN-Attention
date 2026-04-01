# utils_hy.py
import os
import time
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve."""
    def __init__(self, patience=7, verbose=False, delta=0.0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score, model):
        score = val_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Saves model when validation score improves."""
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"💾 Best model saved! -> {self.path}")

def analyze_model_quick(model, device, img_size, output_dir, runs=20):
    """
    Quick analysis: counts parameters and measures a small latency sample.
    Saves results to model_analysis.txt
    """
    model.eval()
    model.to(device)
    # detect first conv for channel
    first_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            first_conv = m
            break
    in_ch = 3 if first_conv is None else first_conv.in_channels

    dummy = torch.randn(1, in_ch, min(img_size,128), min(img_size,128)).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧩 Trainable Parameters: {params:,}")

    # latency
    times = []
    with torch.no_grad():
        # warm-up
        for _ in range(3):
            _ = model(dummy)
        for _ in range(runs):
            start = time.time()
            _ = model(dummy)
            times.append((time.time() - start) * 1000)
    mean_latency = np.mean(times)
    std_latency = np.std(times)
    print(f"⏱️ Inference Latency: {mean_latency:.2f} ± {std_latency:.2f} ms over {runs} runs")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "model_analysis.txt"), "w") as f:
        f.write(f"Trainable Parameters: {params:,}\n")
        f.write(f"Inference Latency: {mean_latency:.2f} ± {std_latency:.2f} ms over {runs} runs\n")
    return params, mean_latency
