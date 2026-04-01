# mainhy.py
import os
import torch
import time
import numpy as np
from datetime import datetime
from train_hy import train_model
from evaluate_hy import evaluate_model
from utils_hy import analyze_model_quick

def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✅ Using device: {device}")
    return device

def main():
    print("🚀 COPD Detection Pipeline (Hybrid Attention CNN) Starting...")

    device = check_device()

    data_root = "/content/data_processed"
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"/content/outputs/hybrid_run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # TRAIN
    print("\n🔁 Starting hybrid model training...\n")
    model, history, checkpoint_path = train_model(
        train_dir=train_dir,
        val_dir=val_dir,
        device=device,
        epochs=50,
        lr=1e-4,
        batch_size=32,
        img_size=224,
        output_dir=output_dir
    )

    # QUICK ANALYSIS (fast)
    print("\n🧠 Performing quick model analysis (params, small latency)...\n")
    analyze_model_quick(model, device, img_size=224, output_dir=output_dir, runs=20)

    # EVALUATE (full)
    print("\n🔍 Evaluating hybrid model on test set...\n")
    # evaluate_model(model, test_dir, device, output_dir, quick=False, history=history, checkpoint_path=checkpoint_path)
    # The evaluate function will load checkpoint internally to ensure exact eval of best model:
    evaluate_model(checkpoint_path, test_dir, device, output_dir, quick=False, history=history)

    print(f"\n✅ Pipeline completed successfully! All outputs saved in: {output_dir}")

if __name__ == "__main__":
    main()
