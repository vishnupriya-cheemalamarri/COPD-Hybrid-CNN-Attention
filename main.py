# main.py
import os
import torch
import time
import numpy as np
from datetime import datetime
from train import train_model
from evaluate import evaluate_model

# Try importing thop for FLOPs calculation (optional)
try:
    from thop import profile
except ImportError:
    profile = None


# -------------------------------------------------
# DEVICE CHECK
# -------------------------------------------------
def check_device():
    """Check and return the best available device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✅ Using device: {device}")
    return device


# -------------------------------------------------
# MODEL ANALYSIS FUNCTION (Supports Quick Mode)
# -------------------------------------------------
def analyze_model(model, device, img_size, output_dir, quick=True, flops_enabled=False, runs=20):
    """
    Faster / more accurate model analysis:
    - quick=True: skip FLOPs and reduce timing runs
    - flops_enabled: run thop only when explicitly requested
    - runs: number of timed forward passes (default 20)
    """
    model.eval()
    model.to(device)

    # Safe first Conv2d detection
    first_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            first_conv = m
            break
    in_channels = first_conv.in_channels if first_conv is not None else 3

    dummy = torch.randn(1, in_channels, img_size, img_size).to(device)

    # 1) Params
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧩 Trainable Parameters: {params:,}")

    # 2) FLOPs (optional / explicit)
    flops_g = None
    if (not quick and profile) or flops_enabled:
        try:
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
            flops_g = flops / 1e9
            print(f"⚙️ FLOPs (approx): {flops_g:.3f} GFLOPs")
        except Exception as e:
            print("⚠️ FLOPs profiling failed:", e)

    # 3) Latency (reliable timing)
    if not quick:
        # warm-up
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy)
            # ensure GPU completed warm-up
            if device.type == "cuda":
                torch.cuda.synchronize()

            import time
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                _ = model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)  # ms

        mean_latency = float(np.mean(times))
        std_latency = float(np.std(times))
        print(f"⏱️ Inference Latency: {mean_latency:.2f} ± {std_latency:.2f} ms over {runs} runs")
    else:
        mean_latency = None
        std_latency = None
        print("⚡ Quick Mode: Skipping latency measurement (use quick=False to measure).")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "model_analysis.txt"), "w") as f:
        f.write(f"Trainable Parameters: {params:,}\n")
        if flops_g is not None:
            f.write(f"FLOPs (approx): {flops_g:.3f} GFLOPs\n")
        if mean_latency is not None:
            f.write(f"Inference Latency: {mean_latency:.2f} ± {std_latency:.2f} ms\n")
        if quick:
            f.write("Quick Analysis Mode Enabled: FLOPs and latency skipped or reduced.\n")

    print(f"\n📄 Model analysis saved at: {output_dir}/model_analysis.txt")


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
def main():
    print("🚀 COPD Detection Pipeline (SimpleCNN) Starting...")

    # Step 1: Device
    device = check_device()

    # Step 2: Data directories
    data_root = "/content/data_processed"
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    # Step 3: Output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"/content/outputs/simplecnn_run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Step 4: Train model
    print("\n🔁 Starting model training...\n")
    result = train_model(
        train_dir=train_dir,
        val_dir=val_dir,
        device=device,
        epochs=50,
        lr=1e-4,
        batch_size=64,
        img_size=224,
        output_dir=output_dir
    )

    # train_model may return just model or (model, history)
    if isinstance(result, tuple) and len(result) >= 1:
        model, history = result[0], (result[1] if len(result) > 1 else None)
    else:
        model, history = result, None

    # Model analysis (Quick Mode)
    print("\n🧠 Performing model analysis...\n")
    analyze_model(model, device, img_size=224, output_dir=output_dir, quick=True)

    # Evaluate on test set (Full Mode). Try calling with history if available.
    print("\n🔍 Evaluating model on test set...\n")
    try:
        if history is not None:
            evaluate_model(model, test_dir, device, output_dir, quick=False, history=history)
        else:
            evaluate_model(model, test_dir, device, output_dir, quick=False)
    except TypeError:
        # Fallback: try the original simple call
        evaluate_model(model, test_dir, device, output_dir)

    print(f"\n✅ Pipeline Finished Successfully! All outputs saved in '{output_dir}' folder.\n")


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
if __name__ == "__main__":
    main()
