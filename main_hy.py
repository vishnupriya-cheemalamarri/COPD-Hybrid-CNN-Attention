import os
import argparse
from datetime import datetime
import time
import numpy as np
import torch

# training / evaluation for hybrid model (to implement next)
from train_hy import train_model_hy
from evaluate_hy import evaluate_model_hy

# optional FLOPs
try:
    from thop import profile
except ImportError:
    profile = None


def check_device():
    """Return best available device and print it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✅ Using device: {device}")
    return device


def analyze_model(model, device, img_size, output_dir, quick=True, flops_enabled=False, runs=20):
    """
    Lightweight model analysis:
    - Prints & saves trainable params
    - Optional FLOPs (if thop installed and flops_enabled=True)
    - Optional latency measurement (quick=False)
    """
    model.eval()
    model.to(device)

    # find first Conv2d module to infer channels (safe fallback to 3)
    in_channels = 3
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            in_channels = m.in_channels
            break

    dummy = torch.randn(1, in_channels, img_size, img_size).to(device)

    # params
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [f"Trainable Parameters: {total_params:,}"]
    print(f"\n🧩 Trainable Parameters: {total_params:,}")

    # FLOPs (optional)
    if flops_enabled and profile is not None:
        try:
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
            flops_g = flops / 1e9
            lines.append(f"FLOPs (approx): {flops_g:.3f} GFLOPs")
            print(f"⚙️ FLOPs (approx): {flops_g:.3f} GFLOPs")
        except Exception as e:
            lines.append(f"FLOPs profiling failed: {e}")
            print("⚠️ FLOPs profiling failed:", e)
    else:
        if flops_enabled:
            print("⚠️ thop not installed — skip FLOPs. Install thop if needed.")
        else:
            print("⚡ Quick Mode: FLOPs disabled.")

    # Latency (optional)
    if not quick:
        # warm-up
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()

            import time as _time
            times = []
            for _ in range(runs):
                t0 = _time.perf_counter()
                _ = model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = _time.perf_counter()
                times.append((t1 - t0) * 1000.0)
        mean_latency = float(np.mean(times))
        std_latency = float(np.std(times))
        lines.append(f"Inference Latency (ms): {mean_latency:.2f} ± {std_latency:.2f}")
        print(f"⏱️ Inference Latency: {mean_latency:.2f} ± {std_latency:.2f} ms over {runs} runs")
    else:
        print("⚡ Quick Mode: Skipping latency measurement (use quick=False to measure).")

    # save file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "model_analysis.txt"), "w") as f:
        f.write("\n".join(lines))

    print(f"\n📄 Model analysis saved at: {os.path.join(output_dir, 'model_analysis.txt')}")


def parse_args():
    p = argparse.ArgumentParser(description="Hybrid model pipeline (main_hy.py)")
    # default changed to your provided data root
    p.add_argument("--data_root", default="/content/data_processed",
                    help="Root folder with train/val/test subfolders")
    p.add_argument("--output_root", default=os.path.join(os.getcwd(), "outputs"),
                   help="Root output folder")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--quick_analysis", action="store_true",
                   help="Run quick model analysis (skip latency and FLOPs)")
    p.add_argument("--flops", action="store_true", help="Compute FLOPs (requires thop)")
    return p.parse_args()


# add dataset verification helper
def verify_dataset_dirs(data_root):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")
    missing = [p for p in (train_dir, val_dir, test_dir) if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError(
            f"Dataset folders missing: {missing}\n"
            f"Make sure your dataset root contains 'train', 'val', and 'test' subfolders with class folders inside.\n"
            f"Or run with: python main_hy.py --data_root \"/path/to/your/data_root\""
        )
    return train_dir, val_dir, test_dir


def main():
    print("🚀 COPD Detection Pipeline (Hybrid Model) Starting...")
    args = parse_args()

    device = check_device()

    # verify dataset layout and get canonical paths
    train_dir, val_dir, test_dir = verify_dataset_dirs(args.data_root)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_root, f"hybrid_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Train
    print("\n🔁 Starting hybrid model training...\n")
    result = train_model_hy(
        train_dir=train_dir,
        val_dir=val_dir,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        img_size=args.img_size,
        output_dir=output_dir
    )

    # Accept either model or (model, history)
    if isinstance(result, tuple):
        model = result[0]
        history = result[1] if len(result) > 1 else None
    else:
        model = result
        history = None

    # Analysis
    print("\n🧠 Performing model analysis...\n")
    analyze_model(model, device, img_size=args.img_size, output_dir=output_dir,
                  quick=args.quick_analysis, flops_enabled=args.flops)

    # Evaluate
    print("\n🔍 Evaluating hybrid model on test set...\n")
    # evaluate_model_hy should accept history and save curves/reports inside its implementation
    try:
        evaluate_model_hy(
            model=model,
            test_dir=test_dir,
            device=device,
            output_dir=output_dir,
            img_size=args.img_size,
            history=history
        )
    except TypeError:
        # fallback if evaluate signature differs
        evaluate_model_hy(model=model, test_dir=test_dir, device=device, output_dir=output_dir, img_size=args.img_size)

    print(f"\n✅ Pipeline completed successfully! All outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()