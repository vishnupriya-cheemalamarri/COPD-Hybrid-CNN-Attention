# evaluate_hy.py
import os
import json
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models_hy import HybridAttentionModel


# -----------------------------------------------------
# 🧠 Load trained model
# -----------------------------------------------------
def load_model_from_checkpoint(checkpoint_path, device, num_classes=2,
                               backbone_name='resnet18', use_cbam=True, use_se=True, use_self_att=True):
    model = HybridAttentionModel(
        backbone_name=backbone_name,
        pretrained=False,
        num_classes=num_classes,
        use_cbam=use_cbam,
        use_se=use_se,
        use_self_att=use_self_att
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# -----------------------------------------------------
# 📈 Plot accuracy & loss curves
# -----------------------------------------------------
def plot_training_curves_from_history(history, output_dir):
    if history is None:
        print("⚠️ No training history provided, skipping curve plotting.")
        return

    # helper to get array from multiple possible key names
    def get_array(d, keys):
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    a = np.array(d[k], dtype=float)
                    return a
                except Exception:
                    continue
        return np.array([])

    train_acc = get_array(history, ["train_acc", "train_accs", "trainAcc", "train_accuracy"])
    val_acc = get_array(history, ["val_acc", "val_accs", "valAcc", "val_accuracy"])
    train_loss = get_array(history, ["train_loss", "train_losses", "trainLoss"])
    val_loss = get_array(history, ["val_loss", "val_losses", "valLoss"])

    # safe percent conversion
    def to_percent(a):
        if a.size == 0:
            return a
        if a.max() <= 1.0:
            return a * 100.0
        return a

    train_acc = to_percent(train_acc)
    val_acc = to_percent(val_acc)

    if train_acc.size == 0 and val_acc.size == 0 and train_loss.size == 0 and val_loss.size == 0:
        print("⚠️ Empty history arrays — skipping plotting.")
        return

    plt.figure()
    if train_acc.size:
        plt.plot(train_acc, label="Train Acc (%)")
    if val_acc.size:
        plt.plot(val_acc, label="Val Acc (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve_hy.png"), dpi=200)
    plt.close()

    plt.figure()
    if train_loss.size:
        plt.plot(train_loss, label="Train Loss")
    if val_loss.size:
        plt.plot(val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve_hy.png"), dpi=200)
    plt.close()

    print(f"📈 Saved accuracy & loss curves in: {output_dir}")


# -----------------------------------------------------
# 🔍 Overfitting Analysis
# -----------------------------------------------------
def analyze_overfitting(history_path, output_dir):
    if not os.path.exists(history_path):
        return

    with open(history_path, "r") as f:
        hist = json.load(f)

    train_acc, val_acc = np.array(hist["train_acc"]), np.array(hist["val_acc"])
    train_loss, val_loss = np.array(hist["train_loss"]), np.array(hist["val_loss"])

    diff_loss = val_loss[-1] - train_loss[-1]
    diff_acc = val_acc[-1] - train_acc[-1]

    print("\n🧩 Overfitting Analysis:")
    print(f"→ Final Train Acc: {train_acc[-1]*100:.2f}%, Val Acc: {val_acc[-1]*100:.2f}%")
    print(f"→ Final Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")

    if diff_loss > 0.1 or diff_acc < -0.02:
        print("⚠️ Overfitting detected! Validation diverges from training.")
    else:
        print("✅ No strong overfitting signs detected.")

    # plot
    plt.figure(figsize=(8,5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Overfitting Analysis')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overfitting_analysis.png'))
    plt.close()
    print(f"📊 Overfitting plot saved: {output_dir}/overfitting_analysis.png")


# -----------------------------------------------------
# 🧪 Evaluate model on test data
# -----------------------------------------------------
def evaluate_model(checkpoint_path, test_dir, device, output_dir, history=None, history_path=None, quick=False):
    print("\n🧪 Starting Hybrid Attention Model Evaluation...")

    # load history if not provided (support dict or file)
    loaded_history_path = None
    if history is None:
        history, loaded_history_path = _load_history_from_files(output_dir, history_path)
        if history is not None:
            print(f"ℹ️ Loaded training history from: {loaded_history_path}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    num_classes = len(test_dataset.classes)

    print(f"📂 Loaded {len(test_dataset)} test images across {num_classes} classes: {test_dataset.classes}")

    model = load_model_from_checkpoint(checkpoint_path, device, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()

    total_loss, total_correct = 0.0, 0
    all_preds, all_labels = [], []

    start = time.time()
    model.eval()
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            total_correct += torch.sum(preds == labels).item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if quick and i > 5:
                break

    end = time.time()
    test_acc = total_correct / len(test_dataset) * 100
    test_loss = total_loss / len(test_dataset)

    print(f"\n✅ Test Accuracy: {test_acc:.2f}%")
    print(f"📉 Test Loss: {test_loss:.4f}")
    print(f"⏱ Evaluation Time: {end - start:.2f} sec")

    # Save classification report
    report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, digits=4)
    report_path = os.path.join(output_dir, "classification_report_hy.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {test_acc:.2f}%\nTest Loss: {test_loss:.4f}\n\n")
        f.write(report)
    print(f"📄 Classification report saved at: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Hybrid Attention Model")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix_hy.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"🖼️ Confusion matrix saved: {cm_path}")

    # Sample predictions (unnormalize for display)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    model.cpu()
    for ax in axes.flat:
        idx = random.randint(0, len(test_dataset) - 1)
        img, label = test_dataset[idx]
        with torch.no_grad():
            inp = img.unsqueeze(0)
            if inp.shape[1] == 1:
                inp = inp.repeat(1, 3, 1, 1)
            out = model(inp)
            pred = torch.argmax(out, dim=1).item()
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = (img_np * std + mean)
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        ax.set_title(f"T: {test_dataset.classes[label]}\nP: {test_dataset.classes[pred]}")
        ax.axis("off")
    plt.tight_layout()
    pred_path = os.path.join(output_dir, "sample_predictions_hy.png")
    plt.savefig(pred_path, dpi=200)
    plt.close()
    print(f"📸 Sample predictions saved: {pred_path}")

    # Plot training curves & analyze overfitting
    if history is not None:
        plot_training_curves_from_history(history, output_dir)
        # pass a valid history file path to analyzer if available
        hist_path_for_analysis = loaded_history_path or history_path or os.path.join(output_dir, "training_history_hy.json")
        analyze_overfitting(hist_path_for_analysis, output_dir)
    else:
        print("⚠️ No training history available for plotting/overfitting analysis.")

    print("\n✅ Evaluation completed successfully.")
    return test_acc, test_loss


def _load_history_from_files(output_dir, history_path=None):
    """Try JSON first, then NPZ. Return dict or None and the path used."""
    # priority explicit path
    if history_path and os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                h = json.load(f)
            return h, history_path
        except Exception:
            pass
    # common json location
    jp = os.path.join(output_dir, "training_history_hy.json")
    if os.path.exists(jp):
        try:
            with open(jp, "r") as f:
                h = json.load(f)
            return h, jp
        except Exception:
            pass
    # npz (train_history_hy.npz)
    npz_path = os.path.join(output_dir, "train_history_hy.npz")
    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path, allow_pickle=True)
            h = {}
            for k in data.files:
                try:
                    h[k] = data[k].tolist()
                except Exception:
                    h[k] = data[k]
            return h, npz_path
        except Exception:
            pass
    return None, None


# -----------------------------------------------------
# 🚀 Run evaluation directly
# -----------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "/content/outputs/hybrid_run_2025-10-16_03-50-32/best_model_hy.pth"
    output_dir = "/content/outputs/hybrid_run_2025-10-16_03-50-32"
    test_dir = "/content/dataset/test"  # change to your test directory
    history_path = os.path.join(output_dir, "training_history_hy.json")

    evaluate_model(
        checkpoint_path=checkpoint_path,
        test_dir=test_dir,
        device=device,
        output_dir=output_dir,
        history_path=history_path,
        quick=False
    )
