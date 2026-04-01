# evaluate.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


def plot_training_curves(history, output_dir):
    """Plot and save training accuracy and loss curves."""
    if not history:
        print("⚠️ No training history found — skipping accuracy/loss curve plotting.")
        return

    # Extract history data
    train_acc = history.get('train_acc', [])
    val_acc = history.get('val_acc', [])
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])

    # Accuracy curve
    plt.figure()
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Model Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'))
    plt.close()

    # Loss curve
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Model Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    print("📊 Accuracy and Loss curves saved.")


def evaluate_model(model, test_dir, device, output_dir, quick=False, history=None):
    """
    Evaluate model performance on the test dataset and visualize results.

    Args:
        model: Trained PyTorch model
        test_dir: Path to test dataset
        device: torch.device
        output_dir: Directory to save results
        quick: If True, runs faster (skips plots)
        history: Dictionary containing training metrics (optional)
    """

    print("🧪 Starting model evaluation...")

    # -------------------------------
    # Transforms
    # -------------------------------
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # -------------------------------
    # Dataset & DataLoader
    # -------------------------------
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # -------------------------------
    # Evaluation loop
    # -------------------------------
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct = 0.0, 0
    all_preds, all_labels = [], []

    start_time = time.time()

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Quick mode – limit to 10 batches
            if quick and i > 10:
                break

    end_time = time.time()
    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / len(test_dataset) * 100

    print(f"\n✅ Test Accuracy: {accuracy:.2f}%")
    print(f"📉 Test Loss: {avg_loss:.4f}")
    print(f"⏱️ Evaluation Time: {(end_time - start_time):.2f} seconds")

    # -------------------------------
    # Save Classification Report
    # -------------------------------
    report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
        f.write(f"Test Loss: {avg_loss:.4f}\n\n")
        f.write(report)
    print(f"📄 Classification report saved at: {report_path}")

    # -------------------------------
    # Skip heavy visualizations in quick mode
    # -------------------------------
    if quick:
        print("\n⚡ Quick evaluation mode: Skipped confusion matrix and sample predictions.")
        plot_training_curves(history, output_dir)
        return accuracy, avg_loss

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"🖼️ Confusion matrix saved at: {cm_path}")

    # -------------------------------
    # Random Predictions Visualization
    # -------------------------------
    print("🎨 Generating random prediction samples...")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    model.cpu()
    for i, ax in enumerate(axes.flat):
        idx = np.random.randint(len(test_dataset))
        image, label = test_dataset[idx]
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            pred = torch.argmax(output, dim=1).item()
        ax.imshow(image.permute(1, 2, 0))
        ax.set_title(f"True: {test_dataset.classes[label]}\nPred: {test_dataset.classes[pred]}")
        ax.axis("off")
    plt.tight_layout()
    vis_path = os.path.join(output_dir, "sample_predictions.png")
    plt.savefig(vis_path)
    plt.close()
    print(f"📸 Sample predictions saved at: {vis_path}")

    # -------------------------------
    # Plot Accuracy & Loss Curves
    # -------------------------------
    plot_training_curves(history, output_dir)

    print("\n✅ Evaluation complete! All results saved successfully.\n")
    return accuracy, avg_loss
