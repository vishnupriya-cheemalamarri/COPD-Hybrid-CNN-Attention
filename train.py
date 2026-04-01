# train.py
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# ------------------------------
# 🔒 Reproducibility
# ------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------
# Helper: detect image channels
# ------------------------------
def detect_image_channels(folder):
    """
    Detects if dataset images are grayscale (1 channel) or RGB (3 channels).
    """
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                try:
                    im = Image.open(os.path.join(root, f))
                    mode = im.mode
                    im.close()
                    return 1 if mode == "L" else 3
                except Exception:
                    continue
    return 3  # default to RGB if uncertain


# ------------------------------
# 📦 Data Loaders (with channel fix)
# ------------------------------
def get_data_loaders(train_dir, val_dir, batch_size=64, img_size=224):
    """
    Loads training and validation datasets with consistent channel format (RGB).
    """
    channels = detect_image_channels(train_dir)

    # Standard ImageNet normalization (for RGB)
    norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # Force all images to RGB (prevents channel mismatch)
    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),  
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    transform_val = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),  # 👈 Always RGB
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Always return 3 since we convert all to RGB
    return train_loader, val_loader, 3


# ------------------------------
# 🧠 Simple CNN Architecture
# ------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.flatten_dim = 128 * 1 * 1
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ------------------------------
# 🚀 Training Loop with Validation + Curves + Early Stopping
# ------------------------------
def train_model(train_dir, val_dir, device, epochs=50, lr=5e-5, batch_size=64, img_size=224, output_dir="outputs"):
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)
    print("📦 Preparing data...")

    train_loader, val_loader, in_channels = get_data_loaders(train_dir, val_dir, batch_size, img_size)

    # Sanity checks
    if len(train_loader.dataset) == 0:
        raise RuntimeError(f"No training images found in {train_dir}")
    if len(val_loader.dataset) == 0:
        raise RuntimeError(f"No validation images found in {val_dir}")

    num_classes = len(train_loader.dataset.classes)
    model = SimpleCNN(in_channels=in_channels, num_classes=num_classes).to(device)
    print(f"📚 Model Initialized: SimpleCNN ({in_channels}-channel input, {num_classes} classes)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    patience = 7
    wait = 0

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"🧪 Epoch {epoch}/{epochs} - Train", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / total
        epoch_train_acc = 100.0 * correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Validation", leave=False)
            for images, labels in pbar_val:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(f"✅ Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}%")
        print(f"📊 Val Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}%")

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"💾 Best model saved! (val_acc: {best_val_acc:.2f}%)")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("🛑 Early stopping triggered.")
                break

    total_training_time = time.time() - start_time
    print("\n🎉 Training complete.")
    print(f"🔝 Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"⏱️ Total training time: {total_training_time:.2f} seconds")

    # Save accuracy and loss curves
    plt.figure()
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy (SimpleCNN)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "epoch_accuracy_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model Loss (SimpleCNN)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "epoch_loss_curve.png"))
    plt.close()

    # Save training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_val_acc": best_val_acc,
        "training_time": total_training_time
    }
    np.savez_compressed(os.path.join(output_dir, "train_history.npz"), **history)
    print(f"\n📈 Training curves and history saved in: {output_dir}")

    return model, history
