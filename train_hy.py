# train_hy.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from models_hy import HybridAttentionModel
from utils_hy import EarlyStopping

# reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_loaders(train_dir, val_dir, batch_size=32, img_size=224):
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1,0.1,0.05,0.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])  # for pre-trained ResNet
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # ImageFolder expects class subfolders
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

def train_model(train_dir, val_dir, device, epochs=50, lr=5e-5, batch_size=32, img_size=224, output_dir="outputs"):
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)
    print("📦 Preparing data...")
    train_loader, val_loader = get_data_loaders(train_dir, val_dir, batch_size, img_size)

    if len(train_loader.dataset) == 0:
        raise ValueError(f"No training images found in {train_dir}")
    if len(val_loader.dataset) == 0:
        raise ValueError(f"No validation images found in {val_dir}")

    num_classes = len(train_loader.dataset.classes)
    print(f"📚 Classes: {train_loader.dataset.classes}")

    # model: can be configured with attention toggles
    model = HybridAttentionModel(backbone_name='resnet18', pretrained=True,
                                 num_classes=num_classes, use_cbam=True, use_se=True, use_self_att=True)
    model = model.to(device)
    print(f"📚 Model Initialized: HybridModel (expected 3-channel input, {num_classes} classes)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    early_stopper = EarlyStopping(patience=7, verbose=True, path=os.path.join(output_dir, "best_model_hy.pth"))

    # history
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\n🧪 Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for imgs, labels in tqdm(train_loader, desc="Training", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            # If grayscale dataset, convert to 3-channel by repeating
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1,3,1,1)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = running_corrects / total * 100

        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validation", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                if imgs.shape[1] == 1:
                    imgs = imgs.repeat(1,3,1,1)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_corrects / val_total * 100

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"✅ Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.2f}%")
        print(f"📊 Val Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}%")

        # scheduler step (monitor val acc)
        scheduler.step(epoch_val_acc)

        # Early stopping & saving best
        early_stopper(epoch_val_acc, model)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc

        if early_stopper.early_stop:
            print("🛑 Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print("\n🎉 Training complete.")
    print(f"🔝 Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"⏱️ Total training time: {total_time:.2f} seconds")

    # Save history dict (include train_classes for later)
    history = {
        "train_losses": history['train_loss'],
        "val_losses": history['val_loss'],
        "train_accs": history['train_acc'],
        "val_accs": history['val_acc'],
        "best_val_acc": best_val_acc,
        "training_time": total_time,
        "train_classes": train_loader.dataset.classes
    }
    np.savez_compressed(os.path.join(output_dir, "train_history_hy.npz"), **history)

    # Also save a simple JSON expected by evaluate_hy (use fractions [0..1] for accuracies)
    try:
        import json
        history_json = {
            "train_acc": [a / 100.0 for a in history['train_acc']],
            "val_acc":   [a / 100.0 for a in history['val_acc']],
            "train_loss": history['train_loss'],
            "val_loss":   history['val_loss']
        }
        with open(os.path.join(output_dir, "training_history_hy.json"), "w") as f:
            json.dump(history_json, f)
    except Exception as e:
        print("⚠️ Failed to write training_history_hy.json:", e)

    print(f"\n📈 Training curves and history saved in: {output_dir}")

    return model, history, os.path.join(output_dir, "best_model_hy.pth")
