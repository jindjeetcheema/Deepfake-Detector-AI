import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from tqdm import tqdm
from src.models.xception import Xception

# -----------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------
TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"
SAVE_PATH = "models/final_model.pth"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------
# DATASET & TRANSFORMS
# -----------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_ds = datasets.ImageFolder(VAL_DIR, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# -----------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------
model = Xception(num_classes=2, pretrained=True).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------------------------------------------
# TRAINING FUNCTION
# -----------------------------------------------------------------
def train_epoch(epoch):
    model.train()
    total_loss, total_correct = 0, 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", colour="green"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == labels).sum().item()

    acc = total_correct / len(train_ds)
    avg_loss = total_loss / len(train_loader)
    print(f"✅ Train Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

# -----------------------------------------------------------------
# VALIDATION FUNCTION
# -----------------------------------------------------------------
def validate():
    model.eval()
    total_correct, total_loss = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
    acc = total_correct / len(val_ds)
    avg_loss = total_loss / len(val_loader)
    print(f"🧪 Val Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
    return acc

# -----------------------------------------------------------------
# TRAIN LOOP
# -----------------------------------------------------------------
best_acc = 0.0
for epoch in range(EPOCHS):
    train_epoch(epoch)
    val_acc = validate()

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"💾 Saved best model (val acc: {best_acc:.4f}) -> {SAVE_PATH}")

print("🎉 Training complete!")
