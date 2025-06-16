# train_9class_cxr_densenet.py
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

# Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
epochs = 15
img_size = 224
lr = 1e-4

# 1. Transforms & Dataset
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = "Xray_9_classes"
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Stratified split
targets = [s[1] for s in dataset.samples]
train_idx, val_idx = train_test_split(range(len(targets)), test_size=0.2, stratify=targets, random_state=42)
train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=2)

# 2. Model: DenseNet121
model = models.densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, len(dataset.classes))
model = model.to(device)

# 3. Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

# 4. Training loop with validation
best_acc = 0.0
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for imgs, labels in train_dl:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}: Train Loss = {running_loss/len(train_dl):.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_densenet121_cxr.pt")
        print(f"✅ Best model saved at epoch {epoch+1} with accuracy: {acc:.2f}%")

    scheduler.step()

print("Training complete.")
