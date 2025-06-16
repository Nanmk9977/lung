import torch
import torch.nn.functional as F
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import os
import cv2

# ✅ Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ Load trained ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load(r"C:\Users\Nandhini\Downloads\FYP_final_mod\FYP\best_resnet18_cxr.pt", map_location=device))
model.to(device)
model.eval()

# ✅ Class names 
class_names = [
    "Normal", "Viral Pneumonia", "Pleural Effusion", "Pneumothorax",
    "Chronic Obstructive Pulmonary Disease (COPD)", "Tuberculosis",
    "Bacterial Pneumonia", "Lung Infections and Fibrosis",
    "Atelectasias", "Pulmonary Abscess"
]

label_mapping = {name: idx for idx, name in enumerate(class_names)}

# ✅ Transform for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# ✅ Test folder path
test_root = r"C:\Users\Nandhini\Downloads\Xray"
true_labels, pred_labels, all_probs = [], [], []

# ✅ Limit evaluation to N images per class
MAX_IMAGES = 100

# ✅ For Grad-CAM
last_input_tensor = None
last_pred_class = None
last_img_probs = None

for folder_name in os.listdir(test_root):
    if folder_name not in class_names:
        print(f"❌ Skipping unknown folder: {folder_name}")
        continue

    print(f"\n✅ Evaluating class: {folder_name}")
    folder_path = os.path.join(test_root, folder_name)
    class_idx = label_mapping[folder_name]

    image_count = 0
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(folder_path, filename)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Error loading {filename}: {e}")
            continue

        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        true_labels.append(class_idx)
        pred_labels.append(pred)
        all_probs.append(probs.cpu().numpy()[0])

        last_input_tensor = input_tensor
        last_pred_class = pred
        last_img_probs = probs.cpu().numpy()[0]

        print(f"{filename} → Predicted: {class_names[pred]}, Actual: {folder_name}")

        image_count += 1
        if image_count >= MAX_IMAGES:
            break

# 📊 Classification Report
print("\n📊 Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=[class_names[i] for i in sorted(set(true_labels))]))

accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

print(f"\n🔹 Accuracy: {accuracy:.2f}")
print(f"✅ Precision: {precision:.2f}")
print(f"✅ Recall: {recall:.2f}")
print(f"✅ F1 Score: {f1:.2f}")

# 📉 Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("📉 Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()

# 🔍 Grad-CAM + Heatmap
def generate_gradcam(model, input_tensor, target_class):
    model.eval()
    activations, gradients = [], []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fw = model.layer4[-1].register_forward_hook(forward_hook)
    handle_bw = model.layer4[-1].register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    output[0, target_class].backward()

    grad = gradients[0][0].cpu().detach()
    act = activations[0][0].cpu().detach()

    weights = grad.mean(dim=(1, 2))
    cam = torch.sum(weights[:, None, None] * act, dim=0)
    cam = torch.clamp(cam, min=0)
    cam -= cam.min()
    cam /= cam.max()
    cam = cam.numpy()
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img_np = input_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    img_np = (img_np * 0.5 + 0.5) * 255
    img_np = np.uint8(img_np)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # Plot original, heatmap, and Grad-CAM overlay
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_np)
    axs[0].set_title("🖼️ Original Image")
    axs[1].imshow(heatmap)
    axs[1].set_title("🔥 Grad-CAM Heatmap")
    axs[2].imshow(overlay)
    axs[2].set_title(f"📌 Overlay: {class_names[target_class]}")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    handle_fw.remove()
    handle_bw.remove()

# 📌 Visualize Grad-CAM + Probabilities
if last_input_tensor is not None:
    generate_gradcam(model, last_input_tensor, last_pred_class)

    # 📈 Plot Top-5 class probabilities
    top5_indices = np.argsort(last_img_probs)[::-1][:5]
    top5_probs = last_img_probs[top5_indices]
    top5_classes = [class_names[i] for i in top5_indices]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=top5_probs, y=top5_classes, palette='viridis')
    plt.xlabel("Probability")
    plt.title("📊 Top-5 Class Probabilities")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No valid images found for Grad-CAM.")
