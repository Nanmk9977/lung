import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gdown
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.model_targets import SoftmaxOutputTarget
from pytorch_grad_cam.utils.image import preprocess_image

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure model file is present
model_path = "best_resnet18_cxr.pt"
gdrive_url = "https://drive.google.com/uc?id=11XFgmoX6vqBMwWha0ga-H_D5iUEQMxwd"
if not os.path.exists(model_path):
    print("⏬ Downloading model file...")
    gdown.download(gdrive_url, model_path, quiet=False)

# Load ResNet18 model with 10 output classes
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded successfully with 10 output classes.")

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Ensure 3-channel input
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet stats
                         [0.229, 0.224, 0.225])
])

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    return pred_class, confidence, probs.squeeze().cpu().numpy()

def generate_gradcam(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Set up GradCAM
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    targets = [ClassifierOutputTarget(torch.argmax(model(input_tensor)).item())]

    # Raw image for overlay
    rgb_img = np.array(image.resize((224, 224))) / 255.0
    rgb_img = np.float32(rgb_img)

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Save Grad-CAM result
    os.makedirs('static/uploads', exist_ok=True)
    gradcam_filename = os.path.basename(image_path).split('.')[0] + '_gradcam.jpg'
    gradcam_path = os.path.join('static/uploads', gradcam_filename)
    cv2.imwrite(gradcam_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    return gradcam_path
