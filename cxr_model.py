import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import os
import gdown

# Google Drive model file ID and output filename
file_id = "11XFgmoX6vqBMwWha0ga-H_D5iUEQMxwd"
output_path = "best_resnet18_cxr.pt"

# Only download if the model file does not exist
if not os.path.exists(output_path):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load(r"C:\Users\Nandhini\Downloads\FYP_final_mod\FYP\best_resnet18_cxr.pt", map_location=device))
model.to(device)
model.eval()

# Class names (keep this same)
class_names = [
    "Normal", "Viral Pneumonia", "Pleural Effusion", "Pneumothorax",
    "Chronic Obstructive Pulmonary Disease (COPD)", "Tuberculosis",
    "Bacterial Pneumonia", "Lung Infections and Fibrosis",
    "Atelectasias", "Pulmonary Abscess"
]

def predict_cxr(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load image and preprocess
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Grad-CAM
    target_layer = model.layer4[-1]  # Last conv layer of ResNet18
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_class = torch.max(probs, 1)
        pred_class = pred_class.item()
        confidence = conf.item()
        predicted_label = class_names[pred_class]

    # Generate Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
    original_img = np.array(img.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(original_img, grayscale_cam, use_rgb=True)

    # Save smaller Grad-CAM image using matplotlib
    gradcam_filename = os.path.basename(image_path).split('.')[0] + '_gradcam.jpg'
    gradcam_path = os.path.join('static/uploads', gradcam_filename)

    plt.figure(figsize=(4, 4))  # 8x6 inches
    plt.imshow(cam_image)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(gradcam_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return predicted_label, confidence, gradcam_path
