import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from datasets import load_dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse # To easily pass image paths

# --- Configuration (Should match training script) ---
DATASET_NAME = "kannanwisen/Indian-Traffic-Sign-Classification"
MODEL_NAME = "resnet50" # Ensure this matches the trained model
# MODEL_NAME = "resnet18" # Use this if you trained ResNet18 instead
INPUT_SIZE = 224
# Default path to the weights saved by the enhanced training script
DEFAULT_MODEL_WEIGHTS_PATH = f'best_traffic_sign_model_{MODEL_NAME}_finetuned.pth'

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Expecting model: {MODEL_NAME}")

# --- Transforms (Must match val_test_transform from training) ---
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

preprocess_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)), # Resize directly to input size
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# --- Function to Load Class Names ---
def get_class_info(dataset_name):
    """Loads dataset metadata to get class names and count."""
    print(f"Loading dataset metadata from '{dataset_name}' to get class names...")
    try:
        hf_dataset = load_dataset(dataset_name, split='train', trust_remote_code=True)
        num_classes = hf_dataset.features['label'].num_classes
        class_names = hf_dataset.features['label'].names
        print(f"Found {num_classes} classes.")
        return num_classes, class_names
    except Exception as e:
        print(f"Error loading dataset features: {e}")
        print("Please ensure the dataset name is correct and you have internet access.")
        print("You might need to manually provide num_classes if loading fails.")
        return None, None # Return None to indicate failure

# --- Build Model Function (Copied from training script) ---
def build_model(model_name, num_classes, pretrained=False): # Set pretrained=False for loading
    """Builds the specified model architecture."""
    # We don't need pretrained weights here, just the architecture skeleton
    weights = None # Important: Don't download weights again
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Replace the classifier head to match the trained model's structure
    if hasattr(model, 'fc'): # ResNets
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif hasattr(model, 'classifier'): # Example for other models
         if isinstance(model.classifier, nn.Sequential):
             num_ftrs = model.classifier[-1].in_features
             model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
         else:
             num_ftrs = model.classifier.in_features
             model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        raise AttributeError(f"Model {model_name} doesn't have a standard 'fc' or 'classifier' attribute to replace.")

    # We are not printing the pretrained status here as it's always False
    print(f"Built {model_name} architecture with {num_classes} output classes.")
    return model

# --- Function to Load Model ---
def load_trained_model(weights_path, model_name, num_classes):
    """Loads the specified model architecture and trained weights."""
    # Build the model architecture first
    model = build_model(model_name, num_classes, pretrained=False)

    # Load the trained weights
    print(f"Loading trained weights from '{weights_path}'...")
    if not os.path.exists(weights_path):
        print(f"Error: Model weights file not found at '{weights_path}'")
        return None # Return None to indicate failure
    try:
        # Load the state dictionary
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print(f"Ensure the weights file '{weights_path}' corresponds to the {model_name} architecture with {num_classes} classes.")
        return None # Return None to indicate failure

    model = model.to(device)
    model.eval() # Set the model to evaluation mode (IMPORTANT!)
    return model

# --- Prediction Function ---
@torch.no_grad() # Disable gradient calculations for inference
def predict_image(model, image_path, transform, class_names, device):
    """Loads an image, preprocesses it, and returns the predicted class name and confidence."""
    try:
        img = Image.open(image_path).convert('RGB') # Ensure image is RGB
        original_img = img.copy() # Keep original for display
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None, None, None
    except Exception as e:
        print(f"Error opening image '{image_path}': {e}")
        return None, None, None

    # Preprocess the image using the SAME transform as validation/test
    img_tensor = transform(img)
    # Add batch dimension (model expects B x C x H x W)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device) # Move tensor to the correct device

    # Perform inference
    outputs = model(img_tensor)
    # Apply softmax to get probabilities/confidences
    probabilities = torch.softmax(outputs, dim=1)
    # Get the top prediction (index and probability)
    confidence_tensor, predicted_idx_tensor = torch.max(probabilities, 1)

    predicted_idx = predicted_idx_tensor.item()
    confidence = confidence_tensor.item()

    # Map index to class name
    if class_names and 0 <= predicted_idx < len(class_names):
        predicted_class_name = class_names[predicted_idx]
    else:
        predicted_class_name = f"Class Index {predicted_idx}" # Fallback if names aren't available

    return original_img, predicted_class_name, confidence

# --- Main Execution ---
if __name__ == "__main__":
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description=f"Predict traffic sign class using a trained {MODEL_NAME} model.")
    parser.add_argument(
        "image_paths",
        metavar="IMAGE_PATH",
        type=str,
        nargs='+', # Allows one or more image paths
        help="Path(s) to the input image file(s)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_WEIGHTS_PATH,
        help=f"Path to the trained model weights file (default: {DEFAULT_MODEL_WEIGHTS_PATH})."
    )
    parser.add_argument(
        '--show',
        action='store_true', # If this flag is present, show will be True
        help="Display the image with its prediction."
    )

    args = parser.parse_args()

    # --- Load Class Info ---
    num_classes, class_names = get_class_info(DATASET_NAME)
    if num_classes is None:
        exit() # Exit if we couldn't get necessary info

    # --- Load Model ---
    model = load_trained_model(args.model_path, MODEL_NAME, num_classes)
    if model is None:
         exit() # Exit if model loading failed

    # --- Predict for each image provided ---
    for img_path in args.image_paths:
        print("-" * 30)
        print(f"Predicting for: {img_path}")

        original_image, prediction, confidence = predict_image(
            model,
            img_path,
            preprocess_transform, # Use the val/test transform
            class_names,
            device
        )

        if prediction is not None:
            print(f"==> Predicted Class: {prediction}")
            print(f"==> Confidence: {confidence:.4f}")

            # --- Display Image with Prediction (Optional) ---
            if args.show and original_image is not None:
                plt.figure(figsize=(6, 6))
                plt.imshow(original_image)
                plt.title(f"Prediction: {prediction}\nConfidence: {confidence:.2f}")
                plt.axis('off')
                plt.show()
        else:
            print(f"Could not process image: {img_path}")

    print("-" * 30)
    print("Prediction complete.")
    # plt.close('all') # Optionally close all plot windows at the end