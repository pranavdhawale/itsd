import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import transforms
import torchvision.models as models
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import os
import time

# --- Configuration ---
DATASET_NAME = "kannanwisen/Indian-Traffic-Sign-Classification"
MODEL_NAME = "resnet50" # Using ResNet50 now
# MODEL_NAME = "resnet18" # Can switch back if needed
BEST_MODEL_PATH = f'best_traffic_sign_model_{MODEL_NAME}_finetuned.pth'
FINAL_MODEL_PATH = f'final_traffic_sign_model_{MODEL_NAME}_finetuned.pth'

# Training Hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = 2 # Adjust based on your system
INPUT_SIZE = 224
# Phase 1 (Classifier Training)
LR_CLASSIFIER = 1e-3
NUM_EPOCHS_CLASSIFIER = 5 # Train classifier head for a few epochs
WEIGHT_DECAY_CLASSIFIER = 1e-4
# Phase 2 (Fine-tuning)
LR_FINETUNE = 5e-5 # Much lower learning rate for fine-tuning
NUM_EPOCHS_FINETUNE = 15 # More epochs for fine-tuning
WEIGHT_DECAY_FINETUNE = 1e-4
# Scheduler & Early Stopping
SCHEDULER_PATIENCE = 2 # ReduceLR patience
EARLY_STOPPING_PATIENCE = 5 # Stop if val acc doesn't improve for this many epochs
LABEL_SMOOTHING = 0.1 # Apply label smoothing

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # Might make things slightly slower, but more reproducible
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Using Model: {MODEL_NAME}")

# --- Transforms ---
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    # transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # Can be aggressive
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
    # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)), # Optional: Cutout/Erasing
])

# Use the same simple transform for validation and test
val_test_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# --- Dataset Class (Unchanged) ---
class TrafficSignDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.transform = transform
        if isinstance(hf_dataset, Subset):
            self.indices = hf_dataset.indices
            self.original_dataset = hf_dataset.dataset
        else:
            self.indices = list(range(len(hf_dataset)))
            self.original_dataset = hf_dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        sample = self.original_dataset[original_idx]
        image = sample['image'].convert('RGB')
        label = sample['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Visualization Function (Unchanged, adjusted call slightly) ---
def visualize_augmentations(dataset, num_samples=5):
    # Ensure dataset has samples before proceeding
    if len(dataset) == 0:
        print("Visualization skipped: Dataset is empty.")
        return

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    # Create a temporary dataset instance *only* for visualization with train transforms
    viz_dataset = TrafficSignDataset(dataset.original_dataset, transform=train_transform)
    viz_dataset.indices = dataset.indices # Use the same indices as the input dataset

    for i in range(num_samples):
        random_subset_idx = np.random.randint(len(viz_dataset))
        img, label = viz_dataset[random_subset_idx]
        img = img.cpu().numpy().transpose((1, 2, 0)) # Move to CPU if needed
        img = np.clip(img * imagenet_std + imagenet_mean, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.suptitle("Sample Augmentations (using train_transform)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plt.show(block=False)
    plt.pause(1)

# --- Build Model Function ---
def build_model(model_name, num_classes, pretrained=True):
    """Builds the specified model."""
    weights = None
    if pretrained:
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 # Use V2 weights
            model = models.resnet50(weights=weights)
        # Add other models here (e.g., efficientnet) if needed
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
    else:
         if model_name == "resnet18":
             model = models.resnet18(weights=None)
         elif model_name == "resnet50":
             model = models.resnet50(weights=None)
         else:
             raise ValueError(f"Unsupported model name: {model_name}")

    # Replace the classifier head
    if hasattr(model, 'fc'): # ResNets
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif hasattr(model, 'classifier'): # Some other models like EfficientNet
         # Example for EfficientNet - adjust if using a different architecture
         if isinstance(model.classifier, nn.Sequential):
             num_ftrs = model.classifier[-1].in_features
             model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
         else: # Simple Linear layer
             num_ftrs = model.classifier.in_features
             model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        raise AttributeError(f"Model {model_name} doesn't have a standard 'fc' or 'classifier' attribute to replace.")

    print(f"Built {model_name} with {num_classes} output classes. Pretrained: {pretrained}")
    return model

# --- Training Loop Function (Enhanced) ---
def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs, phase_name,
                early_stopping_patience=None):
    """Trains the model, tracks validation accuracy, implements early stopping."""
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    epochs_no_improve = 0 # For early stopping

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n--- Starting {phase_name} Phase ---")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 15)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                loader = dataloaders['train']
            else:
                model.eval()   # Set model to evaluate mode
                loader = dataloaders['val']

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            progress_bar = tqdm(loader, desc=f"{phase.capitalize()} Epoch {epoch+1}", leave=False)
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                batch_loss = loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == labels.data)
                running_loss += batch_loss
                running_corrects += batch_corrects
                total_samples += inputs.size(0)

                progress_bar.set_postfix(
                    loss=(running_loss / total_samples if total_samples > 0 else 0.0),
                    acc=(running_corrects.double() / total_samples if total_samples > 0 else 0.0)
                )

            # Ensure total_samples is not zero before division
            if total_samples == 0:
                print(f"Warning: No samples processed in {phase} phase for epoch {epoch+1}. Check dataloader.")
                epoch_loss = 0.0
                epoch_acc = 0.0
            else:
                epoch_loss = running_loss / total_samples
                epoch_acc = running_corrects.double() / total_samples

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item()) # Use .item() for scalar tensors
            else: # Validation phase
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

                # Scheduler step (based on validation accuracy)
                if scheduler:
                    # Check if scheduler is ReduceLROnPlateau (needs metric) or other
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                         scheduler.step(epoch_acc)
                    else:
                         scheduler.step() # For schedulers like StepLR, CosineAnnealingLR

                # Check for improvement and early stopping
                if epoch_acc > best_val_acc:
                    print(f"Validation accuracy improved ({best_val_acc:.4f} --> {epoch_acc:.4f}). Saving model...")
                    best_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), BEST_MODEL_PATH) # Save best model weights
                    epochs_no_improve = 0 # Reset counter
                else:
                    epochs_no_improve += 1
                    print(f"Validation accuracy did not improve for {epochs_no_improve} epoch(s).")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.6f}")

        # Early stopping check
        if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs due to no improvement in validation accuracy for {early_stopping_patience} epochs.")
            break # Exit training loop

    time_elapsed = time.time() - start_time
    print(f'\n{phase_name} Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy ({phase_name}): {best_val_acc:4f}')

    # Load best model weights back before returning
    model.load_state_dict(best_model_wts)
    return model, best_val_acc, history

# --- Evaluation Function (Unchanged - used for final test set) ---
@torch.no_grad() # Decorator for efficiency
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(test_loader, desc="Testing", leave=True) # Keep progress bar after completion
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)

        running_loss += loss.item() * inputs.size(0)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(loss=(running_loss/total if total > 0 else 0.0),
                                 acc=(correct/total if total > 0 else 0.0))

    test_loss = running_loss / total if total > 0 else 0.0
    test_acc = correct / total if total > 0 else 0.0
    print(f"\nTest Set Results:")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    return test_acc

# --- Plotting Function ---
def plot_history(history, phase1_epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    if phase1_epochs > 0 and phase1_epochs < len(epochs):
        ax1.axvline(x=phase1_epochs, color='grey', linestyle='--', label='Fine-tuning Start')
    ax1.legend()
    ax1.grid(True)

    # Plot Accuracy
    ax2.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    if phase1_epochs > 0 and phase1_epochs < len(epochs):
        ax2.axvline(x=phase1_epochs, color='grey', linestyle='--', label='Fine-tuning Start')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    plt.pause(1) # Ensure plot displays

# --- Main Execution ---
if __name__ == "__main__":
    # --- 1. Load Dataset ---
    print("Loading base Hugging Face dataset...")
    # Set cache dir if needed: cache_dir="/path/to/cache"
    hf_base_dataset = load_dataset(DATASET_NAME, split='train', trust_remote_code=True)
    num_classes = hf_base_dataset.features['label'].num_classes
    class_names = hf_base_dataset.features['label'].names # Get class names if needed later
    print(f"Dataset loaded. Num classes: {num_classes}, Total samples: {len(hf_base_dataset)}")

    # --- 2. Split Dataset (Train/Val/Test) ---
    total_size = len(hf_base_dataset)
    if total_size == 0: raise ValueError("Loaded dataset is empty!")

    test_split_size = int(0.15 * total_size) # 15% for test
    val_split_size = int(0.15 * total_size)  # 15% for validation
    train_split_size = total_size - test_split_size - val_split_size # Remaining for train (~70%)

    print(f"Splitting: Train={train_split_size}, Val={val_split_size}, Test={test_split_size}")

    # Use a fixed generator for reproducible splits
    generator = torch.Generator().manual_seed(SEED)
    train_indices, val_indices, test_indices = random_split(
        range(total_size),
        [train_split_size, val_split_size, test_split_size],
        generator=generator
    )

    # Create Subset wrappers (these are lightweight)
    train_subset_wrapper = Subset(hf_base_dataset, train_indices.indices)
    val_subset_wrapper = Subset(hf_base_dataset, val_indices.indices)
    test_subset_wrapper = Subset(hf_base_dataset, test_indices.indices)

    # --- 3. Create Datasets and DataLoaders ---
    train_dataset = TrafficSignDataset(train_subset_wrapper, transform=train_transform)
    val_dataset = TrafficSignDataset(val_subset_wrapper, transform=val_test_transform)
    test_dataset = TrafficSignDataset(test_subset_wrapper, transform=val_test_transform)

    print(f"Dataset instances created: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True if device.type == 'cuda' else False),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True if device.type == 'cuda' else False),
        'test': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True if device.type == 'cuda' else False)
    }

    # Visualize augmentations on training data
    visualize_augmentations(train_dataset)

    # --- 4. Build Model ---
    model = build_model(MODEL_NAME, num_classes, pretrained=True)
    model = model.to(device)

    # --- 5. Define Loss ---
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # --- PHASE 1: Train Classifier Head ---
    print("\n--- Setting up Phase 1: Training Classifier ---")
    # Freeze all layers except the final classifier
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the parameters of the final layer (fc or classifier)
    if hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
        params_to_optimize_p1 = model.fc.parameters()
        print("Unfrozen parameters in 'model.fc'")
    elif hasattr(model, 'classifier'):
         # Assuming the replacement was the last layer or a simple linear layer
         if isinstance(model.classifier, nn.Sequential):
             final_layer = model.classifier[-1]
         else:
             final_layer = model.classifier
         for param in final_layer.parameters():
             param.requires_grad = True
         params_to_optimize_p1 = final_layer.parameters()
         print(f"Unfrozen parameters in final layer of 'model.classifier'")
    else:
        raise AttributeError("Could not find 'fc' or 'classifier' to unfreeze.")


    optimizer_p1 = optim.AdamW(params_to_optimize_p1, lr=LR_CLASSIFIER, weight_decay=WEIGHT_DECAY_CLASSIFIER)
    scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p1, mode='max', factor=0.2, patience=SCHEDULER_PATIENCE, verbose=True)

    # Train only the classifier
    model, best_val_acc_p1, history_p1 = train_model(
        model, criterion, optimizer_p1, scheduler_p1, dataloaders, device,
        num_epochs=NUM_EPOCHS_CLASSIFIER, phase_name="Classifier Training",
        early_stopping_patience=None # No early stopping for this short phase
    )
    print(f"Phase 1 Best Validation Acc: {best_val_acc_p1:.4f}")
    # Note: The best weights from Phase 1 are already loaded into 'model'

    # --- PHASE 2: Fine-tuning ---
    print("\n--- Setting up Phase 2: Fine-tuning ---")
    # Unfreeze all layers (or choose specific layers to unfreeze)
    print("Unfreezing all model parameters for fine-tuning...")
    for param in model.parameters():
        param.requires_grad = True

    # Use a lower learning rate for all parameters
    # Filter is technically not needed here as we set all to requires_grad=True, but good practice
    params_to_optimize_p2 = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_p2 = optim.AdamW(params_to_optimize_p2, lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY_FINETUNE)
    # Reset scheduler or create a new one
    scheduler_p2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p2, mode='max', factor=0.2, patience=SCHEDULER_PATIENCE, verbose=True)
    # scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=NUM_EPOCHS_FINETUNE, eta_min=LR_FINETUNE/100) # Alternative

    # Fine-tune the model with early stopping
    model, best_val_acc_p2, history_p2 = train_model(
        model, criterion, optimizer_p2, scheduler_p2, dataloaders, device,
        num_epochs=NUM_EPOCHS_FINETUNE, phase_name="Fine-tuning",
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )
    print(f"Phase 2 Best Validation Acc: {best_val_acc_p2:.4f}")
    # Note: The best weights from Phase 2 (potentially overall best) are loaded into 'model'

    # --- 6. Final Evaluation on Test Set ---
    print("\n--- Evaluating final best model on Test Set ---")
    # Ensure the *absolute best* model (based on validation acc across both phases) is loaded
    # The train_model function already loads the best weights from the respective phase.
    # If phase 2 improved, model has phase 2's best. If not, it *should* have phase 1's best
    # (but the current logic loads phase 2's last best). A safer way is to explicitly load BEST_MODEL_PATH:
    print(f"Loading best weights from {BEST_MODEL_PATH} for final test evaluation.")
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Warning: Could not load best weights from {BEST_MODEL_PATH}. Evaluating with current model state. Error: {e}")

    test_accuracy = evaluate(model, dataloaders['test'], criterion, device)
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")

    # --- 7. Save Final Model (Optional) ---
    # Save the model that performed best on the validation set during the entire process
    # (which should be the one loaded before evaluation)
    # torch.save(model.state_dict(), FINAL_MODEL_PATH)
    # print(f"Final best model weights saved to {FINAL_MODEL_PATH}")

    # --- 8. Plot History ---
    # Combine history dictionaries
    full_history = {
        'train_loss': history_p1['train_loss'] + history_p2['train_loss'],
        'train_acc': history_p1['train_acc'] + history_p2['train_acc'],
        'val_loss': history_p1['val_loss'] + history_p2['val_loss'],
        'val_acc': history_p1['val_acc'] + history_p2['val_acc'],
    }
    plot_history(full_history, phase1_epochs=NUM_EPOCHS_CLASSIFIER)

    print("\nTraining and evaluation finished.")
    plt.close('all') # Close plots