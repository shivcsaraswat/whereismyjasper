"""
Transfer Learning Training Script for Jasper Classifier

Entry point: jasper-train

This script:
1. Loads images from data/jasper and data/not_jasper folders
2. Applies data augmentation to expand the small dataset
3. Creates a ResNet50 model with frozen backbone
4. Trains only the final classification layer
5. Saves the best model based on validation accuracy

Data Augmentation Strategy:
    With only ~7 images, we use aggressive augmentation:
    - Random horizontal flip (Jasper facing left or right)
    - Random rotation (up to 15 degrees)
    - Color jitter (brightness, contrast variations)
    - Random resized crop (simulates different distances)

    This can turn 7 images into hundreds of unique variations!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
import argparse
from datetime import datetime

# Import from our classifier module
from ml_service.models.classifier import create_model, save_model, get_device


# =============================================================================
# DATA TRANSFORMS (Augmentation)
# =============================================================================

def get_train_transforms() -> transforms.Compose:
    """
    Get training transforms with data augmentation.

    These transforms artificially expand our small dataset by creating
    variations of each image during training.

    Returns:
        Compose object with chained transforms
    """
    return transforms.Compose([
        # Randomly resize and crop to 224x224
        # scale=(0.8, 1.0) means use 80-100% of the image
        # This simulates Jasper at different distances
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),

        # 50% chance to flip horizontally
        # Jasper looks the same facing left or right
        transforms.RandomHorizontalFlip(p=0.5),

        # Random rotation up to 15 degrees
        # Handles slightly tilted camera angles
        transforms.RandomRotation(degrees=15),

        # Random color adjustments
        # Handles different lighting conditions
        transforms.ColorJitter(
            brightness=0.2,  # ±20% brightness
            contrast=0.2,    # ±20% contrast
            saturation=0.2,  # ±20% saturation
            hue=0.1,         # ±10% hue shift
        ),

        # Convert PIL Image to PyTorch tensor
        # Changes shape from (H, W, C) to (C, H, W)
        # Changes values from 0-255 to 0.0-1.0
        transforms.ToTensor(),

        # Normalize using ImageNet statistics
        # ResNet was trained with these values, so we must use them
        # mean: average pixel value per channel in ImageNet
        # std: standard deviation per channel in ImageNet
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Get validation/inference transforms (NO augmentation).

    For validation and inference, we want deterministic results,
    so we don't apply random augmentations.

    Returns:
        Compose object with chained transforms
    """
    return transforms.Compose([
        # Resize to 256, then center crop to 224
        # This is the standard ResNet inference preprocessing
        transforms.Resize(256),
        transforms.CenterCrop(224),

        # Same tensor conversion and normalization as training
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# =============================================================================
# DATA LOADING
# =============================================================================

def create_dataloaders(
    data_dir: Path,
    batch_size: int = 8,
    val_split: float = 0.2,
    num_workers: int = 0,
) -> tuple:
    """
    Create training and validation DataLoaders.

    Expects folder structure:
        data_dir/
        ├── jasper/        (class 0 or 1 depending on sort order)
        │   ├── img1.jpg
        │   └── img2.jpg
        └── not_jasper/    (class 0 or 1 depending on sort order)
            ├── img1.jpg
            └── img2.jpg

    Args:
        data_dir: Path to data folder containing class subfolders
        batch_size: Number of images per batch
        val_split: Fraction of data to use for validation (0.2 = 20%)
        num_workers: Number of parallel data loading processes

    Returns:
        Tuple of (train_loader, val_loader, class_names)
    """
    data_dir = Path(data_dir)

    # ImageFolder automatically:
    # - Finds all images in subfolders
    # - Assigns class labels based on folder names (alphabetical order)
    # - Applies the transform to each image

    # Load full dataset with training transforms initially
    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=get_train_transforms()
    )

    # Get class names (folder names)
    class_names = full_dataset.classes
    print(f"Found classes: {class_names}")
    print(f"Total images: {len(full_dataset)}")

    # Split into training and validation sets
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )

    # Override validation transforms (no augmentation)
    # Note: This is a bit hacky but necessary with random_split
    val_dataset.dataset.transform = get_val_transforms()

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create DataLoaders
    # DataLoader handles batching, shuffling, and parallel loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data each epoch
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, class_names


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple:
    """
    Train for one epoch.

    Args:
        model: The neural network
        train_loader: DataLoader for training data
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer (Adam)
        device: Device to train on (cuda/mps/cpu)

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()  # Set model to training mode

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to device (GPU if available)
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients from previous batch
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass (compute gradients)
        loss.backward()

        # Update weights
        optimizer.step()

        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Evaluate model on validation set.

    Args:
        model: The neural network
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient computation for validation (saves memory)
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total

    return val_loss, val_acc


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    val_split: float = 0.2,
) -> None:
    """
    Full training pipeline.

    Args:
        data_dir: Path to data folder with jasper/ and not_jasper/ subfolders
        output_dir: Where to save the trained model
        epochs: Number of training epochs
        batch_size: Images per batch
        learning_rate: Learning rate for optimizer
        val_split: Fraction for validation
    """
    print("=" * 60)
    print("JASPER CLASSIFIER TRAINING")
    print("=" * 60)

    # Setup
    device = get_device()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, class_names = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=val_split,
    )

    # Create model
    print("\nCreating model...")
    model = create_model(num_classes=len(class_names), freeze_backbone=True)
    model = model.to(device)

    # Loss function and optimizer
    # CrossEntropyLoss: standard for multi-class classification
    # Adam: adaptive learning rate optimizer (good default)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    # Learning rate scheduler: reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Training loop
    print("\nStarting training...")
    print("-" * 60)

    best_val_acc = 0.0
    best_model_path = output_dir / "jasper_classifier_best.pth"

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Print progress
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            metadata = {
                "epoch": epoch,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "class_names": class_names,
                "timestamp": datetime.now().isoformat(),
            }
            save_model(model, best_model_path, metadata)
            print(f"         ↳ New best model saved! (Val Acc: {val_acc:.1f}%)")

    # Final summary
    print("-" * 60)
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {best_model_path}")
    print(f"\nClass mapping: {dict(enumerate(class_names))}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """
    CLI entry point for jasper-train command.

    Usage:
        jasper-train                    # Use defaults
        jasper-train --epochs 30        # Custom epochs
        jasper-train --data ./my_data   # Custom data directory
    """
    parser = argparse.ArgumentParser(
        description="Train Jasper classifier using transfer learning"
    )

    # Get the ml-service directory (parent of ml_service package)
    default_data_dir = Path(__file__).parent.parent.parent / "data"
    default_output_dir = Path(__file__).parent.parent.parent / "models"

    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=default_data_dir,
        help="Path to data directory with jasper/ and not_jasper/ subfolders"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=default_output_dir,
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)"
    )

    args = parser.parse_args()

    # Run training
    train(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
