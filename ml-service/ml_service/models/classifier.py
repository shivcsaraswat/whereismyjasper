"""
Jasper Classifier Model

This module handles:
- Creating a ResNet50 model modified for binary classification
- Freezing/unfreezing layers for transfer learning
- Saving and loading trained models

Architecture:
    ResNet50 (pretrained on ImageNet)
    └── Replace final FC layer: 2048 → 2 (Jasper / Not Jasper)
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from typing import Optional


def create_model(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """
    Create a ResNet50 model configured for transfer learning.

    Args:
        num_classes: Number of output classes (2 for Jasper/Not Jasper)
        freeze_backbone: If True, freeze all layers except the final FC layer
                        This is "feature extraction" mode - fastest to train

    Returns:
        Modified ResNet50 model ready for training

    How it works:
        1. Load ResNet50 with pretrained ImageNet weights
        2. Optionally freeze all convolutional layers (no gradient updates)
        3. Replace the final fully connected layer:
           - Original: 2048 → 1000 (ImageNet classes)
           - Modified: 2048 → 2 (Jasper / Not Jasper)
    """
    # Load pretrained ResNet50
    # weights="IMAGENET1K_V2" is the latest, best-performing weights
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze backbone layers if requested (feature extraction mode)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        print("Backbone frozen: Only training the final FC layer")

    # Replace the final fully connected layer
    # model.fc is the last layer: Linear(2048, 1000)
    # We replace it with: Linear(2048, num_classes)
    num_features = model.fc.in_features  # This is 2048 for ResNet50
    model.fc = nn.Linear(num_features, num_classes)

    # The new FC layer has requires_grad=True by default
    # So even if backbone is frozen, this layer will train

    print(f"Model created: ResNet50 with {num_classes} output classes")
    print(f"Final layer: Linear({num_features}, {num_classes})")

    return model


def unfreeze_layer4(model: nn.Module, lr_backbone: float = 1e-5) -> list:
    """
    Unfreeze Layer4 (the last convolutional block) for fine-tuning.

    Use this AFTER initial training if you want the model to learn
    more Jasper-specific features.

    Args:
        model: The ResNet50 model
        lr_backbone: Learning rate for the unfrozen layers (should be small!)

    Returns:
        List of parameter groups for the optimizer with different learning rates

    Why different learning rates?
        - FC layer (new): Higher LR (1e-3) - needs to learn from scratch
        - Layer4 (pretrained): Lower LR (1e-5) - just fine-tuning
        - This prevents destroying the pretrained features
    """
    # Unfreeze layer4 parameters
    for param in model.layer4.parameters():
        param.requires_grad = True

    print("Layer4 unfrozen for fine-tuning")

    # Return parameter groups with different learning rates
    # This is used with the optimizer like:
    # optimizer = Adam(param_groups)
    param_groups = [
        {"params": model.fc.parameters(), "lr": 1e-3},  # FC layer: normal LR
        {"params": model.layer4.parameters(), "lr": lr_backbone},  # Layer4: tiny LR
    ]

    return param_groups


def save_model(model: nn.Module, filepath: Path, metadata: Optional[dict] = None) -> None:
    """
    Save the trained model to disk.

    Args:
        model: Trained PyTorch model
        filepath: Where to save (e.g., "models/jasper_classifier.pth")
        metadata: Optional dict with training info (accuracy, epochs, etc.)

    What gets saved:
        - state_dict: The learned weights (not the architecture)
        - metadata: Training information for reproducibility
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }

    torch.save(save_dict, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath: Path, num_classes: int = 2) -> tuple:
    """
    Load a trained model from disk.

    Args:
        filepath: Path to the saved model file
        num_classes: Number of classes (must match what was trained)

    Returns:
        Tuple of (model, metadata)

    Note:
        We create a fresh model architecture, then load the weights.
        This ensures compatibility even if the code changes slightly.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")

    # Create model architecture (don't freeze - we're loading trained weights)
    model = create_model(num_classes=num_classes, freeze_backbone=False)

    # Load the saved weights
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])

    # Set to evaluation mode (disables dropout, batch norm in eval mode)
    model.eval()

    print(f"Model loaded from: {filepath}")

    return model, checkpoint.get("metadata", {})


def get_device() -> torch.device:
    """
    Get the best available device for training/inference.

    Priority: CUDA GPU > Apple MPS > CPU

    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU (GPU not available)")

    return device
