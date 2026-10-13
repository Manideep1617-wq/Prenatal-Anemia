"""
Grad-CAM (Gradient-weighted Class Activation Mapping) explainability module
for binary prenatal anemia detection using ResNet-18 ImageModel.
"""

import os
import sys
import glob
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Ensure project root is accessible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import ImageModel

# Default paths and configuration
DEFAULT_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "results", "checkpoints", "image_model_best.pt")
CLASS_NAMES = ["Non-Anemic", "Anemic"]

# Global cache for loaded model
_CACHED_MODEL = None
_CACHED_MODEL_PATH = None


def get_device() -> torch.device:
    """Return available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image_transform() -> transforms.Compose:
    """Return standard image preprocessing transform used during training/inference."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_image_model(checkpoint_path: Optional[str] = None, device: Optional[torch.device] = None) -> ImageModel:
    """
    Load the ImageModel (ResNet-18 based) from checkpoint in eval mode.

    Args:
        checkpoint_path: Path to model checkpoint .pt file.
                         Defaults to 'results/checkpoints/image_model_best.pt'.
        device: Target torch.device. If None, auto-detected (CUDA/CPU).

    Returns:
        Loaded ImageModel in eval mode on target device.
    """
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT_PATH

    # Resolve relative paths against PROJECT_ROOT if not directly found
    if not os.path.isabs(checkpoint_path) and not os.path.exists(checkpoint_path):
        resolved = os.path.join(PROJECT_ROOT, checkpoint_path)
        if os.path.exists(resolved):
            checkpoint_path = resolved

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    if device is None:
        device = get_device()

    model = ImageModel(num_classes=2, pretrained=False)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        state_dict = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def compute_gradcam(
    model: ImageModel,
    pil_image: Union[Image.Image, str],
    target_class: Optional[int] = None,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, Image.Image]:
    """
    Compute Grad-CAM heatmap and overlay for a given PIL image.

    Args:
        model: Trained ImageModel instance.
        pil_image: PIL Image instance (or file path).
        target_class: Target class index (0 for Non-Anemic, 1 for Anemic).
                      If None, uses model's predicted class.
        alpha: Overlay blending weight for heatmap (default 0.4).

    Returns:
        Tuple of (heatmap_np_array, overlay_pil_image):
            - heatmap_np_array: 2D np.ndarray float [0.0, 1.0] resized to original image dimensions.
            - overlay_pil_image: PIL Image with jet colormap overlay blended with original image.
    """
    if isinstance(pil_image, (str, os.PathLike)):
        pil_image = Image.open(pil_image)

    orig_image = pil_image.convert("RGB")
    orig_w, orig_h = orig_image.size

    # Prepare input tensor
    transform = get_image_transform()
    device = next(model.parameters()).device
    input_tensor = transform(orig_image).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    # Hooks to capture activations and gradients
    activations = []
    gradients = []

    def forward_hook(module, input_val, output_val):
        activations.append(output_val)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.backbone.layer4
    hook_handles = [target_layer.register_forward_hook(forward_hook)]

    if hasattr(target_layer, "register_full_backward_hook"):
        hook_handles.append(target_layer.register_full_backward_hook(backward_hook))
    else:
        hook_handles.append(target_layer.register_backward_hook(backward_hook))

    try:
        model.eval()
        with torch.enable_grad():
            model.zero_grad()
            output = model(input_tensor)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            score = output[0, target_class]
            model.zero_grad()
            score.backward(retain_graph=False)

        if not activations or not gradients:
            raise RuntimeError("Failed to capture activations or gradients during Grad-CAM.")

        act = activations[0]   # (1, 512, 7, 7)
        grad = gradients[0]   # (1, 512, 7, 7)

        # Global average pooling of gradients as importance weights alpha_k
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # (1, 512, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * act, dim=1, keepdim=True)    # (1, 1, 7, 7)

        # ReLU to keep features with positive influence on the target class
        cam = torch.relu(cam)

        cam_np = cam.squeeze().detach().cpu().numpy()          # (7, 7)

        # Normalize heatmap to [0.0, 1.0]
        cam_min = float(cam_np.min())
        cam_max = float(cam_np.max())
        if cam_max - cam_min > 1e-8:
            heatmap_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            heatmap_norm = np.zeros_like(cam_np, dtype=np.float32)

        # Resize heatmap to original image dimensions (width, height)
        heatmap_resized = cv2.resize(heatmap_norm.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

        # Apply colormap (jet) overlay with alpha=0.4
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        orig_np = np.array(orig_image, dtype=np.float32)
        heatmap_rgb_f = heatmap_rgb.astype(np.float32)

        overlay_np = np.clip(alpha * heatmap_rgb_f + (1.0 - alpha) * orig_np, 0, 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_np)

        return heatmap_resized, overlay_pil

    finally:
        for handle in hook_handles:
            handle.remove()


def explain_image(
    pil_image: Union[Image.Image, str],
    checkpoint_path: Optional[str] = None,
    model: Optional[ImageModel] = None,
) -> Dict[str, Any]:
    """
    Generate complete prediction and Grad-CAM explanation for an image.

    Args:
        pil_image: PIL Image instance (or file path).
        checkpoint_path: Optional path to checkpoint. Defaults to best image model.
        model: Optional pre-loaded ImageModel instance to reuse.

    Returns:
        dict with keys:
            - 'prediction': 'Anemic' or 'Non-Anemic'
            - 'confidence': float 0-1
            - 'heatmap': np.ndarray 2D float array [0, 1]
            - 'overlay': PIL.Image with heatmap overlaid
            - 'probabilities': {'Anemic': float, 'Non-Anemic': float}
    """
    global _CACHED_MODEL, _CACHED_MODEL_PATH

    if isinstance(pil_image, (str, os.PathLike)):
        pil_image = Image.open(pil_image)

    orig_image = pil_image.convert("RGB")

    # Load or reuse model
    if model is None:
        resolved_ckpt = checkpoint_path or DEFAULT_CHECKPOINT_PATH
        if _CACHED_MODEL is None or _CACHED_MODEL_PATH != resolved_ckpt:
            _CACHED_MODEL = load_image_model(checkpoint_path=resolved_ckpt)
            _CACHED_MODEL_PATH = resolved_ckpt
        model = _CACHED_MODEL

    device = next(model.parameters()).device
    transform = get_image_transform()
    input_tensor = transform(orig_image).unsqueeze(0).to(device)

    # Compute prediction probabilities
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()

    prob_non_anemic = float(probs[0])
    prob_anemic = float(probs[1])

    pred_idx = int(np.argmax(probs))
    prediction = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    probabilities = {
        "Anemic": prob_anemic,
        "Non-Anemic": prob_non_anemic,
    }

    # Compute Grad-CAM for the predicted class
    heatmap, overlay = compute_gradcam(model, orig_image, target_class=pred_idx, alpha=0.4)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "heatmap": heatmap,
        "overlay": overlay,
        "probabilities": probabilities,
    }


def main():
    print("=" * 60)
    print("Grad-CAM Explainability Module Test")
    print("=" * 60)

    # Search for sample images
    sample_pattern = os.path.join(PROJECT_ROOT, "data", "Dataset_sample", "**", "*.png")
    sample_images = glob.glob(sample_pattern, recursive=True)

    if not sample_images:
        sample_pattern = os.path.join(PROJECT_ROOT, "data", "Dataset_sample", "**", "*.[jJ][pP]*[gG]")
        sample_images = glob.glob(sample_pattern, recursive=True)

    if sample_images:
        test_img_path = sample_images[0]
        print(f"Testing with sample image: {test_img_path}")
        pil_img = Image.open(test_img_path)
    else:
        print("No sample images found in data/Dataset_sample/. Creating synthetic test image.")
        pil_img = Image.new("RGB", (256, 256), color=(200, 150, 150))

    print(f"Original image size: {pil_img.size}")

    # Test load_image_model
    print("\n1. Loading model...")
    model = load_image_model()
    print("   Model loaded successfully into eval mode.")

    # Test compute_gradcam
    print("\n2. Computing Grad-CAM directly...")
    heatmap, overlay = compute_gradcam(model, pil_img)
    print(f"   Heatmap shape: {heatmap.shape}, min={heatmap.min():.4f}, max={heatmap.max():.4f}")
    print(f"   Overlay size: {overlay.size}, mode={overlay.mode}")

    # Test explain_image
    print("\n3. Testing explain_image()...")
    result = explain_image(pil_img)
    print(f"   Prediction:    {result['prediction']}")
    print(f"   Confidence:    {result['confidence']:.4f}")
    print(f"   Probabilities: Anemic={result['probabilities']['Anemic']:.4f}, Non-Anemic={result['probabilities']['Non-Anemic']:.4f}")
    print(f"   Heatmap shape: {result['heatmap'].shape}")
    print(f"   Overlay size:  {result['overlay'].size}")

    # Save output visualization
    output_dir = os.path.join(PROJECT_ROOT, "results", "explainability")
    os.makedirs(output_dir, exist_ok=True)
    out_overlay_path = os.path.join(output_dir, "gradcam_test_overlay.png")
    result["overlay"].save(out_overlay_path)
    print(f"\nSaved test overlay to: {out_overlay_path}")
    print("\nAll Grad-CAM tests passed successfully!")


if __name__ == "__main__":
    main()
