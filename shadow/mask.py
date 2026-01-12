import cv2
import numpy as np

def extract_mask(fg):
    """Extract mask from foreground image"""
    if fg is None:
        raise ValueError("Foreground image is None - check if image loaded correctly")
    
    # Debug info
    print(f"🔍 Mask extraction - Input shape: {fg.shape}")
    
    # If image has alpha channel (RGBA)
    if fg.shape[2] == 4:
        mask = fg[:, :, 3].astype(np.float32) / 255.0
    else:
        print("⚠️ No alpha channel found, creating mask from image content")
        # Create mask from image (assuming non-black areas are foreground)
        gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32) / 255.0
    
    print(f"✅ Mask extracted - Shape: {mask.shape}, Range: [{mask.min():.2f}, {mask.max():.2f}]")
    return mask