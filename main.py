import cv2
import argparse
import numpy as np
import os
import sys
from shadow.mask import extract_mask
from shadow.projection import project_shadow
from shadow.falloff import apply_falloff

parser = argparse.ArgumentParser()
parser.add_argument("--foreground", required=True)
parser.add_argument("--background", required=True)
parser.add_argument("--angle", type=float, default=45)
parser.add_argument("--elevation", type=float, default=45)
args = parser.parse_args()

# FIRST: Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# DEBUG: Print current directory and check if files exist
print(f"🔍 Current working directory: {os.getcwd()}")
print(f"📁 Looking for foreground: {args.foreground}")
print(f"📁 Looking for background: {args.background}")
print(f"📁 Full foreground path: {os.path.abspath(args.foreground)}")
print(f"📁 Full background path: {os.path.abspath(args.background)}")

# Check if files exist before trying to load
if not os.path.exists(args.foreground):
    print(f"❌ ERROR: Foreground file not found: {args.foreground}")
    print("💡 Try using an absolute path or check the file location")
    print("💡 Example: --foreground C:/Users/SHALINI JADA/Downloads/foreground.png")
    sys.exit(1)

if not os.path.exists(args.background):
    print(f"❌ ERROR: Background file not found: {args.background}")
    sys.exit(1)

# Load images
fg = cv2.imread(args.foreground, cv2.IMREAD_UNCHANGED)
bg = cv2.imread(args.background, cv2.IMREAD_UNCHANGED)  # FIXED: Added IMREAD_UNCHANGED

# Check if images loaded successfully
if fg is None:
    print(f"❌ ERROR: Failed to load foreground image (check if it's a valid image file)")
    sys.exit(1)
if bg is None:
    print(f"❌ ERROR: Failed to load background image (check if it's a valid image file)")
    sys.exit(1)

print(f"✅ Images loaded successfully!")
print(f"   Foreground shape: {fg.shape}")
print(f"   Background shape: {bg.shape}")

# Check if foreground has alpha channel
if fg.shape[2] != 4:
    print("⚠️ WARNING: Foreground image should have 4 channels (RGBA)")
    print("   Adding dummy alpha channel...")
    # Add alpha channel if missing
    fg = cv2.cvtColor(fg, cv2.COLOR_BGR2BGRA)
    fg[:, :, 3] = 255  # Set alpha to fully opaque

# Check if background has 3 channels
if len(bg.shape) == 2:  # Grayscale
    bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
elif bg.shape[2] == 4:  # RGBA
    bg = cv2.cvtColor(bg, cv2.COLOR_BGRA2BGR)

# Resize background to match foreground if needed
if bg.shape[:2] != fg.shape[:2]:
    print(f"⚠️ Resizing background to match foreground dimensions")
    bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))

# Extract mask
mask = extract_mask(fg)

# Generate shadow
shadow, alpha = project_shadow(mask, args.angle, args.elevation)
shadow = apply_falloff(shadow, alpha)

# Save debug images
cv2.imwrite("output/shadow_only.png", (shadow * 255).astype("uint8"))
cv2.imwrite("output/mask_debug.png", (mask * 255).astype("uint8"))

# Extract foreground components
fg_rgb = fg[:, :, :3]
fg_alpha = fg[:, :, 3:] / 255.0

# Create composite
comp = bg.copy().astype("float32")
comp = comp * (1 - alpha[..., None]) + shadow[..., None] * 255
comp = comp * (1 - fg_alpha) + fg_rgb * fg_alpha
comp = np.clip(comp, 0, 255).astype("uint8")

# Save final composite
cv2.imwrite("output/composite.png", comp)
print(f"✅ Processing complete!")
print(f"📁 Results saved in 'output/' folder:")
print(f"   - composite.png")
print(f"   - shadow_only.png")
print(f"   - mask_debug.png")