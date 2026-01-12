# create_test.py
import cv2
import numpy as np
import os

# Create directories
os.makedirs("input", exist_ok=True)

# Create a simple foreground with alpha
fg = np.zeros((400, 300, 4), dtype=np.uint8)
# Draw a person silhouette
cv2.circle(fg, (150, 100), 30, (0, 0, 255, 255), -1)  # Red head
cv2.rectangle(fg, (135, 100), (165, 250), (0, 0, 255, 255), -1)  # Red body
cv2.rectangle(fg, (115, 130), (135, 180), (0, 0, 255, 255), -1)  # Left arm
cv2.rectangle(fg, (165, 130), (185, 180), (0, 0, 255, 255), -1)  # Right arm

# Create background
bg = np.ones((400, 300, 3), dtype=np.uint8) * 200
# Add some texture
cv2.rectangle(bg, (50, 150), (250, 300), (150, 200, 150), -1)  # Green floor

# Save
cv2.imwrite("input/foreground.png", fg)
cv2.imwrite("input/background.png", bg)
print("✅ Test images created in 'input/' folder")