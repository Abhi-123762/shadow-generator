import cv2
import numpy as np

def project_shadow(mask, angle=45, elevation=45):
    """Project shadow based on light angle and elevation"""
    height, width = mask.shape
    
    # Convert angles to radians
    angle_rad = np.deg2rad(angle)
    
    # Calculate offset based on elevation (higher elevation = shorter shadow)
    distance = int(50 * (90 - elevation) / 90)  # Max 50 pixels at 0 elevation
    
    # Calculate x and y offsets
    offset_x = int(distance * np.cos(angle_rad))
    offset_y = int(distance * np.sin(angle_rad))
    
    # Create transformation matrix
    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    
    # Apply affine transformation
    shadow = cv2.warpAffine(mask, M, (width, height))
    
    # Return shadow and alpha (use mask as alpha for blending)
    return shadow, mask.copy()