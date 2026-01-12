
import cv2
import numpy as np
def apply_falloff(shadow, alpha):
    blur = cv2.GaussianBlur(alpha, (0, 0), 15)
    return shadow * blur
