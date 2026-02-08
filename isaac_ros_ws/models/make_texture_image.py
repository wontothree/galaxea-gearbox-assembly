import cv2
import numpy as np

# Create a 16x16 pixel light gray image
color = int(0.752941 * 255) # 192
texture = np.full((16, 16, 3), (color, color, color), dtype=np.uint8)

# Save as gear_texture.png
cv2.imwrite("gear_texture.png", texture)
