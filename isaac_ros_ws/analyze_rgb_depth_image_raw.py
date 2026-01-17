import cv2
import numpy as np
import argparse
import os

# --- Constants from Camera Info ---
# Extracted from camera_info.txt [cite: 1, 2, 3]
FX = 448.15853517
FY = 448.15853517
CX = 640.0
CY = 360.0

def process_robotics_scene(rgb_path, depth_path):
    """
    Detects white parts, calculates 3D position and real-world dimensions.
    """
    # 1. Load Images
    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print(f"Error: One or both files not found.\nRGB: {rgb_path}\nDepth: {depth_path}")
        return

    rgb = cv2.imread(rgb_path)
    # depth image represents distance in meters
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if rgb is None or depth_map is None:
        print("Error: Could not decode image files.")
        return

    # 2. Part Detection
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # Using threshold to isolate white gears/components
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_objects = []
    viz_img = rgb.copy()

    # 3. Analyze each part
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            continue

        x_px, y_px, w_px, h_px = cv2.boundingRect(cnt)
        
        # Calculate Geometric Center (Centroid)
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        u, v = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        
        # Get Depth (Z) in meters
        z_m = float(depth_map[v, u])
        
        # Fallback for holes in gears: use median of valid depth in the bounding box
        if z_m <= 0:
            roi = depth_map[y_px:y_px+h_px, x_px:x_px+w_px]
            valid_z = roi[roi > 0]
            z_m = np.median(valid_z) if valid_z.size > 0 else 0
        
        if z_m > 0:
            # 4. 3D Coordinates (Camera Local Frame)
            cam_x = (u - CX) * z_m / FX
            cam_y = (v - CY) * z_m / FY
            cam_z = z_m
            
            # 5. Dimensions in Centimeters
            width_cm = (w_px * z_m / FX) * 100
            height_cm = (h_px * z_m / FY) * 100
            area_cm2 = width_cm * height_cm

            obj_data = {
                'width': round(width_cm, 2),
                'height': round(height_cm, 2),
                'area': round(area_cm2, 2),
                'x': round(cam_x, 4),
                'y': round(cam_y, 4),
                'z': round(cam_z, 4)
            }
            detected_objects.append(obj_data)

            # 6. Visualization
            cv2.drawContours(viz_img, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(viz_img, (u, v), 3, (0, 0, 255), -1)
            label = f"{width_cm:.1f}x{height_cm:.1f}cm"
            cv2.putText(viz_img, label, (x_px, y_px - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 7. CLI Output
    header = f"{'ID':<4} | {'Width(cm)':<10} | {'Height(cm)':<11} | {'Area(cm^2)':<11} | {'X(m)':<8} | {'Y(m)':<8} | {'Z(m)':<8}"
    print("\nDetection Summary:")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for i, obj in enumerate(detected_objects):
        print(f"{i:<4} | {obj['width']:<10} | {obj['height']:<11} | {obj['area']:<11} | {obj['x']:<8} | {obj['y']:<8} | {obj['z']:<8}")

    # 8. Result Display
    cv2.imshow("Object Measurement", viz_img)
    print("\nPress any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return detected_objects

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze robotics parts from RGB-D data.")
    parser.add_argument("--rgb", type=str, required=True, help="Path to the RGB image (e.g., 0_rgb.jpg)")
    parser.add_argument("--depth", type=str, required=True, help="Path to the Depth image (e.g., 0_depth.png)")
    
    args = parser.parse_args()
    process_robotics_scene(args.rgb, args.depth)