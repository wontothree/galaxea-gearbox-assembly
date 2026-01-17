import cv2
import numpy as np
import os
import time
from datetime import datetime

class TableSceneAnalyzer:
    def __init__(self):
        # --- Camera Intrinsics ---
        self.fx = 448.15853517
        self.fy = 448.15853517
        self.cx = 640.0
        self.cy = 360.0
        
        # --- Extrinsics & Transform Constants ---
        self.z_intersection = 0.69702865
        self.R = np.array([
            [ 1.00000000e+00, -2.13171516e-04, -1.16013877e-04],
            [-2.42695931e-04, -8.78348099e-01, -4.78021504e-01],
            [ 0.00000000e+00,  4.78021518e-01, -8.78348125e-01]
        ])
        
        self.base_offset_x = 0.585
        self.base_offset_y = -0.0015
        self.base_z_ref = 0.902

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"run_{self.run_id}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.last_save_time = 0  
        self.save_interval = 1.0 
        self.detected_objects = []

    def classify_gear(self, area):
        if area > 300: return "Ring Gear"
        elif area > 100: return "Planetary Carrier"
        elif area > 25: return "Sun Planetary Gear"
        else: return "Planetary Reducer"

    def get_base_coordinates(self, u, v, depth):
        xc = (u - self.cx) * depth / self.fx
        yc = (v - self.cy) * depth / self.fy
        zc = depth
        pc_shifted = np.array([xc, yc, zc - self.z_intersection])
        p_table = self.R.T @ pc_shifted
        
        x_tab, y_tab = p_table[0], p_table[1]
        x_base = y_tab + self.base_offset_x
        y_base = -x_tab + self.base_offset_y
        z_base = p_table[2] + self.base_z_ref
        
        # Otherwise "subtract_frame_transforms" function won't work in *fsm.py
        return float(np.float32(x_base)), float(np.float32(y_base)), float(np.float32(z_base))

    def detect_carrier_pins(self, depth_map, carrier_contour, render_img):
        """Detects pin tops using peak height detection without the red grid overlay."""
        pins = []
        x, y, w, h = cv2.boundingRect(carrier_contour)
        
        # 1. Vectorized Height Map Generation (Faster than nested loops)
        # Create grids of u,v coordinates for the ROI
        v_grid, u_grid = np.mgrid[y:y+h, x:x+w]
        roi_depth = depth_map[y:y+h, x:x+w]
        
        # Mask valid depth pixels
        mask = roi_depth > 0
        height_map = np.zeros_like(roi_depth, dtype=np.float32)

        if np.any(mask):
            # Apply transformation math to the entire grid at once
            xc = (u_grid[mask] - self.cx) * roi_depth[mask] / self.fx
            yc = (v_grid[mask] - self.cy) * roi_depth[mask] / self.fy
            zc = roi_depth[mask]
            
            # Rotation matrix application (simplified for Z-height extraction)
            # z_base = (R_inv[2,0]*xc + R_inv[2,1]*yc + R_inv[2,2]*(zc-z_int)) + base_z_ref
            pc_shifted_z = zc - self.z_intersection
            z_base = (self.R[0,2] * xc + self.R[1,2] * yc + self.R[2,2] * pc_shifted_z) + self.base_z_ref
            height_map[mask] = z_base

        # 2. Determine Surface Floor
        valid_heights = height_map[height_map > 0]
        if valid_heights.size == 0: return pins
        carrier_floor_z = np.median(valid_heights)

        # 3. Peak Detection (+10mm threshold for top part)
        peak_mask = np.zeros((h, w), dtype=np.uint8)
        peak_mask[height_map > (carrier_floor_z + 0.010)] = 255
        
        # Morphological cleanup
        kernel = np.ones((3,3), np.uint8)
        peak_mask = cv2.morphologyEx(peak_mask, cv2.MORPH_OPEN, kernel)

        # 4. Find Pin Peaks
        peak_contours, _ = cv2.findContours(peak_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for p_cnt in peak_contours:
            if 3 < cv2.contourArea(p_cnt) < 150:
                M = cv2.moments(p_cnt)
                if M["m00"] == 0: continue
                
                u_p = int(M["m10"] / M["m00"]) + x
                v_p = int(M["m01"] / M["m00"]) + y
                
                # Extract Z from height map and calculate full base coordinates
                pz = float(height_map[v_p - y, u_p - x])
                px, py, _ = self.get_base_coordinates(u_p, v_p, float(depth_map[v_p, u_p]))
                
                pin_data = {'label': "Carrier Pin", 'x': round(px, 4), 'y': round(py, 4), 'z': round(pz, 4)}
                pins.append(pin_data)

                # --- Clean Visual Rendering ---
                # Draw Blue circle at the peak (to contrast with Green gears)
                cv2.circle(render_img, (u_p, v_p), 6, (255, 0, 0), 2)
                
                # Small white label for the height above carrier
                h_mm = round((pz - carrier_floor_z) * 1000, 1)
                cv2.putText(render_img, f"TOP +{h_mm}mm", (u_p + 8, v_p), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        return pins

    def process_frame(self, rgb, depth_map):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        render_img = rgb.copy()
        self.detected_objects = []

        for cnt in contours:
            if cv2.contourArea(cnt) < 50: continue

            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            u, v = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            
            z_m = float(depth_map[v, u])
            if z_m <= 0:
                x_px, y_px, w_px, h_px = cv2.boundingRect(cnt)
                roi = depth_map[y_px:y_px+h_px, x_px:x_px+w_px]
                valid_z = roi[roi > 0]
                z_m = np.median(valid_z) if valid_z.size > 0 else 0
            
            if z_m > 0:
                x, y, z = self.get_base_coordinates(u, v, z_m)
                _, _, w_px, h_px = cv2.boundingRect(cnt)
                area_cm2 = (w_px * z_m / self.fx * 100) * (h_px * z_m / self.fy * 100)
                label = self.classify_gear(area_cm2)

                obj_data = {'label': label, 'x': round(x, 4), 'y': round(y, 4), 'z': round(z, 4)}
                self.detected_objects.append(obj_data)

                # Gear Visualization (Green)
                cv2.drawContours(render_img, [cnt], -1, (0, 255, 0), 2)
                cv2.putText(render_img, label, (u - 40, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if label == "Planetary Carrier":
                    # Call new depth-based detector
                    pin_objects = self.detect_carrier_pins(depth_map, cnt, render_img)
                    self.detected_objects.extend(pin_objects)

        # Periodic Export
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            save_path = os.path.join(self.output_dir, f"diag_{int(current_time)}.png")
            cv2.imwrite(save_path, cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR))
            self.last_save_time = current_time
        
        return self.detected_objects

    def print_summary(self):
        if not self.detected_objects: return
        header = f"{'ID':<3} | {'Type':<20} | {'X':<8} | {'Y':<8} | {'Z':<8}"
        print(f"\n{'='*55}\nRUN: {self.run_id}\n{'-'*55}\n{header}\n{'-'*55}")
        for i, obj in enumerate(self.detected_objects):
            print(f"{i:<3} | {obj['label']:<20} | {obj['x']:<8.4f} | {obj['y']:<8.4f} | {obj['z']:<8.4f}")
        print("="*55 + "\n")