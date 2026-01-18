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
        # Force R to float32 to prevent precision promotion during multiplication
        self.R = np.array([
            [ 1.00000000e+00, -2.13171516e-04, -1.16013877e-04],
            [-2.42695931e-04, -8.78348099e-01, -4.78021504e-01],
            [ 0.00000000e+00,  4.78021518e-01, -8.78348125e-01]
        ], dtype=np.float32)
        
        self.base_offset_x = 0.585
        self.base_offset_y = -0.0015
        self.base_z_ref = 0.902

        self.z_offsets = {
            "Ring Gear": 0.0000,
            "Planetary Carrier": -0.0010,
            "Sun Planetary Gear": -0.0250,
            "Planetary Reducer": -0.0909,
            "Carrier Pin": -0.0320
        }

        self.xy_offsets = {
            "Ring Gear": (0.0000, 0.0000),
            "Planetary Carrier": (0.0000, 0.0000),
            "Sun Planetary Gear": (0.0030, 0.0000),
            "Planetary Reducer": (0.0000, 0.0000),
            "Carrier Pin": (0.0050, 0.0000)
        }

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

    def get_base_coordinates_vectorized(self, u_arr, v_arr, depth_arr):
        """Calculates base coordinates for arrays of pixel points in float32."""
        xc = (u_arr - self.cx) * depth_arr / self.fx
        yc = (v_arr - self.cy) * depth_arr / self.fy
        zc = depth_arr
        
        # Ensure all components are float32
        pc_shifted = np.stack([xc, yc, zc - self.z_intersection], axis=-1).astype(np.float32)
        p_table = pc_shifted @ self.R 
        
        x_tab = p_table[..., 0]
        y_tab = p_table[..., 1]
        z_tab = p_table[..., 2]

        x_base = y_tab + self.base_offset_x
        y_base = -x_tab + self.base_offset_y
        z_base = z_tab + self.base_z_ref
        
        return x_base.astype(np.float32), y_base.astype(np.float32), z_base.astype(np.float32)

    def get_base_coordinates(self, u, v, depth):
        xb, yb, zb = self.get_base_coordinates_vectorized(np.array([u]), np.array([v]), np.array([depth]))
        # Restore explicit cast to ensure Isaac Lab math utils see these as Floats
        return float(np.float32(xb[0])), float(np.float32(yb[0])), float(np.float32(zb[0]))

    def detect_carrier_pins(self, depth_map, carrier_contour, render_img):
        pins = []
        x, y, w, h = cv2.boundingRect(carrier_contour)
        v_grid, u_grid = np.mgrid[y:y+h, x:x+w]
        roi_depth = depth_map[y:y+h, x:x+w]
        mask = roi_depth > 0
        height_map = np.zeros_like(roi_depth, dtype=np.float32)

        if np.any(mask):
            _, _, z_base = self.get_base_coordinates_vectorized(u_grid[mask], v_grid[mask], roi_depth[mask])
            height_map[mask] = z_base

        valid_heights = height_map[height_map > 0]
        if valid_heights.size == 0: return pins
        carrier_floor_z = np.median(valid_heights)

        peak_mask = np.zeros((h, w), dtype=np.uint8)
        peak_mask[height_map > (carrier_floor_z + 0.010)] = 255
        peak_mask = cv2.morphologyEx(peak_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        peak_contours, _ = cv2.findContours(peak_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for p_cnt in peak_contours:
            if 3 < cv2.contourArea(p_cnt) < 150:
                M = cv2.moments(p_cnt)
                if M["m00"] == 0: continue
                u_p, v_p = int(M["m10"] / M["m00"]) + x, int(M["m01"] / M["m00"]) + y
                pz = float(np.float32(height_map[v_p - y, u_p - x] + self.z_offsets.get("Carrier Pin", 0.0)))
                px, py, _ = self.get_base_coordinates(u_p, v_p, float(depth_map[v_p, u_p]))
                px += self.xy_offsets.get("Carrier Pin", (0.0, 0.0))[0]
                py += self.xy_offsets.get("Carrier Pin", (0.0, 0.0))[1]
                pins.append({'label': "Carrier Pin", 'x': round(px, 4), 'y': round(py, 4), 'z': round(pz, 4)})
                cv2.circle(render_img, (u_p, v_p), 6, (255, 0, 0), 2)
                h_mm = round((pz - carrier_floor_z) * 1000, 1)
                cv2.putText(render_img, f"TOP +{h_mm}mm", (u_p + 8, v_p), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        return pins

    def process_frame(self, rgb, depth_map):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        render_img = rgb.copy()
        self.detected_objects = []

        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < 50: continue

            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            u_c, v_c = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            z_c = float(depth_map[v_c, u_c])
            
            _, _, w_px, h_px = cv2.boundingRect(cnt)
            z_ref = z_c if z_c > 0 else np.median(depth_map[v_c-2:v_c+2, u_c-2:u_c+2])
            area_cm2 = (w_px * z_ref / self.fx * 100) * (h_px * z_ref / self.fy * 100)
            label = self.classify_gear(area_cm2)

            if label == "Sun Planetary Gear":
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                surface_mask = cv2.bitwise_and(mask, thresh)
                v_pts, u_pts = np.where((surface_mask > 0) & (depth_map > 0))
                
                if len(u_pts) > 10:
                    d_pts = depth_map[v_pts, u_pts].astype(np.float32)
                    x_bases, y_bases, z_bases = self.get_base_coordinates_vectorized(u_pts, v_pts, d_pts)
                    # Mean calculation in float32
                    x, y, z = np.mean(x_bases), np.mean(y_bases), np.mean(z_bases)
                else:
                    x, y, z = self.get_base_coordinates(u_c, v_c, z_ref)
            else:
                x, y, z = self.get_base_coordinates(u_c, v_c, z_ref)

            # Final precision check and offset application
            z = float(np.float32(z + self.z_offsets.get(label, 0.0)))
            xo, yo = self.xy_offsets.get(label, (0.0, 0.0))
            x = float(np.float32(x + xo))
            y = float(np.float32(y + yo))

            self.detected_objects.append({'label': label, 'x': round(x, 4), 'y': round(y, 4), 'z': round(z, 4)})
            cv2.drawContours(render_img, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(render_img, label, (u_c - 40, v_c - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if label == "Planetary Carrier":
                self.detected_objects.extend(self.detect_carrier_pins(depth_map, cnt, render_img))

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