import cv2
import numpy as np
import argparse

def inspect_depth(image_path):
    # Load the depth image in its original format
    # IMREAD_UNCHANGED is critical for depth maps
    depth_map = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if depth_map is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Create a display-friendly version (normalized for visibility)
    # Since your values are 0-3, we scale them up so they aren't just black
    if depth_map.max() > 0:
        display_img = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        display_img = cv2.applyColorMap(display_img, cv2.COLORMAP_JET)
    else:
        display_img = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get the raw value from the original depth map
            val = depth_map[y, x]
            print(f"Pixel at (x={x}, y={y}) | Raw Depth Value: {val}")
            
            # Temporary overlay on the display image
            temp_img = display_img.copy()
            cv2.circle(temp_img, (x, y), 5, (255, 255, 255), -1)
            cv2.putText(temp_img, f"Depth: {val}", (x + 10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Depth Inspector", temp_img)

    cv2.namedWindow("Depth Inspector")
    cv2.setMouseCallback("Depth Inspector", mouse_callback)

    print("--- Depth Inspector ---")
    print(f"Image Resolution: {depth_map.shape[1]}x{depth_map.shape[0]}")
    print(f"Data Type: {depth_map.dtype}")
    print("Click anywhere on the image to see the depth value.")
    print("Press 'q' or 'Esc' to exit.")

    while True:
        cv2.imshow("Depth Inspector", display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively inspect depth values.")
    parser.add_argument("--depth", type=str, default="0_depth.png", help="Path to depth image")
    args = parser.parse_args()

    inspect_depth(args.depth)