import numpy as np

def solve_extrinsics_from_points(pixel_points):
    # 1. Intrinsics from camera_info.txt
    fx = fy = 448.15853517
    cx, cy = 640.0, 360.0
    
    # 2. Convert to Camera Frame 3D Points
    camera_pts = []
    for u, v, d in pixel_points:
        xc = (u - cx) * d / fx
        yc = (v - cy) * d / fy
        zc = d
        camera_pts.append([xc, yc, zc])
    camera_pts = np.array(camera_pts)

    # 3. Fit Plane: aX + bY + cZ + d = 0
    # Center the data
    centroid = np.mean(camera_pts, axis=0)
    centered_pts = camera_pts - centroid
    # Singular Value Decomposition to find the normal (smallest singular value)
    _, _, vh = np.linalg.svd(centered_pts)
    normal = vh[2, :] # This is the Z-axis of our Table Frame
    
    # Ensure normal points 'up' towards the camera
    if normal[2] > 0: normal = -normal

    # 4. Construct Rotation Matrix (R)
    # New Z is the plane normal
    z_axis = -normal 
    # New X is Right (orthogonal to Z and World Up/Camera Y)
    x_axis = np.cross([0, 1, 0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    # New Y is Forward along the table
    y_axis = np.cross(z_axis, x_axis)
    
    R = np.stack([x_axis, y_axis, z_axis], axis=1).T
    
    # 5. Define Origin
    # Project optical axis (0,0,1) onto the plane to find the intersection
    # Ray: P = t*[0,0,1]. Plane: dot(n, P - centroid) = 0
    t = np.dot(normal, centroid) / normal[2]
    intersection_point = np.array([0, 0, t])
    
    return R, intersection_point

def transform_precise(u, v, d, R, origin):
    fx = fy = 448.15853517
    cx, cy = 640.0, 360.0
    
    # Back-project
    p_c = np.array([
        (u - cx) * d / fx,
        (v - cy) * d / fy,
        d
    ])
    
    # Transform: P_table = R * (P_camera - Origin)
    p_table = R @ (p_c - origin)
    return p_table

# Data
data_points = [
    (306, 169, 0.907639),
    (968, 154, 0.929449),
    (321, 464, 0.618921),
    (980, 461, 0.620822)
]

R_final, origin_final = solve_extrinsics_from_points(data_points)

print(f"{'u':>5} {'v':>5} | {'X_table':>10} {'Y_table':>10} {'Z_table':>10}")
for u, v, d in data_points:
    xt, yt, zt = transform_precise(u, v, d, R_final, origin_final)
    print(f"{u:5} {v:5} | {xt:10.4f} {yt:10.4f} {zt:10.6f}")


'''
import numpy as np

def convert_to_table_frame(u, v, depth):
    """
    Converts depth camera pixel coordinates (u, v) and depth (meters) 
    into a table-aligned coordinate system (X: right, Y: forward, Z: normal).
    
    Args:
        u (float): Pixel x-coordinate.
        v (float): Pixel y-coordinate.
        depth (float): Depth value in meters.
        
    Returns:
        np.array: [x_table, y_table, z_table] in meters.
    """
    # 1. Camera Intrinsics (Extracted from sensor_msgs.msg.CameraInfo)
    # Principal point (cx, cy) and focal lengths (fx, fy)
    fx = 448.15853517 [cite: 1]
    fy = 448.15853517 [cite: 1]
    cx = 640.0 [cite: 1]
    cy = 360.0 [cite: 1]
    
    # 2. Camera Extrinsics (Derived from plane-fit of provided data points)
    # The point where the camera's optical axis hits the table surface
    z_intersection = 0.69702865 [cite: 1]
    
    # Rotation Matrix: Columns are the Table-Frame axes expressed in the Camera Frame
    # R[:, 0] = Table X (Right), R[:, 1] = Table Y (Forward), R[:, 2] = Table Z (Normal)
    R = np.array([
        [ 1.00000000e+00, -2.13171516e-04, -1.16013877e-04],
        [-2.42695931e-04, -8.78348099e-01, -4.78021504e-01],
        [ 0.00000000e+00,  4.78021518e-01, -8.78348125e-01]
    ]) [cite: 1]

    # 3. Step 1: Back-project pixel to Camera Optical Frame
    # X_c: Right, Y_c: Down, Z_c: Forward
    xc = (u - cx) * depth / fx [cite: 1]
    yc = (v - cy) * depth / fy [cite: 1]
    zc = depth [cite: 1]
    
    # 4. Step 2: Shift origin to the table intersection point
    # We subtract the intersection depth from the Z_c component
    pc_shifted = np.array([xc, yc, zc - z_intersection]) [cite: 1]
    
    # 5. Step 3: Rotate into Table Frame
    # P_table = R^T * (P_camera_shifted)
    p_table = R.T @ pc_shifted [cite: 1]
    
    return p_table

# Example usage with your provided data:
if __name__ == "__main__":
    test_points = [
        (306, 169, 0.907639),
        (968, 154, 0.929449),
        (321, 464, 0.618921),
        (980, 461, 0.620822)
    ]
    
    print(f"{'u':>5} {'v':>5} | {'X_table':>10} {'Y_table':>10} {'Z_table':>10}")
    for u, v, d in test_points:
        coords = convert_to_table_frame(u, v, d)
        print(f"{u:5} {v:5} | {coords[0]:10.4f} {coords[1]:10.4f} {coords[2]:10.6f}")
'''