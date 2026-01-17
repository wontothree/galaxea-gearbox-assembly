import sys
import os
from pxr import Sdf

def initialize_ros2_environment():
    """Enables extensions and opens the Action Graph Editor."""
    ext_path = "/isaac-sim/exts"
    if os.path.exists(ext_path) and ext_path not in sys.path:
        sys.path.append(ext_path)

    try:
        from isaacsim.core.utils.extensions import enable_extension
        import omni.kit.commands
        
        enable_extension("isaacsim.ros2.bridge")
        enable_extension("omni.graph.action")
        enable_extension("omni.graph.window.action")

        print("[Success] ROS2 and Action Graph extensions enabled.")
        omni.kit.commands.execute("ShowWindow", window_name="Action Graph")
    except Exception as e:
        print(f"[Error] Failed to initialize environment: {e}")

def setup_camera_publishing(prim_path, resolution=(320, 240)):
    """
    Workflow: OnPlaybackTick -> IsaacCreateRenderProduct -> ROS2 Helpers
    Configured using atomic SET_VALUES structure.
    """
    import omni.graph.core as og

    graph_path = "/World/Push_Camera_Graph"

    # Override
    prim_path = '/World/envs/env_0/Robot/zed_link/head_cam/head_cam'
    resolution=(320, 240)
    
    try:
        # Build and configure the Graph in one block
        # Indices for reference: OnPlaybackTick [0], RenderProduct [1], RGB [2], Info [3], Depth [4]
        (graph, nodes, _, _) = og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("RGBCameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("CameraInfoHelper", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                    ("DepthCameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ],
                og.Controller.Keys.CONNECT: [
                    # Execution flow
                    ("OnPlaybackTick.outputs:tick", "RenderProduct.inputs:execIn"),
                    ("RenderProduct.outputs:execOut", "RGBCameraHelper.inputs:execIn"),
                    ("RenderProduct.outputs:execOut", "CameraInfoHelper.inputs:execIn"),
                    ("RenderProduct.outputs:execOut", "DepthCameraHelper.inputs:execIn"),
                    # Data flow: Connect Render Product Path to all helpers
                    ("RenderProduct.outputs:renderProductPath", "RGBCameraHelper.inputs:renderProductPath"),
                    ("RenderProduct.outputs:renderProductPath", "CameraInfoHelper.inputs:renderProductPath"),
                    ("RenderProduct.outputs:renderProductPath", "DepthCameraHelper.inputs:renderProductPath"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    # Render Product Configuration
                    ("RenderProduct.inputs:cameraPrim", [prim_path]),
                    # ("RenderProduct.inputs:width", resolution[0]),
                    # ("RenderProduct.inputs:height", resolution[1]),

                    # RGB Helper Configuration
                    ("RGBCameraHelper.inputs:type", "rgb"),
                    ("RGBCameraHelper.inputs:topicName", "/rgb/image_rect_color"),
                    ("RGBCameraHelper.inputs:frameId", "tf_camera"),

                    # Camera Info Helper Configuration
                    ("CameraInfoHelper.inputs:topicName", "/rgb/camera_info"),
                    ("CameraInfoHelper.inputs:frameId", "tf_camera"),

                    # Depth Helper Configuration (Input type: depth)
                    ("DepthCameraHelper.inputs:type", "depth"),
                    ("DepthCameraHelper.inputs:topicName", "/depth_image"),
                    ("DepthCameraHelper.inputs:frameId", "tf_camera"),
                ],
            },
        )

        print(f"[Success] Action Graph atomically configured for: {prim_path}")

    except Exception as e:
        print(f"[Fatal] Failed to setup Action Graph: {e}")