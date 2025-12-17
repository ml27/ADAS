"""
Configuration file for ADAS image segmentation system.
"""

# Camera Configuration
CAMERA_PORT = 3  # USB camera port number
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Model Configuration
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"  # Hugging Face model for semantic segmentation
# Alternative models:
# "nvidia/segformer-b1-finetuned-cityscapes-1024-1024" - Better for road/driving scenarios
# "facebook/maskformer-swin-base-ade" - More accurate but slower

# Segmentation Configuration
# Common road/drivable area class IDs (varies by model)
# For ADE20K: road=6, sidewalk=11
# For Cityscapes: road=0, sidewalk=1, terrain=9
DRIVABLE_CLASSES = [6, 11]  # road and sidewalk for ADE20K model
MASK_ALPHA = 0.5  # Transparency for overlay visualization

# Processing Configuration
PROCESS_EVERY_N_FRAMES = 1  # Process every frame (set higher to skip frames for performance)
DISPLAY_OUTPUT = True  # Display output window with segmentation overlay
SAVE_OUTPUT = False  # Save output video
OUTPUT_PATH = "output/segmented_output.avi"

# Performance Configuration
USE_GPU = True  # Use GPU if available
INFERENCE_DEVICE = "cuda"  # "cuda" or "cpu"
