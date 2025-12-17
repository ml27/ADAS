"""
Demo script for testing ADAS segmentation without a physical camera.
Uses a synthetic or sample image for demonstration purposes.
"""

import cv2
import numpy as np
from segmentation import SegmentationModel
import config


def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create a simple test image simulating a road scene.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        np.ndarray: Test image
    """
    # Create a blank image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw sky (blue)
    image[0:height//2, :] = [135, 206, 235]  # Sky blue
    
    # Draw road (gray)
    road_points = np.array([
        [width//4, height//2],
        [3*width//4, height//2],
        [width, height],
        [0, height]
    ])
    cv2.fillPoly(image, [road_points], (100, 100, 100))
    
    # Draw lane markings (white)
    for i in range(0, height, 40):
        y = height//2 + i
        if y < height:
            cv2.line(image, (width//2-2, y), (width//2-2, min(y+20, height)), (255, 255, 255), 4)
    
    # Draw some grass/terrain on sides (green)
    left_points = np.array([
        [0, height//2],
        [width//4, height//2],
        [0, height]
    ])
    right_points = np.array([
        [3*width//4, height//2],
        [width, height//2],
        [width, height]
    ])
    cv2.fillPoly(image, [left_points], (34, 139, 34))
    cv2.fillPoly(image, [right_points], (34, 139, 34))
    
    return image


def demo_without_camera():
    """
    Run a demo without requiring a physical camera.
    """
    print("=" * 60)
    print("ADAS Segmentation Demo (No Camera Required)")
    print("=" * 60)
    
    # Create test image
    print("\n1. Creating synthetic test image...")
    test_image = create_test_image(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    
    # Initialize segmentation model
    print("\n2. Loading segmentation model...")
    segmentation_model = SegmentationModel(model_name=config.MODEL_NAME)
    
    if not segmentation_model.load_model():
        print("Failed to load model")
        return
    
    print("\n3. Processing test image...")
    
    # Perform segmentation
    segmentation = segmentation_model.segment_frame(test_image)
    
    if segmentation is None:
        print("Segmentation failed")
        return
    
    # Extract drivable area
    drivable_mask = segmentation_model.extract_drivable_mask(segmentation)
    
    # Create overlay
    result = segmentation_model.create_overlay(test_image, drivable_mask)
    
    print("\n4. Displaying results...")
    print("   - Original: Synthetic road scene")
    print("   - Segmentation Mask: Binary drivable area mask")
    print("   - Result: Green overlay on drivable areas")
    print("\nPress any key to close windows...")
    
    # Display results
    cv2.imshow("Original Test Image", test_image)
    cv2.imshow("Drivable Area Mask", drivable_mask)
    cv2.imshow("Segmented Result", result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nDemo completed successfully!")
    print("\nTo run with real camera:")
    print("  python main.py --camera-port 3")


if __name__ == "__main__":
    demo_without_camera()
