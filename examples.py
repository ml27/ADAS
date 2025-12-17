"""
Example usage of ADAS segmentation system as a library.
This shows how to integrate the segmentation components into your own application.
"""

import cv2
import numpy as np
from camera_capture import CameraCapture
from segmentation import SegmentationModel
import config


def example_1_basic_usage():
    """
    Example 1: Basic usage with camera.
    """
    print("Example 1: Basic Usage")
    print("-" * 40)
    
    # Initialize camera
    camera = CameraCapture(port=config.CAMERA_PORT)
    if not camera.initialize():
        print("Failed to initialize camera")
        return
    
    # Initialize segmentation model
    model = SegmentationModel()
    if not model.load_model():
        print("Failed to load model")
        camera.release()
        return
    
    # Process a single frame
    ret, frame = camera.read_frame()
    if ret:
        # Perform segmentation
        segmentation = model.segment_frame(frame)
        drivable_mask = model.extract_drivable_mask(segmentation)
        result = model.create_overlay(frame, drivable_mask)
        
        # Display or save result
        cv2.imshow("Result", result)
        cv2.waitKey(0)
    
    camera.release()
    cv2.destroyAllWindows()


def example_2_custom_configuration():
    """
    Example 2: Using custom configuration.
    """
    print("Example 2: Custom Configuration")
    print("-" * 40)
    
    # Initialize with custom settings
    camera = CameraCapture(
        port=0,  # Different camera port
        width=1280,  # Higher resolution
        height=720,
        fps=60
    )
    
    # Use different model
    model = SegmentationModel(
        model_name="nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
    )
    
    # Rest of the code...
    print("Custom configuration applied")


def example_3_processing_loop():
    """
    Example 3: Continuous processing loop with custom logic.
    """
    print("Example 3: Processing Loop")
    print("-" * 40)
    
    camera = CameraCapture()
    if not camera.initialize():
        return
    
    model = SegmentationModel()
    if not model.load_model():
        camera.release()
        return
    
    frame_count = 0
    
    try:
        while frame_count < 100:  # Process 100 frames
            ret, frame = camera.read_frame()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 5th frame
            if frame_count % 5 == 0:
                segmentation = model.segment_frame(frame)
                drivable_mask = model.extract_drivable_mask(segmentation)
                
                # Calculate drivable area percentage
                drivable_pixels = np.sum(drivable_mask == 255)
                total_pixels = drivable_mask.size
                drivable_percentage = (drivable_pixels / total_pixels) * 100
                
                print(f"Frame {frame_count}: {drivable_percentage:.2f}% drivable area")
    
    except KeyboardInterrupt:
        print("Interrupted")
    
    finally:
        camera.release()


def example_4_custom_classes():
    """
    Example 4: Using custom drivable classes.
    """
    print("Example 4: Custom Drivable Classes")
    print("-" * 40)
    
    model = SegmentationModel()
    model.load_model()
    
    # Create a test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Segment the frame
    segmentation = model.segment_frame(test_frame)
    
    # Extract drivable area with custom classes
    # For Cityscapes: road=0, sidewalk=1, terrain=9
    custom_classes = [0, 1, 9]
    drivable_mask = model.extract_drivable_mask(segmentation, custom_classes)
    
    print("Custom classes applied")


def example_5_batch_processing():
    """
    Example 5: Batch processing of multiple images.
    """
    print("Example 5: Batch Processing")
    print("-" * 40)
    
    model = SegmentationModel()
    model.load_model()
    
    # Simulate multiple images
    images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(5)
    ]
    
    results = []
    for i, image in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}...")
        segmentation = model.segment_frame(image)
        drivable_mask = model.extract_drivable_mask(segmentation)
        result = model.create_overlay(image, drivable_mask)
        results.append(result)
    
    print(f"Processed {len(results)} images")
    return results


def example_6_context_manager():
    """
    Example 6: Using context manager for automatic cleanup.
    """
    print("Example 6: Context Manager Usage")
    print("-" * 40)
    
    # Camera automatically releases when exiting the context
    with CameraCapture(port=config.CAMERA_PORT) as camera:
        model = SegmentationModel()
        model.load_model()
        
        ret, frame = camera.read_frame()
        if ret:
            segmentation = model.segment_frame(frame)
            drivable_mask = model.extract_drivable_mask(segmentation)
            result = model.create_overlay(frame, drivable_mask)
            print("Frame processed successfully")
    
    print("Camera automatically released")


def main():
    """
    Run examples.
    """
    print("=" * 60)
    print("ADAS Segmentation - Usage Examples")
    print("=" * 60)
    print()
    
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Custom Configuration", example_2_custom_configuration),
        ("Processing Loop", example_3_processing_loop),
        ("Custom Classes", example_4_custom_classes),
        ("Batch Processing", example_5_batch_processing),
        ("Context Manager", example_6_context_manager),
    ]
    
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nNote: These examples require dependencies to be installed.")
    print("Run: pip install -r requirements.txt")
    print("\nTo run a specific example, modify this script to call")
    print("the desired example function.")


if __name__ == "__main__":
    main()
