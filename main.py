"""
Main ADAS application for real-time drivable area segmentation.
Captures frames from USB camera and performs semantic segmentation
to identify drivable areas for autonomous driving.
"""

import cv2
import numpy as np
import argparse
import sys
from camera_capture import CameraCapture
from segmentation import SegmentationModel
import config


class AdasSegmentation:
    """
    Main ADAS segmentation application.
    """
    
    def __init__(self, camera_port: int = None, model_name: str = None):
        """
        Initialize ADAS segmentation system.
        
        Args:
            camera_port: USB camera port (default from config)
            model_name: Hugging Face model name (default from config)
        """
        self.camera_port = camera_port or config.CAMERA_PORT
        self.model_name = model_name or config.MODEL_NAME
        
        self.camera = None
        self.segmentation_model = None
        self.frame_count = 0
        
    def initialize(self) -> bool:
        """
        Initialize camera and segmentation model.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("Initializing ADAS Segmentation System...")
        
        # Initialize camera
        self.camera = CameraCapture(port=self.camera_port)
        if not self.camera.initialize():
            print("Failed to initialize camera")
            return False
        
        # Initialize segmentation model
        self.segmentation_model = SegmentationModel(model_name=self.model_name)
        if not self.segmentation_model.load_model():
            print("Failed to load segmentation model")
            self.camera.release()
            return False
        
        print("System initialized successfully")
        return True
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame through segmentation pipeline.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            tuple: (processed_frame, drivable_mask)
        """
        # Perform segmentation
        segmentation = self.segmentation_model.segment_frame(frame)
        
        if segmentation is None:
            return frame, None
        
        # Extract drivable area mask
        drivable_mask = self.segmentation_model.extract_drivable_mask(segmentation)
        
        # Create overlay visualization
        processed_frame = self.segmentation_model.create_overlay(frame, drivable_mask)
        
        return processed_frame, drivable_mask
    
    def run(self):
        """
        Run the main segmentation loop.
        """
        print("\nStarting segmentation loop...")
        print("Press 'q' to quit, 's' to save current frame")
        
        # Optional: Setup video writer for saving output
        video_writer = None
        if config.SAVE_OUTPUT:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(
                config.OUTPUT_PATH,
                fourcc,
                config.CAMERA_FPS,
                (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
            )
        
        try:
            while True:
                # Read frame from camera
                ret, frame = self.camera.read_frame()
                
                if not ret:
                    print("Failed to capture frame")
                    break
                
                self.frame_count += 1
                
                # Process frame (every N frames for performance)
                if self.frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
                    processed_frame, drivable_mask = self.process_frame(frame)
                else:
                    processed_frame = frame
                    drivable_mask = None
                
                # Add frame counter to display
                cv2.putText(
                    processed_frame,
                    f"Frame: {self.frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
                
                # Display output
                if config.DISPLAY_OUTPUT:
                    cv2.imshow("ADAS - Drivable Area Segmentation", processed_frame)
                    
                    if drivable_mask is not None:
                        cv2.imshow("Drivable Area Mask", drivable_mask)
                
                # Save output if enabled
                if video_writer is not None:
                    video_writer.write(processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    filename = f"frame_{self.frame_count}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"Saved frame to {filename}")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            self.cleanup(video_writer)
    
    def cleanup(self, video_writer=None):
        """
        Clean up resources.
        
        Args:
            video_writer: OpenCV video writer object to release
        """
        print("Cleaning up...")
        
        if video_writer is not None:
            video_writer.release()
        
        if self.camera is not None:
            self.camera.release()
        
        cv2.destroyAllWindows()
        print("Cleanup complete")


def main():
    """
    Main entry point for the application.
    """
    parser = argparse.ArgumentParser(
        description="ADAS Image Segmentation for Drivable Area Detection"
    )
    parser.add_argument(
        "--camera-port",
        type=int,
        default=None,
        help=f"USB camera port number (default: {config.CAMERA_PORT})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Hugging Face model name (default: {config.MODEL_NAME})"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display output (for headless operation)"
    )
    
    args = parser.parse_args()
    
    # Update config based on arguments
    if args.no_display:
        config.DISPLAY_OUTPUT = False
    
    # Create and initialize ADAS system
    adas = AdasSegmentation(
        camera_port=args.camera_port,
        model_name=args.model
    )
    
    if not adas.initialize():
        print("Failed to initialize ADAS system")
        sys.exit(1)
    
    # Run the application
    try:
        adas.run()
    except Exception as e:
        print(f"Error during execution: {e}")
        adas.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
