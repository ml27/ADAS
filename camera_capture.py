"""
Camera capture module for ADAS system.
Handles USB camera initialization and frame capture.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import config


class CameraCapture:
    """
    Handles camera capture from USB port.
    """
    
    def __init__(self, port: int = config.CAMERA_PORT, 
                 width: int = config.CAMERA_WIDTH, 
                 height: int = config.CAMERA_HEIGHT,
                 fps: int = config.CAMERA_FPS):
        """
        Initialize camera capture.
        
        Args:
            port: USB camera port number
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        
    def initialize(self) -> bool:
        """
        Initialize camera connection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.port)
            
            if not self.cap.isOpened():
                print(f"Error: Cannot open camera on port {self.port}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            print(f"Camera initialized on port {self.port}")
            print(f"Resolution: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the camera.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        """
        Release camera resources.
        """
        if self.cap is not None:
            self.cap.release()
            print("Camera released")
    
    def __enter__(self):
        """Context manager entry."""
        if not self.initialize():
            raise RuntimeError(f"Failed to initialize camera on port {self.port}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
