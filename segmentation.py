"""
Segmentation module using Hugging Face models for drivable area detection.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from typing import Optional, List
import config


class SegmentationModel:
    """
    Handles semantic segmentation using Hugging Face models.
    """
    
    def __init__(self, model_name: str = config.MODEL_NAME,
                 device: Optional[str] = None):
        """
        Initialize segmentation model.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        
        # Determine device
        if device is None:
            if config.USE_GPU and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.processor = None
        self.model = None
        
    def load_model(self) -> bool:
        """
        Load the segmentation model and processor.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Loading model: {self.model_name}")
            print(f"Using device: {self.device}")
            
            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForSemanticSegmentation.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print("Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def segment_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Perform semantic segmentation on a frame.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            np.ndarray: Segmentation mask with class IDs for each pixel
        """
        if self.model is None or self.processor is None:
            print("Error: Model not loaded")
            return None
        
        try:
            # Convert BGR to RGB
            image = Image.fromarray(frame[..., ::-1])
            
            # Preprocess image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get segmentation mask
            logits = outputs.logits
            segmentation = logits.argmax(dim=1).squeeze().cpu().numpy()
            
            # Resize to original frame size
            segmentation_resized = Image.fromarray(segmentation.astype(np.uint8))
            segmentation_resized = segmentation_resized.resize(
                (frame.shape[1], frame.shape[0]), 
                Image.NEAREST
            )
            
            return np.array(segmentation_resized)
            
        except Exception as e:
            print(f"Error during segmentation: {e}")
            return None
    
    def extract_drivable_mask(self, segmentation: np.ndarray,
                             drivable_classes: List[int] = None) -> np.ndarray:
        """
        Extract binary mask for drivable area.
        
        Args:
            segmentation: Segmentation mask with class IDs
            drivable_classes: List of class IDs considered drivable
            
        Returns:
            np.ndarray: Binary mask (255 for drivable, 0 for non-drivable)
        """
        if drivable_classes is None:
            drivable_classes = config.DRIVABLE_CLASSES
        
        # Create binary mask
        drivable_mask = np.zeros_like(segmentation, dtype=np.uint8)
        
        for class_id in drivable_classes:
            drivable_mask[segmentation == class_id] = 255
        
        return drivable_mask
    
    def create_overlay(self, frame: np.ndarray, mask: np.ndarray,
                      color: tuple = (0, 255, 0),
                      alpha: float = config.MASK_ALPHA) -> np.ndarray:
        """
        Create an overlay visualization of the drivable area mask.
        
        Args:
            frame: Original frame
            mask: Binary mask of drivable area
            color: Color for overlay (B, G, R)
            alpha: Transparency (0-1)
            
        Returns:
            np.ndarray: Frame with overlay
        """
        overlay = frame.copy()
        
        # Create colored mask
        colored_mask = np.zeros_like(frame)
        colored_mask[mask == 255] = color
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 1, colored_mask, alpha, 0)
        
        return result
