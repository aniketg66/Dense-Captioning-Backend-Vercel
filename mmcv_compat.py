"""
MMCV compatibility module for Python 3.13
Provides MMCV-like functionality using OpenCV and other available libraries
"""

import cv2
import numpy as np
from typing import Optional, Union
import os

def imread(img_path: str, flag: str = 'color', channel_order: str = 'bgr') -> Optional[np.ndarray]:
    """
    Read image using OpenCV (MMCV-compatible interface)
    
    Args:
        img_path (str): Path to the image file
        flag (str): Reading flag ('color', 'grayscale', 'unchanged')
        channel_order (str): Channel order ('bgr', 'rgb')
    
    Returns:
        np.ndarray: Image array or None if failed
    """
    if not os.path.exists(img_path):
        print(f"Warning: Image file {img_path} does not exist")
        return None
    
    # Map MMCV flags to OpenCV flags
    flag_map = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    
    cv_flag = flag_map.get(flag, cv2.IMREAD_COLOR)
    
    # Read image with OpenCV
    img = cv2.imread(img_path, cv_flag)
    
    if img is None:
        print(f"Warning: Failed to read image {img_path}")
        return None
    
    # Convert channel order if needed
    if channel_order == 'rgb' and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

def imwrite(img_path: str, img: np.ndarray, params: Optional[list] = None) -> bool:
    """
    Write image using OpenCV (MMCV-compatible interface)
    
    Args:
        img_path (str): Path to save the image
        img (np.ndarray): Image array
        params (list, optional): Encoding parameters
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    
    # Convert RGB to BGR if needed (OpenCV expects BGR)
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Check if image is in RGB format (common when working with PIL/matplotlib)
        if img.dtype == np.uint8 and np.max(img[:, :, 0]) > np.max(img[:, :, 2]):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return cv2.imwrite(img_path, img, params or [])

def resize(img: np.ndarray, size: Union[tuple, int], interpolation: str = 'bilinear') -> np.ndarray:
    """
    Resize image using OpenCV (MMCV-compatible interface)
    
    Args:
        img (np.ndarray): Input image
        size (tuple or int): Target size (width, height) or scale factor
        interpolation (str): Interpolation method
    
    Returns:
        np.ndarray: Resized image
    """
    # Map interpolation methods
    interp_map = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    cv_interp = interp_map.get(interpolation, cv2.INTER_LINEAR)
    
    if isinstance(size, int):
        # Scale factor
        h, w = img.shape[:2]
        new_w, new_h = int(w * size), int(h * size)
    else:
        # Target size (width, height)
        new_w, new_h = size
    
    return cv2.resize(img, (new_w, new_h), interpolation=cv_interp)

# Create a mock mmcv module
class MockMMCV:
    """Mock MMCV module with essential functions"""
    
    @staticmethod
    def imread(img_path: str, flag: str = 'color', channel_order: str = 'bgr') -> Optional[np.ndarray]:
        return imread(img_path, flag, channel_order)
    
    @staticmethod
    def imwrite(img_path: str, img: np.ndarray, params: Optional[list] = None) -> bool:
        return imwrite(img_path, img, params)
    
    @staticmethod
    def resize(img: np.ndarray, size: Union[tuple, int], interpolation: str = 'bilinear') -> np.ndarray:
        return resize(img, size, interpolation)

# Try to import real mmcv, fallback to mock if not available
try:
    import mmcv
    print("✅ Using real MMCV")
except ImportError:
    print("⚠️  MMCV not available, using compatibility layer")
    mmcv = MockMMCV() 