"""
Test script to verify installation and check system requirements.
"""

import sys


def check_python_version():
    """Check if Python version is adequate."""
    print("Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ❌ Python 3.8 or higher required")
        return False
    print("  ✓ Python version OK")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'transformers': 'Hugging Face Transformers',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy'
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            if module == 'cv2':
                import cv2
                version = cv2.__version__
            elif module == 'PIL':
                from PIL import Image
                version = Image.__version__ if hasattr(Image, '__version__') else "unknown"
            elif module == 'torch':
                import torch
                version = torch.__version__
            elif module == 'torchvision':
                import torchvision
                version = torchvision.__version__
            elif module == 'transformers':
                import transformers
                version = transformers.__version__
            elif module == 'numpy':
                import numpy
                version = numpy.__version__
            
            print(f"  ✓ {name}: {version}")
        except ImportError:
            print(f"  ❌ {name}: Not installed")
            all_ok = False
    
    return all_ok


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA/GPU support...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA Version: {torch.version.cuda}")
        else:
            print("  ℹ CUDA not available (will use CPU)")
        return True
    except ImportError:
        print("  ❌ Cannot check CUDA (PyTorch not installed)")
        return False


def check_camera():
    """Check if camera is accessible."""
    print("\nChecking camera access...")
    try:
        import cv2
        import config
        
        # Try to open camera
        cap = cv2.VideoCapture(config.CAMERA_PORT)
        if cap.isOpened():
            print(f"  ✓ Camera found on port {config.CAMERA_PORT}")
            ret, frame = cap.read()
            if ret:
                print(f"    Resolution: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
            return True
        else:
            print(f"  ⚠ Camera not found on port {config.CAMERA_PORT}")
            print("    This is OK - you can still run the demo without camera")
            return True
    except Exception as e:
        print(f"  ⚠ Error checking camera: {e}")
        print("    This is OK - you can still run the demo without camera")
        return True


def check_model():
    """Check if model can be loaded."""
    print("\nChecking model loading (this may take a while)...")
    try:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        import config
        
        print(f"  Loading model: {config.MODEL_NAME}")
        processor = AutoImageProcessor.from_pretrained(config.MODEL_NAME)
        model = AutoModelForSemanticSegmentation.from_pretrained(config.MODEL_NAME)
        print("  ✓ Model loaded successfully")
        return True
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ADAS System Check")
    print("=" * 60)
    
    results = []
    
    results.append(("Python Version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("CUDA/GPU", check_cuda()))
    results.append(("Camera", check_camera()))
    results.append(("Model Loading", check_model()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, status in results:
        status_str = "✓ PASS" if status else "❌ FAIL"
        print(f"{name:.<40} {status_str}")
    
    all_passed = all(status for _, status in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All checks passed!")
        print("\nYou can now run:")
        print("  python main.py              # Run with camera")
        print("  python demo.py              # Run demo without camera")
    else:
        print("❌ Some checks failed")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
