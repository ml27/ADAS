# Quick Start Guide

## Installation

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (for deep learning)
- Transformers (Hugging Face models)
- OpenCV (camera and image processing)
- Other required packages

### Step 2: Verify Installation
```bash
python test_setup.py
```

This script will check:
- Python version
- All dependencies
- GPU/CUDA availability
- Camera access
- Model loading capability

## Quick Start

### Option 1: Run with Camera
```bash
python main.py
```

This will:
1. Initialize USB camera on port 3
2. Load the segmentation model
3. Display real-time drivable area detection
4. Show green overlay on drivable areas

**Controls:**
- Press `q` to quit
- Press `s` to save current frame

### Option 2: Run Demo (No Camera Required)
```bash
python demo.py
```

This will:
1. Create a synthetic road scene
2. Perform segmentation
3. Display results
4. No physical camera needed!

### Option 3: Use as Library
```bash
python examples.py
```

See examples.py for programmatic usage patterns.

## Customization

### Change Camera Port
```bash
python main.py --camera-port 0
```

### Use Different Model
```bash
python main.py --model nvidia/segformer-b1-finetuned-cityscapes-1024-1024
```

### Edit Configuration
Edit `config.py` to change:
- Camera resolution and FPS
- Model name
- Drivable area classes
- Processing parameters
- Display and output settings

## Troubleshooting

### "Camera not found"
- Check USB connection
- Try different port: `--camera-port 0` or `--camera-port 1`
- Run: `ls /dev/video*` (Linux) to see available cameras

### "Out of memory"
- Use smaller model (segformer-b0)
- Reduce resolution in config.py
- Set `PROCESS_EVERY_N_FRAMES = 2` or higher

### "Slow performance"
- Install CUDA if you have NVIDIA GPU
- Use GPU-accelerated model
- Skip frames: `PROCESS_EVERY_N_FRAMES = 3`
- Lower camera resolution

### Dependencies won't install
```bash
# Update pip first
pip install --upgrade pip

# Try again
pip install -r requirements.txt

# If still fails, install one by one:
pip install torch torchvision
pip install transformers
pip install opencv-python pillow numpy
```

## System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- CPU: Any modern processor
- Camera: USB webcam

**Recommended:**
- Python 3.10+
- 8GB+ RAM
- GPU: NVIDIA GPU with CUDA support
- Camera: HD webcam (720p or higher)

## Next Steps

1. **Verify Setup**: Run `python test_setup.py`
2. **Test Without Camera**: Run `python demo.py`
3. **Test With Camera**: Run `python main.py`
4. **Customize**: Edit `config.py` for your needs
5. **Integrate**: See `examples.py` for library usage

## Additional Information

- **Full Documentation**: See README.md
- **Configuration**: See config.py
- **Examples**: See examples.py
- **Support**: Check troubleshooting section above

## Performance Tips

1. **Enable GPU acceleration** - Significant speedup
2. **Use appropriate model** - b0 for speed, b1 for accuracy
3. **Adjust frame processing** - Skip frames if needed
4. **Optimize resolution** - Lower resolution = faster processing
5. **Close other applications** - Free up system resources
