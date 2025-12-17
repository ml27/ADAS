# ADAS - Advanced Driver Assistance System

Real-time image segmentation for drivable area detection using deep learning models from Hugging Face.

## Overview

This system captures frames from a USB camera (port 3) and performs semantic segmentation frame-by-frame to identify and mask drivable areas for autonomous driving applications. It uses state-of-the-art deep learning models from Hugging Face's Transformers library.

## Features

- **Real-time Processing**: Frame-by-frame segmentation of camera feed
- **USB Camera Support**: Captures video from USB camera port 3
- **Deep Learning Models**: Uses Hugging Face transformer models (SegFormer, MaskFormer)
- **Drivable Area Detection**: Identifies roads, sidewalks, and other drivable surfaces
- **Visual Overlay**: Real-time visualization with green overlay on drivable areas
- **Configurable**: Easy-to-modify configuration for different models and settings
- **GPU Acceleration**: Automatic GPU detection and usage for faster inference

## Installation

### Prerequisites

- Python 3.8 or higher
- USB camera connected to port 3
- (Optional) CUDA-compatible GPU for faster inference

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ml27/ADAS.git
cd ADAS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the application with default settings:
```bash
python main.py
```

### Command Line Options

```bash
python main.py --help
```

Options:
- `--camera-port PORT`: Specify USB camera port (default: 3)
- `--model MODEL_NAME`: Specify Hugging Face model (default: nvidia/segformer-b0-finetuned-ade-512-512)
- `--no-display`: Run without display output (headless mode)

### Examples

```bash
# Use a different camera port
python main.py --camera-port 0

# Use Cityscapes model (better for road scenes)
python main.py --model nvidia/segformer-b1-finetuned-cityscapes-1024-1024

# Run in headless mode
python main.py --no-display
```

### Controls

While the application is running:
- Press `q` to quit
- Press `s` to save the current frame

## Configuration

Edit `config.py` to customize:

- **Camera Settings**: Port, resolution, FPS
- **Model Selection**: Choose different Hugging Face models
- **Segmentation Classes**: Define which classes are considered drivable
- **Visualization**: Adjust overlay transparency and colors
- **Performance**: GPU usage, frame skipping, output saving

### Recommended Models

1. **nvidia/segformer-b0-finetuned-ade-512-512** (Default)
   - Fast and lightweight
   - Good general-purpose segmentation

2. **nvidia/segformer-b1-finetuned-cityscapes-1024-1024**
   - Optimized for road/driving scenarios
   - Higher accuracy for autonomous driving

3. **facebook/maskformer-swin-base-ade**
   - More accurate but slower
   - Best quality segmentation

## Architecture

```
main.py                 # Main application entry point
camera_capture.py       # Camera interface and frame capture
segmentation.py         # Segmentation model and processing
config.py               # Configuration settings
requirements.txt        # Python dependencies
```

## How It Works

1. **Camera Initialization**: Connects to USB camera on specified port
2. **Model Loading**: Downloads and loads Hugging Face segmentation model
3. **Frame Capture**: Continuously captures frames from camera feed
4. **Segmentation**: Each frame is processed through the neural network
5. **Mask Extraction**: Drivable area classes are extracted into a binary mask
6. **Visualization**: Green overlay is applied to drivable areas
7. **Display**: Processed frame is shown in real-time window

## Troubleshooting

### Camera not detected
- Check USB connection
- Verify camera port number with: `ls /dev/video*` (Linux) or Device Manager (Windows)
- Try different port numbers: `python main.py --camera-port 0`

### Out of memory errors
- Use a smaller model (segformer-b0 instead of b1)
- Reduce camera resolution in `config.py`
- Process fewer frames: Set `PROCESS_EVERY_N_FRAMES = 2` or higher

### Slow performance
- Ensure CUDA/GPU is properly installed
- Use smaller model
- Increase `PROCESS_EVERY_N_FRAMES` to skip frames
- Reduce camera resolution

## Requirements

- torch>=2.6.0
- torchvision>=0.20.0
- transformers>=4.48.0
- opencv-python>=4.8.1.78
- pillow>=10.3.0
- numpy>=1.24.0

## License

This project is open source and available for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Future Enhancements

- [ ] Support for video file input (in addition to camera)
- [ ] Multi-class segmentation with different colors
- [ ] Lane detection integration
- [ ] Object detection for vehicles and pedestrians
- [ ] Recording and playback functionality
- [ ] Performance metrics and FPS counter
- [ ] Web interface for remote monitoring
