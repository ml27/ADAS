# ADAS System Architecture

## High-Level Overview

```
┌─────────────────┐
│  USB Camera     │
│   (Port 3)      │
└────────┬────────┘
         │ Video Feed
         ▼
┌─────────────────────────────────────────┐
│     Camera Capture Module               │
│  (camera_capture.py)                    │
│  • Initialize camera                    │
│  • Set resolution & FPS                 │
│  • Read frames continuously             │
└────────┬────────────────────────────────┘
         │ Raw Frames (BGR)
         ▼
┌─────────────────────────────────────────┐
│   Segmentation Module                   │
│   (segmentation.py)                     │
│  • Load HuggingFace model               │
│  • Process frame through neural network │
│  • Generate segmentation map            │
└────────┬────────────────────────────────┘
         │ Segmentation Map
         ▼
┌─────────────────────────────────────────┐
│   Drivable Area Extraction              │
│   (segmentation.py)                     │
│  • Extract drivable classes (road, etc) │
│  • Create binary mask                   │
└────────┬────────────────────────────────┘
         │ Binary Mask
         ▼
┌─────────────────────────────────────────┐
│   Visualization                         │
│   (segmentation.py)                     │
│  • Apply green overlay                  │
│  • Blend with original frame            │
└────────┬────────────────────────────────┘
         │ Processed Frame
         ▼
┌─────────────────────────────────────────┐
│   Display / Output                      │
│   (main.py)                             │
│  • Show in window                       │
│  • Optional: Save to file               │
└─────────────────────────────────────────┘
```

## Component Details

### 1. Camera Capture Module
**File:** `camera_capture.py`

**Responsibilities:**
- Initialize USB camera connection
- Configure resolution, FPS, and other parameters
- Continuously capture frames
- Handle cleanup and resource management

**Key Methods:**
- `initialize()` - Open camera connection
- `read_frame()` - Get next frame
- `release()` - Close camera and free resources

### 2. Segmentation Module
**File:** `segmentation.py`

**Responsibilities:**
- Load and manage HuggingFace models
- Perform semantic segmentation on frames
- Extract drivable area classes
- Create visualization overlays

**Key Methods:**
- `load_model()` - Initialize neural network
- `segment_frame()` - Process frame through model
- `extract_drivable_mask()` - Identify drivable pixels
- `create_overlay()` - Generate visual output

### 3. Configuration
**File:** `config.py`

**Contains:**
- Camera settings (port, resolution, FPS)
- Model selection and parameters
- Drivable area class definitions
- Processing and display options
- Performance tuning parameters

### 4. Main Application
**File:** `main.py`

**Responsibilities:**
- Command-line interface
- Coordinate all components
- Main processing loop
- User interaction handling
- Resource cleanup

## Data Flow

### Frame Processing Pipeline

```
Input Frame (640x480 BGR)
         ↓
    Preprocessing
    • Convert to RGB
    • Normalize
    • Resize for model
         ↓
  Neural Network Inference
  • SegFormer Model
  • Semantic Segmentation
  • ~100ms on CPU, ~20ms on GPU
         ↓
  Segmentation Map (HxW)
  • Each pixel has class ID
  • 0-149 classes (ADE20K)
  • 0-18 classes (Cityscapes)
         ↓
  Class Filtering
  • Extract road (class 6)
  • Extract sidewalk (class 11)
  • Create binary mask
         ↓
  Binary Mask (640x480)
  • 255 = Drivable
  • 0 = Non-drivable
         ↓
  Overlay Creation
  • Green tint on drivable areas
  • Alpha blending (50%)
  • Add frame counter
         ↓
  Output Frame (640x480 BGR)
```

## Model Information

### Default Model: SegFormer-B0
- **Name:** nvidia/segformer-b0-finetuned-ade-512-512
- **Architecture:** SegFormer (hierarchical Transformer)
- **Dataset:** ADE20K (150 classes)
- **Size:** ~14MB
- **Speed:** Fast (suitable for real-time)
- **Accuracy:** Good for general scenes

### Alternative: SegFormer-B1 (Cityscapes)
- **Name:** nvidia/segformer-b1-finetuned-cityscapes-1024-1024
- **Architecture:** SegFormer
- **Dataset:** Cityscapes (19 classes, optimized for driving)
- **Size:** ~50MB
- **Speed:** Medium
- **Accuracy:** Excellent for road scenes

### Class Mappings

**ADE20K Classes (Default Model):**
- Class 6: Road
- Class 11: Sidewalk
- Class 13: Building
- Class 0: Wall
- etc.

**Cityscapes Classes (Alternative Model):**
- Class 0: Road
- Class 1: Sidewalk
- Class 2: Building
- Class 7: Terrain
- etc.

## Processing Performance

### Factors Affecting Speed:
1. **Hardware**
   - CPU: 5-10 FPS
   - GPU (CUDA): 30-60 FPS

2. **Model Size**
   - B0 (small): Fastest
   - B1 (medium): Slower but more accurate
   - B5 (large): Slowest but best quality

3. **Resolution**
   - 320x240: Very fast
   - 640x480: Good balance (default)
   - 1280x720: Slower but better quality

4. **Frame Skipping**
   - Process every frame: Most accurate
   - Process every 2nd frame: 2x faster
   - Process every 5th frame: 5x faster

## Usage Patterns

### Pattern 1: Real-Time Monitoring
```python
# Continuous processing for monitoring
while True:
    frame = camera.read_frame()
    mask = model.segment_frame(frame)
    display(mask)
```

### Pattern 2: Triggered Processing
```python
# Process only when needed
if should_process():
    frame = camera.read_frame()
    mask = model.segment_frame(frame)
    make_decision(mask)
```

### Pattern 3: Batch Processing
```python
# Process multiple frames
frames = capture_n_frames(100)
for frame in frames:
    mask = model.segment_frame(frame)
    save_result(mask)
```

## Configuration Strategies

### High Performance (GPU Available)
```python
USE_GPU = True
PROCESS_EVERY_N_FRAMES = 1
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
```

### Balanced (CPU Only)
```python
USE_GPU = False
PROCESS_EVERY_N_FRAMES = 2
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
```

### High Accuracy (GPU Required)
```python
USE_GPU = True
PROCESS_EVERY_N_FRAMES = 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MODEL_NAME = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
```

### Low Resource (Embedded Systems)
```python
USE_GPU = False
PROCESS_EVERY_N_FRAMES = 5
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
```

## Extension Points

### Adding New Models
1. Update `MODEL_NAME` in config.py
2. Verify model is compatible (semantic segmentation)
3. Update `DRIVABLE_CLASSES` based on model's dataset
4. Test and tune parameters

### Adding New Classes
1. Identify class IDs from model documentation
2. Update `DRIVABLE_CLASSES` list in config.py
3. Test segmentation accuracy

### Custom Processing
1. Extend `SegmentationModel` class
2. Override `segment_frame()` or add new methods
3. Integrate in main processing loop

### Multiple Cameras
1. Create multiple `CameraCapture` instances
2. Process each feed independently or in parallel
3. Combine results as needed
