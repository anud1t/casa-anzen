# Casa Anzen Security System

A modern home security surveillance system with AI-powered object detection and tracking, built on the foundation of Helios Nano.

## Features

- **Real-time Object Detection**: YOLO-based detection with TensorRT acceleration
- **Multi-object Tracking**: Advanced tracking with trajectory analysis
- **Polygon Security Zones**: Interactive, labelable polygon zones with close-on-first-point UX
- **Event Feed**: Live thumbnails for manual captures and zone-entry triggers; double-click to preview
- **SQLite Persistence**: Zones and security events (detections, alerts) persisted to `data/casa_anzen.db`
- **Intelligent Alerts**: AI-powered threat detection and alerting
- **Video Recording**: Automatic recording with motion detection and overlays
- **Modern Dark UI**: Qt Fusion dark theme with collapsible side panel
- **CUDA Acceleration**: GPU-optimized processing pipeline
- **Cross-platform**: Linux support with Windows/macOS compatibility

## Key Capabilities

### Security Monitoring
- **Intrusion Detection**: Real-time detection of unauthorized access
- **Suspicious Behavior Analysis**: AI-powered analysis of movement patterns
- **Zone-based Alerts**: Configurable security zones with custom alert levels
- **Person Recognition**: Authorized vs unauthorized person detection

### Video Management
- **Automatic Recording**: Motion-triggered recording with configurable quality
- **Overlay System**: Real-time overlays showing detections, alerts, and system info
- **File Management**: Automatic cleanup of old recordings
- **Multiple Formats**: Support for various video formats and codecs

### Performance
- **GPU Acceleration**: CUDA-optimized preprocessing and postprocessing
- **TensorRT Integration**: High-performance inference with NVIDIA TensorRT
- **Multi-threading**: Parallel processing for maximum performance
- **Memory Efficient**: Optimized memory usage for long-running operation

## Getting Started

### Prerequisites

- **NVIDIA GPU** with CUDA support (recommended)
- **Operating System**: Ubuntu 20.04+ or compatible Linux distribution
- **Hardware**: 8GB+ RAM, 10GB+ storage
- **Dependencies**: OpenCV, Qt5, CUDA, TensorRT

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd casa-anzen
   ```

2. **Install dependencies**
   ```bash
   # Install system dependencies
   sudo apt update
   sudo apt install build-essential cmake git
   sudo apt install libopencv-dev libqt5-dev qtbase5-dev
   sudo apt install libcuda-dev nvidia-cuda-toolkit
   
   # Install TensorRT (follow NVIDIA's installation guide)
   # Download from: https://developer.nvidia.com/tensorrt
   ```

3. **Build the project**
   ```bash
   # Standard build
   ./build.sh
   
   # Maximum performance build
   ./build.sh --max-performance
   
   # Debug build
   ./build.sh -b Debug
   ```

4. **Set up models**
   ```bash
   # Create model directories
   mkdir -p models/engine models/onnx
   
   # Convert ONNX models to TensorRT engines
   # (Follow TensorRT conversion guide for your specific models)
   ```

### Usage

**Basic Usage:**
```bash
# Start with camera
./build/casa-anzen -m models/engine/yolo11n.engine --video 0

# Start with video file
./build/casa-anzen -m models/engine/yolo11n.engine --video path/to/video.mp4
```

**Advanced Usage:**
```bash
# Enable recording
./build/casa-anzen -m models/engine/yolo11n.engine --video 0 --recording

# Set confidence threshold
./build/casa-anzen -m models/engine/yolo11n.engine --video 0 --confidence 0.5

# Enable debug mode
./build/casa-anzen -m models/engine/yolo11n.engine --video 0 --debug
```

### UI Guide

- **Drawing Zones**: Click "Draw Zone" in the side panel, then click to add vertices. Click near the first vertex to close the polygon. Enter a label before drawing to name the zone; otherwise, it is auto-named. Zones are saved to SQLite automatically.
- **Clear/Hide Zones**: Use "Clear Zones" to remove all zones (also persisted). Use "Hide/Show Zones" to toggle zone overlays without deleting them.
- **Event Feed**: Shows thumbnails for manual captures and zone-entry triggers only. Double-click a thumbnail to preview the full image.

### Database

- Location: `data/casa_anzen.db` (WAL mode enabled for concurrent access)
- Tables: `security_zones`, `detections`, `security_alerts`, `system_config`
- Examples:
```bash
sqlite3 data/casa_anzen.db ".tables"
sqlite3 -header -csv data/casa_anzen.db "SELECT timestamp,class_name,confidence FROM detections ORDER BY id DESC LIMIT 20;"
```

## Configuration

### Security Zones

Security zones can be configured through the GUI or configuration files:

- **Restricted Zones**: No access allowed, triggers high-priority alerts
- **Monitored Zones**: Access allowed but monitored
- **Sensitive Zones**: High-security areas with strict access control
- **Perimeter Zones**: Property boundary monitoring

### Recording Settings

- **Quality**: Adjustable video quality (1-10)
- **Format**: Multiple video formats supported
- **Motion Detection**: Optional motion-only recording
- **Duration Limits**: Configurable maximum recording duration
- **Cleanup**: Automatic cleanup of old recordings

## Architecture

### Core Components

- **YoloDetector**: Object detection with TensorRT acceleration
- **Tracker**: Multi-object tracking with trajectory analysis
- **SecurityDetector**: AI-powered threat detection and alerting
- **ZoneManager**: Security zone management and monitoring
- **RecordingManager**: Video recording and file management

### Qt Interface

- **SecurityDashboard**: Main application window with dark theme and side panel
- **VideoDisplayWidget**: Video display, detection/alert overlays, and polygon zone drawing
- **Side Panel**: Event feed with previews, zone controls (draw, label, clear, show/hide)

## Performance

### Recommended Hardware

- **GPU**: NVIDIA RTX 3060 or better
- **CPU**: Intel i5-8400 or AMD Ryzen 5 3600
- **RAM**: 16GB or more
- **Storage**: SSD recommended for video recording

### Performance Metrics (reference)

- **Detection FPS**: 30+ FPS on RTX 3060 (model-dependent)
- **Latency**: Low latency pipeline; WAL-enabled DB logging

## Development

### Building from Source

```bash
# Clone repository
git clone <repository-url>
cd casa-anzen

# Build
./build.sh

# Run tests
./build.sh -b Debug
cd build
ctest
```

### Code Structure

```
casa-anzen/
├── src/                    # Source code
│   ├── core/              # Core detection and tracking
│   ├── qt/                # Qt GUI components
│   └── utils/             # Utility functions
├── include/               # Header files
├── models/                # AI models
├── data/                  # Configuration and recordings
├── assets/                # Media assets
└── docs/                  # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on the foundation of Helios Nano
- YOLO object detection models
- NVIDIA TensorRT for inference acceleration
- Qt framework for GUI
- OpenCV for computer vision operations

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation in the `docs/` directory
- Review the troubleshooting guide

---

**Casa Anzen Security System** - Protecting what matters most.