# Casa Anzen Security System

A modern home security surveillance system with AI-powered object detection and tracking.

## Features

- **Real-time Object Detection**: YOLO-based detection with TensorRT acceleration
- **Multi-object Tracking**: Advanced tracking with trajectory analysis
- **Polygon Security Zones**: Interactive, labelable polygon zones with close-on-first-point UX
- **Event Feed**: Live thumbnails for manual captures and zone-entry triggers; double-click to preview; manual AI captioning
- **SQLite Persistence**: Zones and security events (detections, alerts) persisted to `data/casa_anzen.db`
- **Intelligent Alerts**: AI-powered threat detection and alerting
- **Video Recording**: Automatic recording with motion detection and overlays
- **Minimal Military UI**: Clean, high-contrast theme with thin outlines and compact spacing
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

## What's New (Latest)

- **Modular Qt architecture**: `MainCoordinator`, `UICoordinator`, and dedicated managers (`EventManager`, `CaptionManager`, `VideoProcessingCoordinator`, `SystemStatusManager`).
- **Event Feed overhaul**:
  - Manual captioning via Moondream at `http://localhost:2020/v1/caption` (see Captioning section).
  - Clear selection indicator (thin green outline), fixed thumbnail spacing, safe deletions (no crashes).
  - Captions render below the “Captured at… Saved to…” block with proper wrapping.
- **Zones UX**: Draw/finish/hide/show controls stabilized; zones persist and can be toggled without losing data.
- **Quiet-by-default runtime**: Qt debug/info disabled, OpenCV set to errors only, GStreamer debug off, and internal logger defaults to warnings.
- **CUDA console noise removed**: Device/memory prints suppressed in release.

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

### Reducing/Increasing Logs

The app is quiet by default. To temporarily increase verbosity for troubleshooting:

```bash
QT_LOGGING_RULES="*.debug=true;*.info=true" OPENCV_LOG_LEVEL=INFO GST_DEBUG=2 ./build/casa-anzen -m models/engine/yolo11n.engine --video 0
```

To force quiet mode manually (normally not needed):

```bash
QT_LOGGING_RULES="*.debug=false;*.info=false;qt.qpa.*=false" OPENCV_LOG_LEVEL=ERROR GST_DEBUG=0 ./build/casa-anzen -m models/engine/yolo11n.engine --video 0
```

### UI Guide

- **Drawing Zones**: Click "Draw Zone" in the side panel, then click to add vertices. Click near the first vertex to close the polygon. Enter a label before drawing to name the zone; otherwise, it is auto-named. Zones are saved to SQLite automatically.
- **Clear/Hide Zones**: Use "Clear Zones" to remove all zones (also persisted). Use "Hide/Show Zones" to toggle zone overlays without deleting them.
- **Event Feed**:
  - Shows thumbnails for manual captures and zone-entry triggers only.
  - Double-click a thumbnail to preview the full image.
  - Select a thumbnail, then click "CAPTION" to request an AI caption (manual, not automatic). Selection shows a thin green outline; no opaque overlays.
  - Delete/Delete All is safe and cancels any in-flight caption requests.

### Captioning (Moondream)

- Endpoint: `http://localhost:2020/v1/caption` (default in `CaptionManager`).
- Primary payload: `{ "image_url": "/absolute/path/to/image.jpg", "length": "short" }`.
- Fallback: automatic retry with `{ "image_base64": "<raw base64>" }` when needed.
- Captions are appended below the capture block in each event card.

Run a local Moondream server (example command may vary by your setup):

```bash
# Example only; use your local Moondream runner
docker run --rm -p 2020:2020 moondream/server:latest
```

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

- **MainCoordinator / UICoordinator**: High-level orchestration of managers and views.
- **VideoDisplayWidget**: Video, overlays, and polygon zone drawing.
- **EventFeedWidget**: List-based feed showing thumbnails and captions.
- **Managers**:
  - `EventManager`: Adds/removes events, coordinates with the feed, cancels in-flight caption requests on deletion.
  - `CaptionManager`: Handles caption requests, timeouts, and response parsing.
  - `VideoProcessingCoordinator`: Bridges processing thread with UI, emits FPS/detections/alerts.
  - `SystemStatusManager`: Updates custom status bar (status, FPS, detections, alerts, recording).

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
├── src/
│   ├── core/                         # Detection, tracking, recording, DB
│   │   ├── video_processing_thread.cpp
│   │   ├── yolo_detector.cpp
│   │   ├── tracker.cpp
│   │   └── recording_manager.cpp
│   ├── qt/                          # Qt application
│   │   ├── qt_main.cpp              # Entry point (quiet-by-default logging)
│   │   ├── security_dashboard.cpp   # Legacy monolithic UI (being replaced)
│   │   ├── components/              # Reusable widgets
│   │   │   └── event_card.{hpp,cpp}
│   │   ├── views/                   # Screens/sections
│   │   │   ├── event_feed_widget.{hpp,cpp}
│   │   │   ├── status_bar_widget.{hpp,cpp}
│   │   │   └── zone_controls_widget.{hpp,cpp}
│   │   └── managers/                # Coordinators and business logic
│   │       ├── caption_manager.{hpp,cpp}
│   │       ├── event_manager.{hpp,cpp}
│   │       ├── video_processing_coordinator.{hpp,cpp}
│   │       ├── main_coordinator.{hpp,cpp}
│   │       ├── ui_coordinator.{hpp,cpp}
│   │       └── system_status_manager.{hpp,cpp}
│   └── utils/                       # Utilities (logging, CUDA helpers)
│       ├── logger.cpp
│       └── cuda_*.cu / cuda_utils.hpp
├── include/                          # Public headers (mirrors src when applicable)
├── models/
│   ├── onnx/                         # ONNX models
│   └── engine/                       # TensorRT engines
├── data/
│   ├── casa_anzen.db                 # SQLite (WAL mode)
│   └── captures/                     # Saved thumbnails
├── build/                            # CMake build output (generated)
└── README.md
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
