#!/bin/bash

# Casa Anzen Security System Build Script
# Author: Casa Anzen Team

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Release"
CLEAN_BUILD=false
MAX_PERFORMANCE=false
VERBOSE=false
CONFIGURE_ONLY=false
BUILD_ONLY=false
JOBS="$(nproc)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --configure|--configure-only)
            CONFIGURE_ONLY=true
            BUILD_ONLY=false
            shift
            ;;
        --build|--build-only)
            BUILD_ONLY=true
            CONFIGURE_ONLY=false
            shift
            ;;
        --rebuild)
            CLEAN_BUILD=true
            CONFIGURE_ONLY=false
            BUILD_ONLY=false
            shift
            ;;
        --max-performance)
            MAX_PERFORMANCE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -j*)
            # Support -j4 form
            JOBS="${1#-j}"
            shift
            ;;
        --jobs=*)
            JOBS="${1#--jobs=}"
            shift
            ;;
        -h|--help)
            echo "Casa Anzen Security System Build Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -b, --build-type TYPE    Build type (Debug, Release, RelWithDebInfo)"
            echo "      --release            Shortcut for -b Release (default)"
            echo "      --debug              Shortcut for -b Debug"
            echo "  --clean                  Clean build directory before building"
            echo "  --configure-only         Run CMake configure step only"
            echo "  --build-only             Build existing configuration only"
            echo "  --rebuild                Clean, configure, and build"
            echo "  -j, --jobs N             Parallel build jobs (default: $(nproc)); also -jN or --jobs=N"
            echo "  --max-performance        Enable maximum performance optimizations"
            echo "  -v, --verbose            Enable verbose output"
            echo "  -h, --help               Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}Casa Anzen Security System Build Script${NC}"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo -e "${RED}Error: CMakeLists.txt not found. Please run this script from the project root directory.${NC}"
    exit 1
fi

# Create build directory
BUILD_DIR="build"
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure build
if [ "$BUILD_ONLY" = false ]; then
    echo -e "${BLUE}Configuring build...${NC}"
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    if [ "$MAX_PERFORMANCE" = true ]; then
        echo -e "${YELLOW}Enabling maximum performance optimizations...${NC}"
        CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG -march=native -mtune=native -ffast-math -funroll-loops -flto'"
    fi
    if [ "$VERBOSE" = true ]; then
        CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_VERBOSE_MAKEFILE=ON"
    fi
    cmake .. $CMAKE_ARGS
fi

if [ "$CONFIGURE_ONLY" = false ]; then
    # Build
    echo -e "${BLUE}Building Casa Anzen Security System...${NC}"
    if [ "$VERBOSE" = true ]; then
        make -j"$JOBS" VERBOSE=1
    else
        make -j"$JOBS"
    fi
fi

echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
echo "Executables:"
echo "  - casa-anzen: Main security dashboard application"
echo ""
echo ""
echo "To run the application:"
echo "  ./casa-anzen -m models/engine/yolo11n.engine --video 0"
echo ""
echo "To run with a video file:"
echo "  ./casa-anzen -m models/engine/yolo11n.engine --video path/to/video.mp4"
