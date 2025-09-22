cmake_minimum_required(VERSION 3.10)
project(CasaAnzen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==============================================================================
# PERFORMANCE OPTIMIZATION FLAGS
# ==============================================================================
# Aggressive optimization for maximum performance
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math -funroll-loops -flto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# CUDA optimization flags
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -arch=sm_87 -use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_87 -use_fast_math")

# Enable link-time optimization
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# Enable parallel compilation
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# ==============================================================================
# Find Dependencies
# ==============================================================================
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SQLite3 REQUIRED sqlite3)

# Add TensorRT include directories and libraries
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

# ==============================================================================
# Project Configuration
# ==============================================================================
# Add your main include directory. This is crucial.
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${SQLite3_INCLUDE_DIRS}
)

# Enable Qt's automatic meta-object compiler, UI compiler, and resource compiler
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ==============================================================================
# Core Security Library (casa-anzen-lib)
# ==============================================================================
set(CORE_LIB_SOURCES
    src/core/yolo_detector.cpp
    src/core/tracker.cpp
    src/core/security_detector.cpp
    src/core/zone_manager.cpp
    src/core/recording_manager.cpp
    src/core/database_manager.cpp
    src/core/detection_logger.cpp
    src/utils/logger.cpp
)

# CUDA sources for GPU-accelerated preprocessing and postprocessing
set(CUDA_SOURCES
    src/utils/cuda_preprocessing.cu
    src/utils/cuda_postprocessing.cu
)

add_library(casa-anzen-lib ${CORE_LIB_SOURCES} ${CUDA_SOURCES})

# Performance optimization for core library
target_compile_options(casa-anzen-lib PRIVATE 
    $<$<CXX_COMPILER_ID:GNU>:-O3 -march=native -mtune=native -ffast-math -funroll-loops>
    $<$<CXX_COMPILER_ID:Clang>:-O3 -march=native -ffast-math -funroll-loops>
)

# CUDA-specific compilation options for GPU acceleration
set_target_properties(casa-anzen-lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "87"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Find nlohmann/json
find_package(nlohmann_json 3.2.0 QUIET)
if (NOT nlohmann_json_FOUND)
    # Try to find it as a package
    find_package(PkgConfig QUIET)
    if (PkgConfig_FOUND)
        pkg_check_modules(NLOHMANN_JSON QUIET nlohmann_json)
    endif()
    
    if (NOT NLOHMANN_JSON_FOUND)
        # If not found, we'll need to install it or provide a fallback
        message(WARNING "nlohmann/json not found. Configuration management will not work.")
        add_definitions(-DNLOHMANN_JSON_DISABLED)
    endif()
endif()

# Link the core library against its dependencies
target_link_libraries(casa-anzen-lib
    PRIVATE
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${SQLite3_LIBRARIES}
    nvinfer
    nvinfer_plugin
    gomp  # OpenMP for parallel processing
)

# Add nlohmann/json if found
if (nlohmann_json_FOUND)
    target_link_libraries(casa-anzen-lib PRIVATE nlohmann_json::nlohmann_json)
elseif (NLOHMANN_JSON_FOUND)
    target_link_libraries(casa-anzen-lib PRIVATE ${NLOHMANN_JSON_LIBRARIES})
    target_include_directories(casa-anzen-lib PRIVATE ${NLOHMANN_JSON_INCLUDE_DIRS})
endif()

# ==============================================================================
# Qt5 Security Dashboard Executable (casa-anzen)
# ==============================================================================
# Define the source files for the Qt application
set(QT_APP_SOURCES
    src/qt/qt_main.cpp
    src/qt/security_dashboard.cpp
    src/qt/video_display_widget.cpp
    src/core/video_processing_thread.cpp
    
)

# Define the header files for the Qt application
set(QT_APP_HEADERS
    include/qt/security_dashboard.hpp
    include/qt/video_display_widget.hpp
    include/core/video_processing_thread.hpp
    
)

# Add the executable for the Qt application
add_executable(casa-anzen ${QT_APP_SOURCES} ${QT_APP_HEADERS})

# Performance optimization for Qt application
target_compile_options(casa-anzen PRIVATE 
    $<$<CXX_COMPILER_ID:GNU>:-O3 -march=native -mtune=native -ffast-math -funroll-loops>
    $<$<CXX_COMPILER_ID:Clang>:-O3 -march=native -ffast-math -funroll-loops>
)

# Link the Qt application against the core library and Qt5
target_link_libraries(casa-anzen
    PRIVATE
    casa-anzen-lib
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    nvinfer_plugin
    gomp  # OpenMP for parallel processing
    ${OpenCV_LIBS}
)

## Removed unused CLI target and unimplemented dialogs
