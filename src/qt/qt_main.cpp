/**
 * @file qt_main.cpp
 * @brief Main entry point for Casa Anzen Security System Qt application
 * @author Anudit Gautam
 */

#include <QApplication>
#include <QLoggingCategory>
#include <QByteArray>
#include <cstdlib>
#include <opencv2/core/utils/logger.hpp>
#include <QStyleFactory>
#include <QPalette>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QDir>
#include <QDebug>
#include "managers/security_dashboard_adapter.hpp"

int main(int argc, char *argv[])
{
    // Silence verbose libraries by default
    QLoggingCategory::setFilterRules("*.debug=false\n*.info=false\nqt.qpa.*=false");
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    qputenv("GST_DEBUG", QByteArray("0"));

    QApplication app(argc, argv);
    // Apply modern dark theme using Fusion style
    QApplication::setStyle(QStyleFactory::create("Fusion"));
    QPalette darkPalette;
    darkPalette.setColor(QPalette::Window, QColor(37, 37, 38));
    darkPalette.setColor(QPalette::WindowText, Qt::white);
    darkPalette.setColor(QPalette::Base, QColor(30, 30, 30));
    darkPalette.setColor(QPalette::AlternateBase, QColor(45, 45, 48));
    darkPalette.setColor(QPalette::ToolTipBase, Qt::white);
    darkPalette.setColor(QPalette::ToolTipText, Qt::white);
    darkPalette.setColor(QPalette::Text, Qt::white);
    darkPalette.setColor(QPalette::Button, QColor(45, 45, 48));
    darkPalette.setColor(QPalette::ButtonText, Qt::white);
    darkPalette.setColor(QPalette::BrightText, Qt::red);
    darkPalette.setColor(QPalette::Highlight, QColor(10, 132, 255));
    darkPalette.setColor(QPalette::HighlightedText, Qt::white);
    darkPalette.setColor(QPalette::Disabled, QPalette::Text, QColor(127,127,127));
    darkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor(127,127,127));
    app.setPalette(darkPalette);
    app.setStyleSheet(
        "QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }\n"
        "QStatusBar{ background: #252526; }\n"
        "QLabel{ color: #dddddd; }\n"
        "QPushButton{ background-color:#3c3c3c; color:#ffffff; border:1px solid #555; padding:6px 10px; border-radius:4px;}\n"
        "QPushButton:hover{ background-color:#4a4a4a;}\n"
        "QLineEdit, QComboBox, QSpinBox{ background:#2d2d2d; border:1px solid #555; color:#ffffff; }\n"
    );
    app.setApplicationName("Casa Anzen Security System");
    app.setApplicationVersion("1.0");
    app.setOrganizationName("Casa Anzen");
    
    // Set up command line options
    QCommandLineParser parser;
    parser.setApplicationDescription("Casa Anzen Security System - AI-powered home security surveillance\n\n"
                                   "IMPORTANT: Run this command from the project root directory!\n\n"
                                   "Example usage:\n"
                                   "  ./build/casa-anzen -m models/engine/yolo11n.engine --video 0");
    parser.addHelpOption();
    parser.addVersionOption();
    
    QCommandLineOption modelOption(QStringList() << "m" << "model",
        "Path to the YOLO model file", "model.engine");
    parser.addOption(modelOption);
    
    QCommandLineOption videoOption(QStringList() << "video",
        "Video source (camera index or file path)", "0");
    parser.addOption(videoOption);
    
    QCommandLineOption confidenceOption(QStringList() << "confidence",
        "Detection confidence threshold", "0.25");
    parser.addOption(confidenceOption);
    
    QCommandLineOption recordingOption(QStringList() << "recording",
        "Enable automatic recording");
    parser.addOption(recordingOption);
    
    QCommandLineOption debugOption(QStringList() << "debug",
        "Enable debug mode");
    parser.addOption(debugOption);
    
    QCommandLineOption rtspLatencyOption(QStringList() << "rtsp-latency",
        "RTSP stream latency in milliseconds (higher = better quality, more delay)", "200");
    parser.addOption(rtspLatencyOption);
    
    QCommandLineOption rtspBufferOption(QStringList() << "rtsp-buffer",
        "RTSP buffer size (higher = more stable, more memory)", "10");
    parser.addOption(rtspBufferOption);
    
    QCommandLineOption rtspQualityOption(QStringList() << "rtsp-quality",
        "RTSP quality mode: low, medium, high, ultra", "high");
    parser.addOption(rtspQualityOption);
    
    parser.process(app);
    
    // Create and show the main window using new modular architecture
    SecurityDashboardAdapter window;
    window.show();
    
    // Set initial values from command line if provided
    if (parser.isSet(modelOption)) {
        QString modelPath = parser.value(modelOption);
        window.setModelPath(modelPath.toStdString());
    }
    
    if (parser.isSet(videoOption)) {
        QString videoSource = parser.value(videoOption);
        window.setVideoSource(videoSource.toStdString());
    }
    
    if (parser.isSet(confidenceOption)) {
        float confidence = parser.value(confidenceOption).toFloat();
        window.setConfidenceThreshold(confidence);
    }
    
    if (parser.isSet(recordingOption)) {
        window.enableRecording(true);
    }
    
    if (parser.isSet(debugOption)) {
        window.enableDebugMode(true);
    }
    
    if (parser.isSet(rtspLatencyOption)) {
        int latency = parser.value(rtspLatencyOption).toInt();
        window.setRtspLatency(latency);
    }
    
    if (parser.isSet(rtspBufferOption)) {
        int buffer_size = parser.value(rtspBufferOption).toInt();
        window.setRtspBufferSize(buffer_size);
    }
    
    if (parser.isSet(rtspQualityOption)) {
        QString quality = parser.value(rtspQualityOption);
        window.setRtspQuality(quality.toStdString());
    }
    
    // Auto-start if both model and video are provided
    window.autoStartIfConfigured();
    
    return app.exec();
}
