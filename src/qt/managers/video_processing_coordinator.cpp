#include "video_processing_coordinator.hpp"
#include <qt/video_display_widget.hpp>
#include <core/video_processing_thread.hpp>
#include <QDebug>
#include <QThread>

VideoProcessingCoordinator::VideoProcessingCoordinator(QObject* parent)
    : QObject(parent)
    , m_videoDisplay(nullptr)
    , m_processingThread(nullptr)
    , m_statusTimer(nullptr)
    , m_modelPath("")
    , m_videoSource("")
    , m_isProcessing(false)
    , m_currentFPS(0.0)
    , m_currentDetections(0)
    , m_currentAlerts(0)
{
    setupVideoProcessing();
}

VideoProcessingCoordinator::~VideoProcessingCoordinator()
{
    cleanupVideoProcessing();
}

void VideoProcessingCoordinator::setVideoDisplay(casa_anzen::VideoDisplayWidget* display)
{
    m_videoDisplay = display;
    if (m_processingThread) {
        connect(m_processingThread, &casa_anzen::VideoProcessingThread::newFrame,
                m_videoDisplay, &casa_anzen::VideoDisplayWidget::updateFrame);
    }
}

void VideoProcessingCoordinator::setModelPath(const std::string& modelPath)
{
    m_modelPath = modelPath;
    if (m_processingThread) {
        m_processingThread->setModelPath(modelPath);
    }
}

void VideoProcessingCoordinator::setVideoSource(const QString& source)
{
    m_videoSource = source;
    if (m_processingThread) {
        m_processingThread->setVideoSource(source.toStdString());
    }
}

void VideoProcessingCoordinator::startProcessing()
{
    if (m_isProcessing) {
        qDebug() << "Video processing already running";
        return;
    }

    if (!m_processingThread) {
        qDebug() << "Video processing thread not initialized";
        return;
    }

    if (m_modelPath.empty()) {
        emit errorOccurred("Model path not set");
        return;
    }

    if (m_videoSource.isEmpty()) {
        emit errorOccurred("Video source not set");
        return;
    }

    m_processingThread->start();
    m_isProcessing = true;
    emit processingStarted();
    
    qDebug() << "Video processing started with model:" << QString::fromStdString(m_modelPath)
             << "and source:" << m_videoSource;
}

void VideoProcessingCoordinator::stopProcessing()
{
    if (!m_isProcessing) {
        return;
    }

        if (m_processingThread) {
            m_processingThread->stop();
            m_processingThread->wait(5000); // Wait up to 5 seconds
        }

    m_isProcessing = false;
    emit processingStopped();
    
    qDebug() << "Video processing stopped";
}

bool VideoProcessingCoordinator::isProcessing() const
{
    return m_isProcessing;
}

void VideoProcessingCoordinator::setFPS(double fps)
{
    if (m_currentFPS != fps) {
        m_currentFPS = fps;
        emit fpsChanged(fps);
    }
}

void VideoProcessingCoordinator::setDetections(int count)
{
    if (m_currentDetections != count) {
        m_currentDetections = count;
        emit detectionsChanged(count);
    }
}

void VideoProcessingCoordinator::setAlerts(int count)
{
    if (m_currentAlerts != count) {
        m_currentAlerts = count;
        emit alertsChanged(count);
    }
}

void VideoProcessingCoordinator::setupVideoProcessing()
{
    m_processingThread = new casa_anzen::VideoProcessingThread(this);
    
    // Connect thread signals
    connect(m_processingThread, &casa_anzen::VideoProcessingThread::newFrame,
            this, &VideoProcessingCoordinator::onFrameProcessed);
    connect(m_processingThread, &casa_anzen::VideoProcessingThread::detectionData,
            this, &VideoProcessingCoordinator::onDetectionData);
    connect(m_processingThread, &casa_anzen::VideoProcessingThread::securityAlerts,
            this, &VideoProcessingCoordinator::onSecurityAlerts);
    connect(m_processingThread, &casa_anzen::VideoProcessingThread::processingFinished,
            this, &VideoProcessingCoordinator::onProcessingFinished);
    connect(m_processingThread, &casa_anzen::VideoProcessingThread::processingError,
            this, &VideoProcessingCoordinator::errorOccurred);

    // Setup status timer for periodic updates
    m_statusTimer = new QTimer(this);
    m_statusTimer->setInterval(1000); // Update every second
    connect(m_statusTimer, &QTimer::timeout, this, [this]() {
        // Update status information
        if (m_processingThread) {
            setFPS(m_processingThread->getCurrentFPS());
            // TODO: Add detection count tracking when available
            // setDetections(m_processingThread->getDetectionCount());
        }
    });
    m_statusTimer->start(); // Start the timer
}

void VideoProcessingCoordinator::cleanupVideoProcessing()
{
    stopProcessing();
    
    if (m_processingThread) {
        m_processingThread->deleteLater();
        m_processingThread = nullptr;
    }
    
    if (m_statusTimer) {
        m_statusTimer->stop();
        m_statusTimer->deleteLater();
        m_statusTimer = nullptr;
    }
}

void VideoProcessingCoordinator::onProcessingFinished()
{
    m_isProcessing = false;
    emit processingStopped();
    qDebug() << "Video processing thread finished";
}

void VideoProcessingCoordinator::onFrameProcessed(const cv::Mat& frame)
{
    emit newFrame(frame);
    
    // Update video display if available
    if (m_videoDisplay) {
        m_videoDisplay->updateFrame(frame);
    }
}

void VideoProcessingCoordinator::onDetectionData(const std::vector<casa_anzen::TrackedObject>& tracks,
                                                const std::vector<casa_anzen::Detection>& detections)
{
    // Update video display with detection overlays
    if (m_videoDisplay) {
        m_videoDisplay->setDetectionOverlays(tracks, detections);
    }
    
    // Update detection count
    setDetections(static_cast<int>(detections.size()));
    
        // Quiet by default: no per-frame detection logging
}

void VideoProcessingCoordinator::onSecurityAlerts(const std::vector<casa_anzen::SecurityAlert>& alerts)
{
    // Update video display with alert overlays
    if (m_videoDisplay) {
        m_videoDisplay->setAlertOverlays(alerts);
    }
    
    // Update alert count
    setAlerts(static_cast<int>(alerts.size()));
    
        // Quiet by default: suppress alert logs in terminal
}
