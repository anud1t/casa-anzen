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

void VideoProcessingCoordinator::setVideoDisplay(VideoDisplayWidget* display)
{
    m_videoDisplay = display;
    // if (m_processingThread) {
    //     connect(m_processingThread, &VideoProcessingThread::frameProcessed,
    //             m_videoDisplay, &VideoDisplayWidget::onNewFrame);
    // }
}

void VideoProcessingCoordinator::setModelPath(const std::string& modelPath)
{
    m_modelPath = modelPath;
    // if (m_processingThread) {
    //     m_processingThread->setModelPath(modelPath);
    // }
}

void VideoProcessingCoordinator::setVideoSource(const QString& source)
{
    m_videoSource = source;
    // if (m_processingThread) {
    //     m_processingThread->setVideoSource(source.toStdString());
    // }
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

    // m_processingThread->start();
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

    // if (m_processingThread) {
    //     m_processingThread->stop();
    //     m_processingThread->wait(5000); // Wait up to 5 seconds
    // }

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
    m_processingThread = nullptr; // new VideoProcessingThread(this);
    
    // Connect thread signals - commented out for now
    // connect(m_processingThread, &VideoProcessingThread::frameProcessed,
    //         this, &VideoProcessingCoordinator::onFrameProcessed);
    // connect(m_processingThread, &VideoProcessingThread::finished,
    //         this, &VideoProcessingCoordinator::onProcessingFinished);
    // connect(m_processingThread, &VideoProcessingThread::errorOccurred,
    //         this, &VideoProcessingCoordinator::errorOccurred);

    // Setup status timer for periodic updates
    m_statusTimer = new QTimer(this);
    m_statusTimer->setInterval(1000); // Update every second
    connect(m_statusTimer, &QTimer::timeout, this, [this]() {
        // Update status information - commented out for now
        // if (m_processingThread) {
        //     setFPS(m_processingThread->getCurrentFPS());
        //     setDetections(m_processingThread->getDetectionCount());
        // }
    });
}

void VideoProcessingCoordinator::cleanupVideoProcessing()
{
    stopProcessing();
    
    // if (m_processingThread) {
    //     m_processingThread->deleteLater();
    //     m_processingThread = nullptr;
    // }
    
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
    
    // Update video display if available - commented out for now
    // if (m_videoDisplay) {
    //     m_videoDisplay->onNewFrame(frame);
    // }
}
