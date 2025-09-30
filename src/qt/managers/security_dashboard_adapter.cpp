#include "security_dashboard_adapter.hpp"
#include "ui_coordinator.hpp"
#include "menu_bar_manager.hpp"
#include "system_status_manager.hpp"
#include <QKeyEvent>
#include <QCloseEvent>
#include <QDebug>

SecurityDashboardAdapter::SecurityDashboardAdapter(QWidget* parent)
    : QMainWindow(parent)
    , m_uiCoordinator(nullptr)
    , m_initialized(false)
{
    setupAdapter();
}

SecurityDashboardAdapter::~SecurityDashboardAdapter()
{
    // UICoordinator will be automatically destroyed as unique_ptr
}

void SecurityDashboardAdapter::setupAdapter()
{
    // Create UI coordinator
    m_uiCoordinator = std::make_unique<UICoordinator>(this, this);
    
    // Initialize the coordinator
    m_uiCoordinator->initialize();
    
    // Connect signals
    connectSignals();
    
    // Apply legacy styling for compatibility
    applyLegacyStyling();
    
    m_initialized = true;
    qDebug() << "SecurityDashboardAdapter initialized with modular architecture";
}

void SecurityDashboardAdapter::connectSignals()
{
    if (!m_uiCoordinator) return;
    
    // Connect UI coordinator signals
    connect(m_uiCoordinator.get(), &UICoordinator::processingStarted,
            this, &SecurityDashboardAdapter::processingStarted);
    connect(m_uiCoordinator.get(), &UICoordinator::processingStopped,
            this, &SecurityDashboardAdapter::processingStopped);
    connect(m_uiCoordinator.get(), &UICoordinator::errorOccurred,
            this, &SecurityDashboardAdapter::handleProcessingError);
    connect(m_uiCoordinator.get(), &UICoordinator::configurationChanged,
            this, &SecurityDashboardAdapter::configurationChanged);
}

void SecurityDashboardAdapter::applyLegacyStyling()
{
    // Apply the same styling as the original SecurityDashboard
    setStyleSheet(
        "QMainWindow{ "
        "background: #0a0a0a; "
        "color: #00ff00; "
        "font-family: 'Courier New', monospace; "
        "}"
        "QDockWidget{ "
        "background: transparent; "
        "border: none; "
        "}"
        "QDockWidget::title{ "
        "background: transparent; "
        "}"
    );
}

void SecurityDashboardAdapter::setModelPath(const std::string& model_path)
{
    if (m_uiCoordinator) {
        m_uiCoordinator->setModelPath(model_path);
    }
}

void SecurityDashboardAdapter::setVideoSource(const std::string& video_source)
{
    if (m_uiCoordinator) {
        m_uiCoordinator->setVideoSource(video_source);
    }
}

void SecurityDashboardAdapter::setConfidenceThreshold(float threshold)
{
    if (m_uiCoordinator) {
        m_uiCoordinator->setConfidenceThreshold(threshold);
    }
}

void SecurityDashboardAdapter::enableRecording(bool enable)
{
    if (m_uiCoordinator) {
        m_uiCoordinator->enableRecording(enable);
    }
}

void SecurityDashboardAdapter::enableDebugMode(bool enable)
{
    if (m_uiCoordinator) {
        m_uiCoordinator->enableDebugMode(enable);
    }
}

void SecurityDashboardAdapter::setRtspLatency(int latency_ms)
{
    if (m_uiCoordinator) {
        m_uiCoordinator->setRtspLatency(latency_ms);
    }
}

void SecurityDashboardAdapter::setRtspBufferSize(int buffer_size)
{
    if (m_uiCoordinator) {
        m_uiCoordinator->setRtspBufferSize(buffer_size);
    }
}

void SecurityDashboardAdapter::setRtspQuality(const std::string& quality)
{
    if (m_uiCoordinator) {
        m_uiCoordinator->setRtspQuality(quality);
    }
}

void SecurityDashboardAdapter::autoStartIfConfigured()
{
    if (m_uiCoordinator) {
        m_uiCoordinator->autoStartIfConfigured();
    }
}

void SecurityDashboardAdapter::startProcessing()
{
    if (m_uiCoordinator) {
        m_uiCoordinator->startProcessing();
    }
}

void SecurityDashboardAdapter::stopProcessing()
{
    if (m_uiCoordinator) {
        m_uiCoordinator->stopProcessing();
    }
}

void SecurityDashboardAdapter::openVideoFile()
{
    if (m_uiCoordinator && m_uiCoordinator->getMenuBarManager()) {
        // Trigger the menu action
        m_uiCoordinator->getMenuBarManager()->onOpenVideoFile();
    }
}

void SecurityDashboardAdapter::openModelFile()
{
    if (m_uiCoordinator && m_uiCoordinator->getMenuBarManager()) {
        // Trigger the menu action
        m_uiCoordinator->getMenuBarManager()->onOpenModelFile();
    }
}

void SecurityDashboardAdapter::handleProcessingError(const QString& error_message)
{
    qDebug() << "Processing error:" << error_message;
    // Could show a message box or update status here
}

void SecurityDashboardAdapter::updateStatus()
{
    if (m_uiCoordinator && m_uiCoordinator->getSystemStatusManager()) {
        m_uiCoordinator->getSystemStatusManager()->updateSystemStatus();
    }
}

void SecurityDashboardAdapter::onNewFrame(const cv::Mat& frame)
{
    // This would be called by the video processing thread
    // For now, we'll just log it
    qDebug() << "New frame received:" << frame.rows << "x" << frame.cols;
}

void SecurityDashboardAdapter::updateDetectionData(const std::vector<casa_anzen::TrackedObject>& tracks,
                                                 const std::vector<casa_anzen::Detection>& detections)
{
    if (m_uiCoordinator && m_uiCoordinator->getSystemStatusManager()) {
        m_uiCoordinator->getSystemStatusManager()->updateDetectionData(tracks, detections);
    }
}

void SecurityDashboardAdapter::updateSecurityAlerts(const std::vector<casa_anzen::SecurityAlert>& alerts)
{
    if (m_uiCoordinator && m_uiCoordinator->getSystemStatusManager()) {
        m_uiCoordinator->getSystemStatusManager()->updateSecurityAlerts(alerts);
    }
}

void SecurityDashboardAdapter::toggleFullscreen()
{
    if (isFullScreen()) {
        showNormal();
    } else {
        showFullScreen();
    }
}

void SecurityDashboardAdapter::keyPressEvent(QKeyEvent* event)
{
    switch (event->key()) {
        case Qt::Key_F11:
            toggleFullscreen();
            break;
        case Qt::Key_Escape:
            if (isFullScreen()) {
                showNormal();
            }
            break;
        case Qt::Key_R:
            if (event->modifiers() & Qt::ControlModifier) {
                startProcessing();
            }
            break;
        case Qt::Key_S:
            if (event->modifiers() & Qt::ControlModifier) {
                stopProcessing();
            }
            break;
        default:
            QMainWindow::keyPressEvent(event);
            break;
    }
}

void SecurityDashboardAdapter::closeEvent(QCloseEvent* event)
{
    // Stop processing before closing
    if (m_uiCoordinator && m_uiCoordinator->isProcessing()) {
        m_uiCoordinator->stopProcessing();
    }
    
    event->accept();
}
