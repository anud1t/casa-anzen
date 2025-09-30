#include "system_status_manager.hpp"
#include "../views/status_bar_widget.hpp"
#include <QDebug>

SystemStatusManager::SystemStatusManager(QObject* parent)
    : QObject(parent)
    , m_statusBar(nullptr)
    , m_updateTimer(nullptr)
    , m_frameCount(0)
    , m_currentFPS(0.0f)
    , m_detectionCount(0)
    , m_alertCount(0)
    , m_isRecording(false)
    , m_currentStatus("READY")
    , m_fpsCounter(0)
    , m_fpsTimer(nullptr)
{
    setupTimers();
}

void SystemStatusManager::setStatusBar(StatusBarWidget* statusBar)
{
    m_statusBar = statusBar;
    updateStatusLabels();
}

void SystemStatusManager::updateSystemStatus()
{
    updateStatusLabels();
    emit statusChanged(m_currentStatus);
}

void SystemStatusManager::updateDetectionData(const std::vector<casa_anzen::TrackedObject>& tracks,
                                            const std::vector<casa_anzen::Detection>& detections)
{
    m_currentTracks = tracks;
    m_currentDetections = detections;
    
    int newDetectionCount = static_cast<int>(detections.size());
    if (m_detectionCount != newDetectionCount) {
        m_detectionCount = newDetectionCount;
        emit detectionCountChanged(m_detectionCount);
    }
    
    updateStatusLabels();
}

void SystemStatusManager::updateSecurityAlerts(const std::vector<casa_anzen::SecurityAlert>& alerts)
{
    m_currentAlerts = alerts;
    
    int newAlertCount = static_cast<int>(alerts.size());
    if (m_alertCount != newAlertCount) {
        m_alertCount = newAlertCount;
        emit alertCountChanged(m_alertCount);
    }
    
    updateStatusLabels();
}

void SystemStatusManager::updateFrameCount()
{
    m_frameCount++;
    m_fpsCounter++;
}

void SystemStatusManager::updateFPS(float fps)
{
    if (m_currentFPS != fps) {
        m_currentFPS = fps;
        emit fpsChanged(fps);
        updateStatusLabels();
    }
}

void SystemStatusManager::setFPS(float fps)
{
    if (m_currentFPS != fps) {
        m_currentFPS = fps;
        if (m_statusBar) m_statusBar->setFPS(fps);
        emit fpsChanged(fps);
    }
}

void SystemStatusManager::setDetections(int count)
{
    if (m_detectionCount != count) {
        m_detectionCount = count;
        if (m_statusBar) m_statusBar->setDetections(count);
        emit detectionCountChanged(count);
    }
}

void SystemStatusManager::setAlerts(int count)
{
    if (m_alertCount != count) {
        m_alertCount = count;
        if (m_statusBar) m_statusBar->setAlerts(count);
        emit alertCountChanged(count);
    }
}

void SystemStatusManager::setStatusMessage(const QString& message)
{
    if (m_currentStatus != message) {
        m_currentStatus = message;
        if (m_statusBar) m_statusBar->setStatus(message);
        emit statusChanged(message);
    }
}

void SystemStatusManager::setRecordingStatus(bool recording)
{
    if (m_isRecording != recording) {
        m_isRecording = recording;
        if (m_statusBar) m_statusBar->setRecording(recording);
    }
}

int SystemStatusManager::getFrameCount() const
{
    return m_frameCount;
}

float SystemStatusManager::getCurrentFPS() const
{
    return m_currentFPS;
}

int SystemStatusManager::getDetectionCount() const
{
    return m_detectionCount;
}

int SystemStatusManager::getAlertCount() const
{
    return m_alertCount;
}

QString SystemStatusManager::getCurrentStatus() const
{
    return m_currentStatus;
}

void SystemStatusManager::onUpdateTimer()
{
    updateSystemStatus();
}

void SystemStatusManager::updateStatusLabels()
{
    if (!m_statusBar) {
        return;
    }
    
    // Update status bar with current information
    m_statusBar->setStatus(m_currentStatus);
    m_statusBar->setFPS(m_currentFPS);
    m_statusBar->setDetections(m_detectionCount);
    m_statusBar->setAlerts(m_alertCount);
    
    // Update mode based on current detection type
    QString mode = "MODE: PEOPLE + VEHICLES"; // Default mode
    m_statusBar->setMode(mode);
}

void SystemStatusManager::calculateFPS()
{
    // This would be called periodically to calculate FPS
    // For now, we'll rely on external FPS updates
}

void SystemStatusManager::setupTimers()
{
    // Setup update timer for periodic status updates
    m_updateTimer = new QTimer(this);
    m_updateTimer->setInterval(1000); // Update every second
    connect(m_updateTimer, &QTimer::timeout, this, &SystemStatusManager::onUpdateTimer);
    m_updateTimer->start();
    
    // Setup FPS calculation timer
    m_fpsTimer = new QTimer(this);
    m_fpsTimer->setInterval(1000); // Calculate FPS every second
    connect(m_fpsTimer, &QTimer::timeout, this, &SystemStatusManager::calculateFPS);
    m_fpsTimer->start();
}
