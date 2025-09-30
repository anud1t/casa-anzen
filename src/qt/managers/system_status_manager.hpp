#pragma once

#include <QObject>
#include <QTimer>
#include <QString>
#include <vector>
#include "core/types.hpp" // For casa_anzen types
#include <opencv2/opencv.hpp>

// Forward declarations
class StatusBarWidget;

// Include core types
#include "core/types.hpp"

class SystemStatusManager : public QObject
{
    Q_OBJECT

public:
    explicit SystemStatusManager(QObject* parent = nullptr);
    ~SystemStatusManager() = default;

    void setStatusBar(StatusBarWidget* statusBar);
    
    // Status updates
    void updateSystemStatus();
    void updateDetectionData(const std::vector<casa_anzen::TrackedObject>& tracks,
                           const std::vector<casa_anzen::Detection>& detections);
    void updateSecurityAlerts(const std::vector<casa_anzen::SecurityAlert>& alerts);
    void updateFrameCount();
    void updateFPS(float fps);
    
    // Status getters
    int getFrameCount() const;
    float getCurrentFPS() const;
    int getDetectionCount() const;
    int getAlertCount() const;
    QString getCurrentStatus() const;

signals:
    void statusChanged(const QString& status);
    void fpsChanged(float fps);
    void detectionCountChanged(int count);
    void alertCountChanged(int count);

private slots:
    void onUpdateTimer();

private:
    void setupTimers();
    void updateStatusLabels();
    void calculateFPS();

    StatusBarWidget* m_statusBar;
    QTimer* m_updateTimer;
    
    // Current data
    std::vector<casa_anzen::TrackedObject> m_currentTracks;
    std::vector<casa_anzen::Detection> m_currentDetections;
    std::vector<casa_anzen::SecurityAlert> m_currentAlerts;
    
    // Status counters
    int m_frameCount;
    float m_currentFPS;
    int m_detectionCount;
    int m_alertCount;
    QString m_currentStatus;
    
    // FPS calculation
    int m_fpsCounter;
    QTimer* m_fpsTimer;
};
