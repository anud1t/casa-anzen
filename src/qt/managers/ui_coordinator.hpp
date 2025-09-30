#pragma once

#include <QObject>
#include <QMainWindow>
#include <QString>
#include <memory>
#include <opencv2/opencv.hpp>
#include "core/types.hpp"

// Forward declarations
namespace casa_anzen {
    class VideoDisplayWidget;
}

class EventFeedWidget;
class StatusBarWidget;
class ZoneControlsWidget;
class MenuBarManager;
class ConfigurationManager;
class SystemStatusManager;
class EventManager;
class CaptionManager;
class VideoProcessingCoordinator;

class UICoordinator : public QObject
{
    Q_OBJECT

public:
    explicit UICoordinator(QMainWindow* mainWindow, QObject* parent = nullptr);
    ~UICoordinator() = default;

    void initialize();
    void setupUI();
    void applyMilitaryTheme();
    
    // Configuration delegation
    void setModelPath(const std::string& modelPath);
    void setVideoSource(const std::string& videoSource);
    void setConfidenceThreshold(float threshold);
    void enableRecording(bool enabled);
    void enableDebugMode(bool debugMode);
    void setRtspLatency(int latencyMs);
    void setRtspBufferSize(int bufferSize);
    void setRtspQuality(const std::string& quality);
    void autoStartIfConfigured();
    
    // Processing control
    void startProcessing();
    void stopProcessing();
    bool isProcessing() const;
    
    // Component getters
    casa_anzen::VideoDisplayWidget* getVideoDisplay() const;
    EventFeedWidget* getEventFeed() const;
    StatusBarWidget* getStatusBar() const;
    ZoneControlsWidget* getZoneControls() const;
    MenuBarManager* getMenuBarManager() const;
    ConfigurationManager* getConfigurationManager() const;
    SystemStatusManager* getSystemStatusManager() const;

signals:
    void processingStarted();
    void processingStopped();
    void errorOccurred(const QString& error);
    void configurationChanged();

private slots:
    void onProcessingStarted();
    void onProcessingStopped();
    void onErrorOccurred(const QString& error);
    void onConfigurationChanged();
    void onMenuActionTriggered(const QString& action);
    void onZoneCreated(const casa_anzen::SecurityZone& zone, const cv::Mat& frame);
    void onCaptureRequested(const QString& class_name, const cv::Rect& bbox, const cv::Mat& frame);

private:
    void setupComponents();
    void setupConnections();
    void setupLayout();
    void connectSignals();

    QMainWindow* m_mainWindow;
    
    // UI Components
    casa_anzen::VideoDisplayWidget* m_videoDisplay;
    EventFeedWidget* m_eventFeed;
    StatusBarWidget* m_statusBar;
    ZoneControlsWidget* m_zoneControls;
    
    // Managers
    MenuBarManager* m_menuBarManager;
    ConfigurationManager* m_configurationManager;
    SystemStatusManager* m_systemStatusManager;
    EventManager* m_eventManager;
    CaptionManager* m_captionManager;
    VideoProcessingCoordinator* m_videoCoordinator;
    
    // State
    bool m_initialized;
    bool m_processing;
    bool m_zonesVisible;
    std::vector<casa_anzen::SecurityZone> m_zones;
};
