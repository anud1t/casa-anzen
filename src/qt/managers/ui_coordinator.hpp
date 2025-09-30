#pragma once

#include <QObject>
#include <QMainWindow>
#include <QString>
#include <memory>

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
    std::unique_ptr<MenuBarManager> m_menuBarManager;
    std::unique_ptr<ConfigurationManager> m_configurationManager;
    std::unique_ptr<SystemStatusManager> m_systemStatusManager;
    std::unique_ptr<EventManager> m_eventManager;
    std::unique_ptr<CaptionManager> m_captionManager;
    std::unique_ptr<VideoProcessingCoordinator> m_videoCoordinator;
    
    // State
    bool m_initialized;
    bool m_processing;
};
