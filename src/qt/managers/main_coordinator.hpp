#pragma once

#include <QObject>
#include <QMainWindow>
#include <QString>
#include <QTimer>
#include <QListWidgetItem>
#include <opencv2/opencv.hpp>

// Forward declarations to avoid circular includes
class VideoDisplayWidget;
class EventFeedWidget;
class StatusBarWidget;
class ZoneControlsWidget;
class VideoProcessingCoordinator;
class EventManager;
class CaptionManager;

class MainCoordinator : public QObject
{
    Q_OBJECT

public:
    explicit MainCoordinator(QMainWindow* mainWindow, QObject* parent = nullptr);
    ~MainCoordinator() = default;

    void initialize();
    void setModelPath(const std::string& modelPath);
    void setVideoSource(const QString& source);
    
    void startProcessing();
    void stopProcessing();
    bool isProcessing() const;

    // Component getters for external access
    VideoDisplayWidget* getVideoDisplay() const;
    EventFeedWidget* getEventFeed() const;
    StatusBarWidget* getStatusBar() const;
    ZoneControlsWidget* getZoneControls() const;

signals:
    void processingStarted();
    void processingStopped();
    void errorOccurred(const QString& error);

private slots:
    void onVideoProcessingStarted();
    void onVideoProcessingStopped();
    void onFPSChanged(double fps);
    void onDetectionsChanged(int count);
    void onAlertsChanged(int count);
    void onNewFrame(const cv::Mat& frame);
    void onEventAdded(const QString& title);
    void onEventDeleted(const QString& title);
    void onCaptionReady(QListWidgetItem* item, const QString& caption);
    void onDrawZoneRequested();
    void onClearZonesRequested();
    void onHideZonesRequested();
    void onShowZonesRequested();
    void onZoneNameChanged(const QString& name);

private:
    void setupComponents();
    void setupConnections();
    void setupUI();
    void applyMilitaryTheme();

    QMainWindow* m_mainWindow;
    
    // UI Components
    VideoDisplayWidget* m_videoDisplay;
    EventFeedWidget* m_eventFeed;
    StatusBarWidget* m_statusBar;
    ZoneControlsWidget* m_zoneControls;
    
    // Managers and Coordinators
    VideoProcessingCoordinator* m_videoCoordinator;
    EventManager* m_eventManager;
    CaptionManager* m_captionManager;
    
    // State
    std::string m_modelPath;
    QString m_videoSource;
    bool m_isProcessing;
};
