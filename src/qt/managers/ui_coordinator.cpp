#include "ui_coordinator.hpp"
#include <qt/video_display_widget.hpp>
#include "../views/event_feed_widget.hpp"
#include "../views/status_bar_widget.hpp"
#include "../views/zone_controls_widget.hpp"
#include "menu_bar_manager.hpp"
#include "configuration_manager.hpp"
#include "system_status_manager.hpp"
#include "event_manager.hpp"
#include "caption_manager.hpp"
#include "video_processing_coordinator.hpp"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QWidget>
#include <QDebug>

UICoordinator::UICoordinator(QMainWindow* mainWindow, QObject* parent)
    : QObject(parent)
    , m_mainWindow(mainWindow)
    , m_videoDisplay(nullptr)
    , m_eventFeed(nullptr)
    , m_statusBar(nullptr)
    , m_zoneControls(nullptr)
    , m_initialized(false)
    , m_processing(false)
{
    setupComponents();
    setupConnections();
}

void UICoordinator::initialize()
{
    if (m_initialized) {
        qDebug() << "UICoordinator already initialized";
        return;
    }
    
    setupUI();
    applyMilitaryTheme();
    
    // Initialize all managers
    if (m_configurationManager) {
        m_configurationManager->setModelPath("");
        m_configurationManager->setVideoSource("");
    }
    
    if (m_systemStatusManager) {
        m_systemStatusManager->updateSystemStatus();
    }
    
    m_initialized = true;
    qDebug() << "UICoordinator initialized successfully";
}

void UICoordinator::setupUI()
{
    if (!m_mainWindow) {
        qDebug() << "Main window not set for UI coordinator";
        return;
    }
    
    setupLayout();
    
    // Set status bar
    if (m_statusBar) {
        m_mainWindow->setStatusBar(m_statusBar);
    }
    
    // Setup menu bar
    if (m_menuBarManager) {
        m_menuBarManager->setupMenuBar();
    }
}

void UICoordinator::setupLayout()
{
    // Create central widget
    QWidget* centralWidget = new QWidget(m_mainWindow);
    m_mainWindow->setCentralWidget(centralWidget);

    // Create main layout
    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);
    mainLayout->setContentsMargins(4, 4, 4, 4);
    mainLayout->setSpacing(4);

    // Create splitter for video and side panel
    QSplitter* splitter = new QSplitter(Qt::Horizontal, centralWidget);
    
    // Add video display to splitter
    if (m_videoDisplay) {
        splitter->addWidget(m_videoDisplay);
        splitter->setStretchFactor(0, 1); // Video takes more space
    }

    // Create side panel
    QWidget* sidePanel = new QWidget();
    QVBoxLayout* sideLayout = new QVBoxLayout(sidePanel);
    sideLayout->setContentsMargins(8, 8, 8, 8);
    sideLayout->setSpacing(8);

    // Add components to side panel
    if (m_eventFeed) {
        sideLayout->addWidget(m_eventFeed);
    }
    if (m_zoneControls) {
        sideLayout->addWidget(m_zoneControls);
    }

    // Add side panel to splitter
    splitter->addWidget(sidePanel);
    splitter->setStretchFactor(1, 0); // Side panel takes less space

    // Add splitter to main layout
    mainLayout->addWidget(splitter);
}

void UICoordinator::applyMilitaryTheme()
{
    if (!m_mainWindow) return;

    m_mainWindow->setStyleSheet(
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

void UICoordinator::setupComponents()
{
    // Create managers
    m_menuBarManager = new MenuBarManager(m_mainWindow, this);
    m_configurationManager = new ConfigurationManager(this);
    m_systemStatusManager = new SystemStatusManager(this);
    m_eventManager = new EventManager(this);
    m_captionManager = new CaptionManager(this);
    m_videoCoordinator = new VideoProcessingCoordinator(this);
    
    // Create UI components
    m_videoDisplay = new casa_anzen::VideoDisplayWidget();
    m_eventFeed = new EventFeedWidget();
    m_statusBar = new StatusBarWidget();
    m_zoneControls = new ZoneControlsWidget();
    
    // Set up component relationships
    m_systemStatusManager->setStatusBar(m_statusBar);
    m_eventManager->setEventFeed(m_eventFeed);
    m_eventManager->setCaptionManager(m_captionManager);
    m_videoCoordinator->setVideoDisplay(m_videoDisplay);
}

void UICoordinator::setupConnections()
{
    // Configuration manager connections
    if (m_configurationManager) {
        connect(m_configurationManager, &ConfigurationManager::configurationChanged,
                this, &UICoordinator::onConfigurationChanged);
    }
    
    // Video coordinator connections
    if (m_videoCoordinator) {
        connect(m_videoCoordinator, &VideoProcessingCoordinator::processingStarted,
                this, &UICoordinator::onProcessingStarted);
        connect(m_videoCoordinator, &VideoProcessingCoordinator::processingStopped,
                this, &UICoordinator::onProcessingStopped);
        connect(m_videoCoordinator, &VideoProcessingCoordinator::errorOccurred,
                this, &UICoordinator::onErrorOccurred);
    }
    
    // Menu bar manager connections
    if (m_menuBarManager) {
        connect(m_menuBarManager, &MenuBarManager::startProcessingRequested,
                this, &UICoordinator::startProcessing);
        connect(m_menuBarManager, &MenuBarManager::stopProcessingRequested,
                this, &UICoordinator::stopProcessing);
        connect(m_menuBarManager, &MenuBarManager::recordingToggled,
                [this](bool enabled) { enableRecording(enabled); });
        connect(m_menuBarManager, &MenuBarManager::debugModeToggled,
                [this](bool enabled) { enableDebugMode(enabled); });
    }
    
    // Event feed connections
    if (m_eventFeed && m_eventManager) {
        connect(m_eventFeed, &EventFeedWidget::viewRequested,
                m_eventManager, &EventManager::viewEvent);
        connect(m_eventFeed, &EventFeedWidget::captionRequested,
                [this](QListWidgetItem* item) {
                    QString path = item->data(Qt::UserRole).toString();
                    m_eventManager->requestCaption(item, path);
                });
        connect(m_eventFeed, &EventFeedWidget::deleteRequested,
                m_eventManager, &EventManager::deleteEvent);
        connect(m_eventFeed, &EventFeedWidget::deleteAllRequested,
                m_eventManager, &EventManager::deleteAllEvents);
    }
    
    // Zone controls connections
    if (m_zoneControls) {
        connect(m_zoneControls, &ZoneControlsWidget::drawZoneRequested,
                [this]() { onMenuActionTriggered("drawZone"); });
        connect(m_zoneControls, &ZoneControlsWidget::clearZonesRequested,
                [this]() { onMenuActionTriggered("clearZones"); });
        connect(m_zoneControls, &ZoneControlsWidget::hideZonesRequested,
                [this]() { onMenuActionTriggered("hideZones"); });
        connect(m_zoneControls, &ZoneControlsWidget::showZonesRequested,
                [this]() { onMenuActionTriggered("showZones"); });
    }
}

void UICoordinator::setModelPath(const std::string& modelPath)
{
    if (m_configurationManager) {
        m_configurationManager->setModelPath(modelPath);
    }
    if (m_videoCoordinator) {
        m_videoCoordinator->setModelPath(modelPath);
    }
}

void UICoordinator::setVideoSource(const std::string& videoSource)
{
    if (m_configurationManager) {
        m_configurationManager->setVideoSource(videoSource);
    }
    if (m_videoCoordinator) {
        m_videoCoordinator->setVideoSource(QString::fromStdString(videoSource));
    }
}

void UICoordinator::setConfidenceThreshold(float threshold)
{
    if (m_configurationManager) {
        m_configurationManager->setConfidenceThreshold(threshold);
    }
}

void UICoordinator::enableRecording(bool enabled)
{
    if (m_configurationManager) {
        m_configurationManager->setRecordingEnabled(enabled);
    }
    if (m_menuBarManager) {
        m_menuBarManager->setRecordingEnabled(enabled);
    }
}

void UICoordinator::enableDebugMode(bool debugMode)
{
    if (m_configurationManager) {
        m_configurationManager->setDebugMode(debugMode);
    }
    if (m_menuBarManager) {
        m_menuBarManager->setDebugMode(debugMode);
    }
}

void UICoordinator::setRtspLatency(int latencyMs)
{
    if (m_configurationManager) {
        m_configurationManager->setRtspLatency(latencyMs);
    }
}

void UICoordinator::setRtspBufferSize(int bufferSize)
{
    if (m_configurationManager) {
        m_configurationManager->setRtspBufferSize(bufferSize);
    }
}

void UICoordinator::setRtspQuality(const std::string& quality)
{
    if (m_configurationManager) {
        m_configurationManager->setRtspQuality(quality);
    }
}

void UICoordinator::autoStartIfConfigured()
{
    if (m_configurationManager && m_configurationManager->isConfigurationValid()) {
        startProcessing();
    }
}

void UICoordinator::startProcessing()
{
    if (m_processing) {
        qDebug() << "Processing already running";
        return;
    }
    
    if (m_videoCoordinator) {
        m_videoCoordinator->startProcessing();
    }
}

void UICoordinator::stopProcessing()
{
    if (!m_processing) {
        return;
    }
    
    if (m_videoCoordinator) {
        m_videoCoordinator->stopProcessing();
    }
}

bool UICoordinator::isProcessing() const
{
    return m_processing;
}

casa_anzen::VideoDisplayWidget* UICoordinator::getVideoDisplay() const
{
    return m_videoDisplay;
}

EventFeedWidget* UICoordinator::getEventFeed() const
{
    return m_eventFeed;
}

StatusBarWidget* UICoordinator::getStatusBar() const
{
    return m_statusBar;
}

ZoneControlsWidget* UICoordinator::getZoneControls() const
{
    return m_zoneControls;
}

MenuBarManager* UICoordinator::getMenuBarManager() const
{
    return m_menuBarManager;
}

ConfigurationManager* UICoordinator::getConfigurationManager() const
{
    return m_configurationManager;
}

SystemStatusManager* UICoordinator::getSystemStatusManager() const
{
    return m_systemStatusManager;
}

void UICoordinator::onProcessingStarted()
{
    m_processing = true;
    emit processingStarted();
    qDebug() << "Processing started via UI coordinator";
}

void UICoordinator::onProcessingStopped()
{
    m_processing = false;
    emit processingStopped();
    qDebug() << "Processing stopped via UI coordinator";
}

void UICoordinator::onErrorOccurred(const QString& error)
{
    emit errorOccurred(error);
    qDebug() << "Error occurred:" << error;
}

void UICoordinator::onConfigurationChanged()
{
    emit configurationChanged();
    qDebug() << "Configuration changed via UI coordinator";
}

void UICoordinator::onMenuActionTriggered(const QString& action)
{
    qDebug() << "Menu action triggered:" << action;
    // Handle specific menu actions here
}
