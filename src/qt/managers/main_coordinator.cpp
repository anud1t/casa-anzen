#include "main_coordinator.hpp"
#include <qt/video_display_widget.hpp>
#include "../views/event_feed_widget.hpp"
#include "../views/status_bar_widget.hpp"
#include "../views/zone_controls_widget.hpp"
#include "video_processing_coordinator.hpp"
#include "event_manager.hpp"
#include "caption_manager.hpp"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QDockWidget>
#include <QListWidgetItem>
#include <QDebug>
#include <QWidget>

MainCoordinator::MainCoordinator(QMainWindow* mainWindow, QObject* parent)
    : QObject(parent)
    , m_mainWindow(mainWindow)
    , m_videoDisplay(nullptr)
    , m_eventFeed(nullptr)
    , m_statusBar(nullptr)
    , m_zoneControls(nullptr)
    , m_videoCoordinator(nullptr)
    , m_eventManager(nullptr)
    , m_captionManager(nullptr)
    , m_modelPath("")
    , m_videoSource("")
    , m_isProcessing(false)
{
    setupComponents();
    setupConnections();
}

void MainCoordinator::initialize()
{
    setupUI();
    applyMilitaryTheme();
    
    // Initialize status
    if (m_statusBar) {
        m_statusBar->setStatus("READY");
        m_statusBar->setMode("MODE: PEOPLE + VEHICLES");
        m_statusBar->setFPS(0.0);
        m_statusBar->setDetections(0);
        m_statusBar->setAlerts(0);
        m_statusBar->setRecording(false);
    }
    
    // reduced verbosity
}

void MainCoordinator::setModelPath(const std::string& modelPath)
{
    m_modelPath = modelPath;
    if (m_videoCoordinator) {
        m_videoCoordinator->setModelPath(modelPath);
    }
}

void MainCoordinator::setVideoSource(const QString& source)
{
    m_videoSource = source;
    if (m_videoCoordinator) {
        m_videoCoordinator->setVideoSource(source);
    }
}

void MainCoordinator::startProcessing()
{
    if (m_isProcessing) {
        return;
    }

    if (m_videoCoordinator) {
        m_videoCoordinator->startProcessing();
    }
}

void MainCoordinator::stopProcessing()
{
    if (!m_isProcessing) {
        return;
    }

    if (m_videoCoordinator) {
        m_videoCoordinator->stopProcessing();
    }
}

bool MainCoordinator::isProcessing() const
{
    return m_isProcessing;
}

casa_anzen::VideoDisplayWidget* MainCoordinator::getVideoDisplay() const
{
    return m_videoDisplay;
}

EventFeedWidget* MainCoordinator::getEventFeed() const
{
    return m_eventFeed;
}

StatusBarWidget* MainCoordinator::getStatusBar() const
{
    return m_statusBar;
}

ZoneControlsWidget* MainCoordinator::getZoneControls() const
{
    return m_zoneControls;
}

void MainCoordinator::setupComponents()
{
    // Create managers and coordinators
    m_videoCoordinator = new VideoProcessingCoordinator(this);
    m_eventManager = new EventManager(this);
    m_captionManager = new CaptionManager(this);
    
    // Create UI components
    m_videoDisplay = new casa_anzen::VideoDisplayWidget();
    m_eventFeed = new EventFeedWidget();
    m_statusBar = new StatusBarWidget();
    m_zoneControls = new ZoneControlsWidget();
    
    // Set up component relationships
    m_videoCoordinator->setVideoDisplay(m_videoDisplay);
    m_eventManager->setEventFeed(m_eventFeed);
    m_eventManager->setCaptionManager(m_captionManager);
}

void MainCoordinator::setupConnections()
{
    // Video processing connections
    connect(m_videoCoordinator, &VideoProcessingCoordinator::processingStarted,
            this, &MainCoordinator::onVideoProcessingStarted);
    connect(m_videoCoordinator, &VideoProcessingCoordinator::processingStopped,
            this, &MainCoordinator::onVideoProcessingStopped);
    connect(m_videoCoordinator, &VideoProcessingCoordinator::fpsChanged,
            this, &MainCoordinator::onFPSChanged);
    connect(m_videoCoordinator, &VideoProcessingCoordinator::detectionsChanged,
            this, &MainCoordinator::onDetectionsChanged);
    connect(m_videoCoordinator, &VideoProcessingCoordinator::alertsChanged,
            this, &MainCoordinator::onAlertsChanged);
    connect(m_videoCoordinator, &VideoProcessingCoordinator::newFrame,
            this, &MainCoordinator::onNewFrame);
    connect(m_videoCoordinator, &VideoProcessingCoordinator::errorOccurred,
            this, &MainCoordinator::errorOccurred);

    // Event management connections
    connect(m_eventManager, &EventManager::eventAdded,
            this, &MainCoordinator::onEventAdded);
    connect(m_eventManager, &EventManager::eventDeleted,
            this, &MainCoordinator::onEventDeleted);
    connect(m_eventManager, &EventManager::captionReady,
            this, &MainCoordinator::onCaptionReady);

    // Event feed connections
    connect(m_eventFeed, &EventFeedWidget::viewRequested,
            m_eventManager, &EventManager::viewEvent);
    connect(m_eventFeed, &EventFeedWidget::captionRequested,
            [this](QListWidgetItem* item) {
                // Get image path from item data
                QString path = item->data(Qt::UserRole).toString();
                m_eventManager->requestCaption(item, path);
            });
    connect(m_eventFeed, &EventFeedWidget::deleteRequested,
            m_eventManager, &EventManager::deleteEvent);
    connect(m_eventFeed, &EventFeedWidget::deleteAllRequested,
            m_eventManager, &EventManager::deleteAllEvents);

    // Zone controls connections
    connect(m_zoneControls, &ZoneControlsWidget::drawZoneRequested,
            this, &MainCoordinator::onDrawZoneRequested);
    connect(m_zoneControls, &ZoneControlsWidget::clearZonesRequested,
            this, &MainCoordinator::onClearZonesRequested);
    connect(m_zoneControls, &ZoneControlsWidget::hideZonesRequested,
            this, &MainCoordinator::onHideZonesRequested);
    connect(m_zoneControls, &ZoneControlsWidget::showZonesRequested,
            this, &MainCoordinator::onShowZonesRequested);
    connect(m_zoneControls, &ZoneControlsWidget::zoneNameChanged,
            this, &MainCoordinator::onZoneNameChanged);
}

void MainCoordinator::setupUI()
{
    if (!m_mainWindow) {
        qDebug() << "Main window not set";
        return;
    }

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
    sideLayout->addWidget(m_eventFeed);
    sideLayout->addWidget(m_zoneControls);

    // Add side panel to splitter
    splitter->addWidget(sidePanel);
    splitter->setStretchFactor(1, 0); // Side panel takes less space

    // Add splitter to main layout
    mainLayout->addWidget(splitter);

    // Set status bar
    m_mainWindow->setStatusBar(m_statusBar);
}

void MainCoordinator::applyMilitaryTheme()
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

void MainCoordinator::onVideoProcessingStarted()
{
    m_isProcessing = true;
    if (m_statusBar) {
        m_statusBar->setStatus("PROCESSING");
    }
    emit processingStarted();
    // reduced verbosity
}

void MainCoordinator::onVideoProcessingStopped()
{
    m_isProcessing = false;
    if (m_statusBar) {
        m_statusBar->setStatus("STOPPED");
    }
    emit processingStopped();
    // reduced verbosity
}

void MainCoordinator::onFPSChanged(double fps)
{
    if (m_statusBar) {
        m_statusBar->setFPS(fps);
    }
}

void MainCoordinator::onDetectionsChanged(int count)
{
    if (m_statusBar) {
        m_statusBar->setDetections(count);
    }
}

void MainCoordinator::onAlertsChanged(int count)
{
    if (m_statusBar) {
        m_statusBar->setAlerts(count);
    }
}

void MainCoordinator::onNewFrame(const cv::Mat& frame)
{
    // Handle new frame if needed
    Q_UNUSED(frame)
}

void MainCoordinator::onEventAdded(const QString& title)
{
    // reduced verbosity
}

void MainCoordinator::onEventDeleted(const QString& title)
{
    // reduced verbosity
}

void MainCoordinator::onCaptionReady(QListWidgetItem* item, const QString& caption)
{
    Q_UNUSED(item)
    Q_UNUSED(caption)
    // reduced verbosity
}

void MainCoordinator::onDrawZoneRequested()
{
    // reduced verbosity
    // Implement zone drawing logic
}

void MainCoordinator::onClearZonesRequested()
{
    // reduced verbosity
    // Implement zone clearing logic
}

void MainCoordinator::onHideZonesRequested()
{
    // reduced verbosity
    // Implement zone hiding logic
}

void MainCoordinator::onShowZonesRequested()
{
    // reduced verbosity
    // Implement zone showing logic
}

void MainCoordinator::onZoneNameChanged(const QString& name)
{
    Q_UNUSED(name)
}
