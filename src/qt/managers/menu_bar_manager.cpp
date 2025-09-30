#include "menu_bar_manager.hpp"
#include <QMenu>
#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>

MenuBarManager::MenuBarManager(QMainWindow* mainWindow, QObject* parent)
    : QObject(parent)
    , m_mainWindow(mainWindow)
    , m_menuBar(nullptr)
    , m_openVideoAction(nullptr)
    , m_openModelAction(nullptr)
    , m_exitAction(nullptr)
    , m_toggleFullscreenAction(nullptr)
    , m_startProcessingAction(nullptr)
    , m_stopProcessingAction(nullptr)
    , m_toggleRecordingAction(nullptr)
    , m_toggleDebugAction(nullptr)
    , m_modelPath("")
    , m_videoSource("")
    , m_recordingEnabled(false)
    , m_debugMode(false)
{
    setupMenuBar();
}

void MenuBarManager::setupMenuBar()
{
    if (!m_mainWindow) {
        qDebug() << "Main window not set for menu bar manager";
        return;
    }

    m_menuBar = m_mainWindow->menuBar();
    if (!m_menuBar) {
        qDebug() << "Menu bar not available";
        return;
    }

    createFileMenu();
    createViewMenu();
    createProcessingMenu();
    createSettingsMenu();
    createHelpMenu();
}

void MenuBarManager::createFileMenu()
{
    QMenu* fileMenu = m_menuBar->addMenu("&File");
    
    m_openVideoAction = new QAction("&Open Video...", this);
    m_openVideoAction->setShortcut(QKeySequence::Open);
    m_openVideoAction->setStatusTip("Open video file or stream");
    connect(m_openVideoAction, &QAction::triggered, this, &MenuBarManager::onOpenVideoFile);
    fileMenu->addAction(m_openVideoAction);
    
    m_openModelAction = new QAction("Open &Model...", this);
    m_openModelAction->setStatusTip("Open AI model file");
    connect(m_openModelAction, &QAction::triggered, this, &MenuBarManager::onOpenModelFile);
    fileMenu->addAction(m_openModelAction);
    
    fileMenu->addSeparator();
    
    m_exitAction = new QAction("E&xit", this);
    m_exitAction->setShortcut(QKeySequence::Quit);
    m_exitAction->setStatusTip("Exit the application");
    connect(m_exitAction, &QAction::triggered, m_mainWindow, &QMainWindow::close);
    fileMenu->addAction(m_exitAction);
}

void MenuBarManager::createViewMenu()
{
    QMenu* viewMenu = m_menuBar->addMenu("&View");
    
    m_toggleFullscreenAction = new QAction("Toggle &Fullscreen", this);
    m_toggleFullscreenAction->setShortcut(QKeySequence::FullScreen);
    m_toggleFullscreenAction->setStatusTip("Toggle fullscreen mode");
    connect(m_toggleFullscreenAction, &QAction::triggered, this, &MenuBarManager::onToggleFullscreen);
    viewMenu->addAction(m_toggleFullscreenAction);
}

void MenuBarManager::createProcessingMenu()
{
    QMenu* processingMenu = m_menuBar->addMenu("&Processing");
    
    m_startProcessingAction = new QAction("&Start Processing", this);
    m_startProcessingAction->setShortcut(QKeySequence("Ctrl+R"));
    m_startProcessingAction->setStatusTip("Start video processing");
    connect(m_startProcessingAction, &QAction::triggered, this, &MenuBarManager::onStartProcessing);
    processingMenu->addAction(m_startProcessingAction);
    
    m_stopProcessingAction = new QAction("&Stop Processing", this);
    m_stopProcessingAction->setShortcut(QKeySequence("Ctrl+S"));
    m_stopProcessingAction->setStatusTip("Stop video processing");
    connect(m_stopProcessingAction, &QAction::triggered, this, &MenuBarManager::onStopProcessing);
    processingMenu->addAction(m_stopProcessingAction);
}

void MenuBarManager::createSettingsMenu()
{
    QMenu* settingsMenu = m_menuBar->addMenu("&Settings");
    
    m_toggleRecordingAction = new QAction("Toggle &Recording", this);
    m_toggleRecordingAction->setCheckable(true);
    m_toggleRecordingAction->setChecked(m_recordingEnabled);
    m_toggleRecordingAction->setStatusTip("Toggle video recording");
    connect(m_toggleRecordingAction, &QAction::triggered, this, &MenuBarManager::onToggleRecording);
    settingsMenu->addAction(m_toggleRecordingAction);
    
    m_toggleDebugAction = new QAction("Toggle &Debug Mode", this);
    m_toggleDebugAction->setCheckable(true);
    m_toggleDebugAction->setChecked(m_debugMode);
    m_toggleDebugAction->setStatusTip("Toggle debug mode");
    connect(m_toggleDebugAction, &QAction::triggered, this, &MenuBarManager::onToggleDebugMode);
    settingsMenu->addAction(m_toggleDebugAction);
}

void MenuBarManager::createHelpMenu()
{
    QMenu* helpMenu = m_menuBar->addMenu("&Help");
    
    QAction* aboutAction = new QAction("&About", this);
    aboutAction->setStatusTip("About Casa Anzen");
    connect(aboutAction, &QAction::triggered, [this]() {
        QMessageBox::about(m_mainWindow, "About Casa Anzen", 
                          "Casa Anzen Security System\n"
                          "Advanced AI-powered security monitoring\n"
                          "Version 1.0");
    });
    helpMenu->addAction(aboutAction);
}

void MenuBarManager::setModelPath(const std::string& modelPath)
{
    m_modelPath = modelPath;
}

void MenuBarManager::setVideoSource(const std::string& videoSource)
{
    m_videoSource = videoSource;
}

void MenuBarManager::setRecordingEnabled(bool enabled)
{
    m_recordingEnabled = enabled;
    if (m_toggleRecordingAction) {
        m_toggleRecordingAction->setChecked(enabled);
    }
}

void MenuBarManager::setDebugMode(bool debugMode)
{
    m_debugMode = debugMode;
    if (m_toggleDebugAction) {
        m_toggleDebugAction->setChecked(debugMode);
    }
}

void MenuBarManager::onOpenVideoFile()
{
    QString fileName = QFileDialog::getOpenFileName(
        m_mainWindow,
        "Open Video File",
        "",
        "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
    );
    
    if (!fileName.isEmpty()) {
        m_videoSource = fileName.toStdString();
        emit openVideoFileRequested();
    }
}

void MenuBarManager::onOpenModelFile()
{
    QString fileName = QFileDialog::getOpenFileName(
        m_mainWindow,
        "Open Model File",
        "",
        "Model Files (*.engine *.onnx *.trt);;All Files (*)"
    );
    
    if (!fileName.isEmpty()) {
        m_modelPath = fileName.toStdString();
        emit openModelFileRequested();
    }
}

void MenuBarManager::onToggleFullscreen()
{
    emit toggleFullscreenRequested();
}

void MenuBarManager::onStartProcessing()
{
    emit startProcessingRequested();
}

void MenuBarManager::onStopProcessing()
{
    emit stopProcessingRequested();
}

void MenuBarManager::onToggleRecording()
{
    m_recordingEnabled = !m_recordingEnabled;
    emit recordingToggled(m_recordingEnabled);
}

void MenuBarManager::onToggleDebugMode()
{
    m_debugMode = !m_debugMode;
    emit debugModeToggled(m_debugMode);
}
