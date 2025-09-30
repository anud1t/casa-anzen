#pragma once

#include <QObject>
#include <QMenuBar>
#include <QAction>
#include <QMainWindow>

class MenuBarManager : public QObject
{
    Q_OBJECT

public:
    explicit MenuBarManager(QMainWindow* mainWindow, QObject* parent = nullptr);
    ~MenuBarManager() = default;

    void setupMenuBar();
    void setModelPath(const std::string& modelPath);
    void setVideoSource(const std::string& videoSource);
    void setRecordingEnabled(bool enabled);
    void setDebugMode(bool debugMode);

signals:
    void openVideoFileRequested();
    void openModelFileRequested();
    void toggleFullscreenRequested();
    void startProcessingRequested();
    void stopProcessingRequested();
    void recordingToggled(bool enabled);
    void debugModeToggled(bool enabled);

public slots:
    void onOpenVideoFile();
    void onOpenModelFile();
    void onToggleFullscreen();
    void onStartProcessing();
    void onStopProcessing();
    void onToggleRecording();
    void onToggleDebugMode();

private:
    void createFileMenu();
    void createViewMenu();
    void createProcessingMenu();
    void createSettingsMenu();
    void createHelpMenu();

    QMainWindow* m_mainWindow;
    QMenuBar* m_menuBar;
    
    // File menu
    QAction* m_openVideoAction;
    QAction* m_openModelAction;
    QAction* m_exitAction;
    
    // View menu
    QAction* m_toggleFullscreenAction;
    
    // Processing menu
    QAction* m_startProcessingAction;
    QAction* m_stopProcessingAction;
    
    // Settings menu
    QAction* m_toggleRecordingAction;
    QAction* m_toggleDebugAction;
    
    // State
    std::string m_modelPath;
    std::string m_videoSource;
    bool m_recordingEnabled;
    bool m_debugMode;
};
