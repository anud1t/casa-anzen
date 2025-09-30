#pragma once

#include <QObject>
#include <QMainWindow>
#include <QString>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

// Forward declarations
class UICoordinator;

// Include core types
#include "core/types.hpp"

/**
 * @brief Adapter class to integrate new modular architecture with existing SecurityDashboard
 * 
 * This class acts as a bridge between the old SecurityDashboard interface and the new
 * modular architecture, allowing for gradual migration while maintaining compatibility.
 */
class SecurityDashboardAdapter : public QMainWindow
{
    Q_OBJECT

public:
    explicit SecurityDashboardAdapter(QWidget* parent = nullptr);
    ~SecurityDashboardAdapter();

    // Configuration methods (matching SecurityDashboard interface)
    void setModelPath(const std::string& model_path);
    void setVideoSource(const std::string& video_source);
    void setConfidenceThreshold(float threshold);
    void enableRecording(bool enable);
    void enableDebugMode(bool enable);
    void setRtspLatency(int latency_ms);
    void setRtspBufferSize(int buffer_size);
    void setRtspQuality(const std::string& quality);
    void autoStartIfConfigured();

signals:
    void processingStarted();
    void processingStopped();
    void configurationChanged();

public slots:
    void startProcessing();
    void stopProcessing();
    void openVideoFile();
    void openModelFile();
    void handleProcessingError(const QString& error_message);
    void updateStatus();
    void onNewFrame(const cv::Mat& frame);
    void updateDetectionData(const std::vector<casa_anzen::TrackedObject>& tracks,
                           const std::vector<casa_anzen::Detection>& detections);
    void updateSecurityAlerts(const std::vector<casa_anzen::SecurityAlert>& alerts);
    void toggleFullscreen();

protected:
    void keyPressEvent(QKeyEvent* event) override;
    void closeEvent(QCloseEvent* event) override;

private:
    void setupAdapter();
    void connectSignals();
    void applyLegacyStyling();

    std::unique_ptr<UICoordinator> m_uiCoordinator;
    bool m_initialized;
};
