#pragma once

#include <QObject>
#include <QTimer>
#include <QString>
#include <opencv2/opencv.hpp>

// Forward declarations to avoid circular includes
namespace casa_anzen {
    class VideoDisplayWidget;
}
class VideoProcessingThread;

class VideoProcessingCoordinator : public QObject
{
    Q_OBJECT

public:
    explicit VideoProcessingCoordinator(QObject* parent = nullptr);
    ~VideoProcessingCoordinator();

    void setVideoDisplay(casa_anzen::VideoDisplayWidget* display);
    void setModelPath(const std::string& modelPath);
    void setVideoSource(const QString& source);
    
    void startProcessing();
    void stopProcessing();
    bool isProcessing() const;

    void setFPS(double fps);
    void setDetections(int count);
    void setAlerts(int count);

signals:
    void processingStarted();
    void processingStopped();
    void fpsChanged(double fps);
    void detectionsChanged(int count);
    void alertsChanged(int count);
    void newFrame(const cv::Mat& frame);
    void errorOccurred(const QString& error);

private slots:
    void onProcessingFinished();
    void onFrameProcessed(const cv::Mat& frame);

private:
    void setupVideoProcessing();
    void cleanupVideoProcessing();

    casa_anzen::VideoDisplayWidget* m_videoDisplay;
    VideoProcessingThread* m_processingThread;
    QTimer* m_statusTimer;
    
    std::string m_modelPath;
    QString m_videoSource;
    bool m_isProcessing;
    
    double m_currentFPS;
    int m_currentDetections;
    int m_currentAlerts;
};
