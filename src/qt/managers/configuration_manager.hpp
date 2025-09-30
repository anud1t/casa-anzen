#pragma once

#include <QObject>
#include <QString>
#include <string>

class ConfigurationManager : public QObject
{
    Q_OBJECT

public:
    explicit ConfigurationManager(QObject* parent = nullptr);
    ~ConfigurationManager() = default;

    // Model configuration
    void setModelPath(const std::string& modelPath);
    std::string getModelPath() const;
    
    // Video configuration
    void setVideoSource(const std::string& videoSource);
    std::string getVideoSource() const;
    
    // Detection configuration
    void setConfidenceThreshold(float threshold);
    float getConfidenceThreshold() const;
    
    // Recording configuration
    void setRecordingEnabled(bool enabled);
    bool isRecordingEnabled() const;
    
    // Debug configuration
    void setDebugMode(bool debugMode);
    bool isDebugMode() const;
    
    // RTSP configuration
    void setRtspLatency(int latencyMs);
    int getRtspLatency() const;
    
    void setRtspBufferSize(int bufferSize);
    int getRtspBufferSize() const;
    
    void setRtspQuality(const std::string& quality);
    std::string getRtspQuality() const;
    
    // Detection mode
    void setDetectionMode(int mode);
    int getDetectionMode() const;
    
    // Validation
    bool isModelPathValid() const;
    bool isVideoSourceValid() const;
    bool isConfigurationValid() const;

signals:
    void configurationChanged();
    void modelPathChanged(const std::string& path);
    void videoSourceChanged(const std::string& source);
    void confidenceThresholdChanged(float threshold);
    void recordingEnabledChanged(bool enabled);
    void debugModeChanged(bool debugMode);
    void rtspSettingsChanged();

private:
    void validateConfiguration();

    // Model configuration
    std::string m_modelPath;
    
    // Video configuration
    std::string m_videoSource;
    
    // Detection configuration
    float m_confidenceThreshold;
    int m_detectionMode; // 0=PEOPLE,1=VEHICLES,2=PEOPLE+VEHICLES,3=ALL
    
    // Recording configuration
    bool m_recordingEnabled;
    
    // Debug configuration
    bool m_debugMode;
    
    // RTSP configuration
    int m_rtspLatency;
    int m_rtspBufferSize;
    std::string m_rtspQuality;
    
    // Validation state
    bool m_configurationValid;
};
