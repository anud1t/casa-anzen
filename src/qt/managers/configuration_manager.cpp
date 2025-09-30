#include "configuration_manager.hpp"
#include <QFile>
#include <QDebug>

ConfigurationManager::ConfigurationManager(QObject* parent)
    : QObject(parent)
    , m_modelPath("")
    , m_videoSource("")
    , m_confidenceThreshold(0.25f)
    , m_detectionMode(2) // PEOPLE+VEHICLES
    , m_recordingEnabled(false)
    , m_debugMode(false)
    , m_rtspLatency(200)
    , m_rtspBufferSize(10)
    , m_rtspQuality("high")
    , m_configurationValid(false)
{
    validateConfiguration();
}

void ConfigurationManager::setModelPath(const std::string& modelPath)
{
    if (m_modelPath != modelPath) {
        m_modelPath = modelPath;
        validateConfiguration();
        emit modelPathChanged(modelPath);
        emit configurationChanged();
    }
}

std::string ConfigurationManager::getModelPath() const
{
    return m_modelPath;
}

void ConfigurationManager::setVideoSource(const std::string& videoSource)
{
    if (m_videoSource != videoSource) {
        m_videoSource = videoSource;
        validateConfiguration();
        emit videoSourceChanged(videoSource);
        emit configurationChanged();
    }
}

std::string ConfigurationManager::getVideoSource() const
{
    return m_videoSource;
}

void ConfigurationManager::setConfidenceThreshold(float threshold)
{
    if (m_confidenceThreshold != threshold) {
        m_confidenceThreshold = threshold;
        emit confidenceThresholdChanged(threshold);
        emit configurationChanged();
    }
}

float ConfigurationManager::getConfidenceThreshold() const
{
    return m_confidenceThreshold;
}

void ConfigurationManager::setRecordingEnabled(bool enabled)
{
    if (m_recordingEnabled != enabled) {
        m_recordingEnabled = enabled;
        emit recordingEnabledChanged(enabled);
        emit configurationChanged();
    }
}

bool ConfigurationManager::isRecordingEnabled() const
{
    return m_recordingEnabled;
}

void ConfigurationManager::setDebugMode(bool debugMode)
{
    if (m_debugMode != debugMode) {
        m_debugMode = debugMode;
        emit debugModeChanged(debugMode);
        emit configurationChanged();
    }
}

bool ConfigurationManager::isDebugMode() const
{
    return m_debugMode;
}

void ConfigurationManager::setRtspLatency(int latencyMs)
{
    if (m_rtspLatency != latencyMs) {
        m_rtspLatency = latencyMs;
        emit rtspSettingsChanged();
        emit configurationChanged();
    }
}

int ConfigurationManager::getRtspLatency() const
{
    return m_rtspLatency;
}

void ConfigurationManager::setRtspBufferSize(int bufferSize)
{
    if (m_rtspBufferSize != bufferSize) {
        m_rtspBufferSize = bufferSize;
        emit rtspSettingsChanged();
        emit configurationChanged();
    }
}

int ConfigurationManager::getRtspBufferSize() const
{
    return m_rtspBufferSize;
}

void ConfigurationManager::setRtspQuality(const std::string& quality)
{
    if (m_rtspQuality != quality) {
        m_rtspQuality = quality;
        emit rtspSettingsChanged();
        emit configurationChanged();
    }
}

std::string ConfigurationManager::getRtspQuality() const
{
    return m_rtspQuality;
}

void ConfigurationManager::setDetectionMode(int mode)
{
    if (m_detectionMode != mode) {
        m_detectionMode = mode;
        emit configurationChanged();
    }
}

int ConfigurationManager::getDetectionMode() const
{
    return m_detectionMode;
}

bool ConfigurationManager::isModelPathValid() const
{
    if (m_modelPath.empty()) {
        return false;
    }
    
    QFile file(QString::fromStdString(m_modelPath));
    return file.exists();
}

bool ConfigurationManager::isVideoSourceValid() const
{
    if (m_videoSource.empty()) {
        return false;
    }
    
    // Check if it's a file path
    if (m_videoSource.find("://") == std::string::npos) {
        QFile file(QString::fromStdString(m_videoSource));
        return file.exists();
    }
    
    // For URLs/streams, assume valid for now
    return true;
}

bool ConfigurationManager::isConfigurationValid() const
{
    return m_configurationValid;
}

void ConfigurationManager::validateConfiguration()
{
    bool wasValid = m_configurationValid;
    m_configurationValid = isModelPathValid() && isVideoSourceValid();
    
    if (wasValid != m_configurationValid) {
        qDebug() << "Configuration validity changed:" << m_configurationValid;
    }
}
