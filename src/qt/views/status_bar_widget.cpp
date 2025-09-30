#include "status_bar_widget.hpp"

StatusBarWidget::StatusBarWidget(QWidget* parent)
    : QWidget(parent)
    , m_layout(nullptr)
    , m_statusLabel(nullptr)
    , m_modeLabel(nullptr)
    , m_fpsLabel(nullptr)
    , m_detectionsLabel(nullptr)
    , m_alertsLabel(nullptr)
    , m_recordingLabel(nullptr)
    , m_status("READY")
    , m_mode("MODE: PEOPLE + VEHICLES")
    , m_fps(0.0)
    , m_detections(0)
    , m_alerts(0)
    , m_recording(false)
{
    setupUI();
    applyMilitaryTheme();
}

void StatusBarWidget::setupUI()
{
    m_layout = new QHBoxLayout(this);
    m_layout->setContentsMargins(8, 4, 8, 4);
    m_layout->setSpacing(8);

    // Status label
    m_statusLabel = new QLabel("● READY", this);
    m_layout->addWidget(m_statusLabel);

    // Mode label
    m_modeLabel = new QLabel("MODE: PEOPLE + VEHICLES", this);
    m_layout->addWidget(m_modeLabel);

    // FPS label
    m_fpsLabel = new QLabel("FPS: 0", this);
    m_layout->addWidget(m_fpsLabel);

    // Detections label
    m_detectionsLabel = new QLabel("Detections: 0", this);
    m_layout->addWidget(m_detectionsLabel);

    // Alerts label
    m_alertsLabel = new QLabel("Alerts: 0", this);
    m_layout->addWidget(m_alertsLabel);

    // Recording label
    m_recordingLabel = new QLabel("Recording: OFF", this);
    m_layout->addWidget(m_recordingLabel);
}

void StatusBarWidget::applyMilitaryTheme()
{
    setStyleSheet(
        "QWidget{ "
        "background: #0a0a0a; "
        "border-top: 1px solid #333333; "
        "color: #00ff00; "
        "font-weight: 600; "
        "font-family: 'Courier New', monospace; "
        "padding: 4px 8px; "
        "}"
    );

    m_statusLabel->setStyleSheet(
        "color: #00ff00; "
        "font-weight: 700; "
        "font-size: 12px; "
        "font-family: 'Courier New', monospace; "
        "padding: 4px 8px; "
        "background: #0d4d0d; "
        "border-radius: 2px; "
        "border: 1px solid #00aa00;"
    );

    m_modeLabel->setStyleSheet(
        "color: #cccccc; "
        "font-weight: 600; "
        "font-size: 11px; "
        "font-family: 'Courier New', monospace; "
        "padding: 4px 8px; "
        "background: #1a1a1a; "
        "border-radius: 2px; "
        "border: 1px solid #333333;"
    );

    m_fpsLabel->setStyleSheet(
        "color: #ffff00; "
        "font-weight: 700; "
        "font-size: 11px; "
        "font-family: 'Courier New', monospace; "
        "padding: 4px 8px; "
        "background: #2a2a00; "
        "border-radius: 2px; "
        "border: 1px solid #aaaa00;"
    );

    m_detectionsLabel->setStyleSheet(
        "color: #ff0000; "
        "font-weight: 700; "
        "font-size: 11px; "
        "font-family: 'Courier New', monospace; "
        "padding: 4px 8px; "
        "background: #2a0000; "
        "border-radius: 2px; "
        "border: 1px solid #aa0000;"
    );

    m_alertsLabel->setStyleSheet(
        "color: #ff8800; "
        "font-weight: 700; "
        "font-size: 11px; "
        "font-family: 'Courier New', monospace; "
        "padding: 4px 8px; "
        "background: #2a1a00; "
        "border-radius: 2px; "
        "border: 1px solid #aa4400;"
    );

    m_recordingLabel->setStyleSheet(
        "color: #888888; "
        "font-weight: 700; "
        "font-size: 11px; "
        "font-family: 'Courier New', monospace; "
        "padding: 4px 8px; "
        "background: #1a1a1a; "
        "border-radius: 2px; "
        "border: 1px solid #333333;"
    );
}

void StatusBarWidget::setStatus(const QString& status)
{
    if (m_status != status) {
        m_status = status;
        updateStatusLabel();
        emit statusChanged(status);
    }
}

void StatusBarWidget::setMode(const QString& mode)
{
    if (m_mode != mode) {
        m_mode = mode;
        m_modeLabel->setText(mode);
    }
}

void StatusBarWidget::setFPS(double fps)
{
    if (m_fps != fps) {
        m_fps = fps;
        m_fpsLabel->setText(QString("FPS: %1").arg(fps, 0, 'f', 1));
    }
}

void StatusBarWidget::setDetections(int count)
{
    if (m_detections != count) {
        m_detections = count;
        m_detectionsLabel->setText(QString("Detections: %1").arg(count));
    }
}

void StatusBarWidget::setAlerts(int count)
{
    if (m_alerts != count) {
        m_alerts = count;
        m_alertsLabel->setText(QString("Alerts: %1").arg(count));
    }
}

void StatusBarWidget::setRecording(bool recording)
{
    if (m_recording != recording) {
        m_recording = recording;
        m_recordingLabel->setText(recording ? "Recording: ON" : "Recording: OFF");
    }
}

QString StatusBarWidget::getStatus() const
{
    return m_status;
}

QString StatusBarWidget::getMode() const
{
    return m_mode;
}

double StatusBarWidget::getFPS() const
{
    return m_fps;
}

int StatusBarWidget::getDetections() const
{
    return m_detections;
}

int StatusBarWidget::getAlerts() const
{
    return m_alerts;
}

bool StatusBarWidget::isRecording() const
{
    return m_recording;
}

void StatusBarWidget::updateStatusLabel()
{
    m_statusLabel->setText("● " + m_status.toUpper());
}
