#include "status_bar_widget.hpp"
#include <QMouseEvent>

StatusBarWidget::StatusBarWidget(QWidget* parent)
    : QStatusBar(parent)
    , m_statusLabel(nullptr)
    , m_modeButton(nullptr)
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
    , m_currentMode(2) // Default to PEOPLE + VEHICLES
{
    setupUI();
    applyMilitaryTheme();
}

void StatusBarWidget::setupUI()
{
    // QStatusBar manages its own layout, we just add widgets directly
    
    // Status label
    m_statusLabel = new QLabel("● READY", this);
    addWidget(m_statusLabel);

    // Mode button (clickable)
    m_modeButton = new QPushButton("MODE: PEOPLE + VEHICLES", this);
    m_modeButton->setCursor(Qt::PointingHandCursor);
    m_modeButton->setFocusPolicy(Qt::StrongFocus); // Ensure button can receive focus
    m_modeButton->setMinimumHeight(30); // Ensure button has minimum height
    m_modeButton->setStyleSheet("QPushButton { background-color: #2b2b2b; border: 1px solid #555; padding: 5px; } QPushButton:hover { background-color: #3b3b3b; }");
    connect(m_modeButton, &QPushButton::clicked, this, &StatusBarWidget::onModeButtonClicked);
    addWidget(m_modeButton);

    // FPS label
    m_fpsLabel = new QLabel("FPS: 0", this);
    addWidget(m_fpsLabel);

    // Detections label
    m_detectionsLabel = new QLabel("Detections: 0", this);
    addWidget(m_detectionsLabel);

    // Alerts label
    m_alertsLabel = new QLabel("Alerts: 0", this);
    addWidget(m_alertsLabel);

    // Recording label (right-aligned)
    m_recordingLabel = new QLabel("Recording: OFF", this);
    addPermanentWidget(m_recordingLabel);
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

    m_modeButton->setStyleSheet(
        "QPushButton { "
        "color: #cccccc; "
        "font-weight: 600; "
        "font-size: 11px; "
        "font-family: 'Courier New', monospace; "
        "padding: 4px 8px; "
        "background: #1a1a1a; "
        "border-radius: 2px; "
        "border: 1px solid #333333; "
        "text-align: left; "
        "}"
        "QPushButton:hover { "
        "background: #2a2a2a; "
        "border: 1px solid #555555; "
        "}"
        "QPushButton:pressed { "
        "background: #0a0a0a; "
        "border: 1px solid #777777; "
        "}"
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
        m_modeButton->setText(mode);
    }
}

void StatusBarWidget::setMode(int mode)
{
    if (mode >= 0 && mode <= 3) {
        m_currentMode = mode;
        
        QString modeText;
        switch (mode) {
            case 0: modeText = "MODE: PEOPLE"; break;
            case 1: modeText = "MODE: VEHICLES"; break;
            case 2: modeText = "MODE: PEOPLE + VEHICLES"; break;
            case 3: modeText = "MODE: ALL"; break;
            default: modeText = "MODE: UNKNOWN"; break;
        }
        
        m_mode = modeText;
        m_modeButton->setText(modeText);
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

void StatusBarWidget::onModeButtonClicked()
{
    emit modeClicked();
}

void StatusBarWidget::cycleMode()
{
    // Cycle through modes: 0=PEOPLE, 1=VEHICLES, 2=PEOPLE+VEHICLES, 3=ALL
    m_currentMode = (m_currentMode + 1) % 4;
    
    QString modeText;
    switch (m_currentMode) {
        case 0: modeText = "MODE: PEOPLE"; break;
        case 1: modeText = "MODE: VEHICLES"; break;
        case 2: modeText = "MODE: PEOPLE + VEHICLES"; break;
        case 3: modeText = "MODE: ALL"; break;
        default: modeText = "MODE: UNKNOWN"; break;
    }
    
    m_mode = modeText;
    m_modeButton->setText(modeText);
}

void StatusBarWidget::mousePressEvent(QMouseEvent* event)
{
    QStatusBar::mousePressEvent(event);
}

int StatusBarWidget::getCurrentMode() const
{
    return m_currentMode;
}
