#pragma once

#include <QStatusBar>
#include <QLabel>
#include <QHBoxLayout>

class StatusBarWidget : public QStatusBar
{
    Q_OBJECT

public:
    explicit StatusBarWidget(QWidget* parent = nullptr);
    ~StatusBarWidget() = default;

    void setStatus(const QString& status);
    void setMode(const QString& mode);
    void setFPS(double fps);
    void setDetections(int count);
    void setAlerts(int count);
    void setRecording(bool recording);

    QString getStatus() const;
    QString getMode() const;
    double getFPS() const;
    int getDetections() const;
    int getAlerts() const;
    bool isRecording() const;

signals:
    void statusChanged(const QString& status);

private:
    void setupUI();
    void applyMilitaryTheme();
    void updateStatusLabel();

    // QStatusBar manages its own layout, no need for m_layout
    QLabel* m_statusLabel;
    QLabel* m_modeLabel;
    QLabel* m_fpsLabel;
    QLabel* m_detectionsLabel;
    QLabel* m_alertsLabel;
    QLabel* m_recordingLabel;

    QString m_status;
    QString m_mode;
    double m_fps;
    int m_detections;
    int m_alerts;
    bool m_recording;
};
