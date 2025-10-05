#pragma once

#include <QStatusBar>
#include <QLabel>
#include <QPushButton>
#include <QHBoxLayout>

class StatusBarWidget : public QStatusBar
{
    Q_OBJECT

public:
    explicit StatusBarWidget(QWidget* parent = nullptr);
    ~StatusBarWidget() = default;

    void setStatus(const QString& status);
    void setMode(const QString& mode);
    void setMode(int mode);
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
    int getCurrentMode() const;

signals:
    void statusChanged(const QString& status);
    void modeClicked();

private slots:
    void onModeButtonClicked();

private:
    void setupUI();
    void applyMilitaryTheme();
    void updateStatusLabel();
    void cycleMode();
    
    // Event handlers
    void mousePressEvent(QMouseEvent* event) override;

    // QStatusBar manages its own layout, no need for m_layout
    QLabel* m_statusLabel;
    QPushButton* m_modeButton;
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
    int m_currentMode; // 0=PEOPLE, 1=VEHICLES, 2=PEOPLE+VEHICLES, 3=ALL
};
