#pragma once

#include <QWidget>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>

class ZoneControlsWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ZoneControlsWidget(QWidget* parent = nullptr);
    ~ZoneControlsWidget() = default;

    QString getZoneName() const;
    void setZoneName(const QString& name);
    void clearZoneName();
    void setDrawingMode(bool enabled);

signals:
    void drawZoneRequested();
    void finishDrawingRequested();
    void clearZonesRequested();
    void hideZonesRequested();
    void showZonesRequested();
    void zoneNameChanged(const QString& name);

private slots:
    void onDrawZoneClicked();
    void onClearZonesClicked();
    void onHideZonesClicked();
    void onZoneNameChanged();

private:
    void setupUI();
    void applyMilitaryTheme();

    QVBoxLayout* m_mainLayout;
    QLabel* m_header;
    QWidget* m_controlsWidget;
    QHBoxLayout* m_controlsLayout;
    
    QLineEdit* m_zoneNameEdit;
    QPushButton* m_drawZoneBtn;
    QPushButton* m_clearZonesBtn;
    QPushButton* m_hideZonesBtn;
    
    QString m_zoneName;
    bool m_zonesHidden;
    bool m_drawingMode;
};
