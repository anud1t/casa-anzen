#include "zone_controls_widget.hpp"

ZoneControlsWidget::ZoneControlsWidget(QWidget* parent)
    : QWidget(parent)
    , m_mainLayout(nullptr)
    , m_header(nullptr)
    , m_controlsWidget(nullptr)
    , m_controlsLayout(nullptr)
    , m_zoneNameEdit(nullptr)
    , m_drawZoneBtn(nullptr)
    , m_clearZonesBtn(nullptr)
    , m_hideZonesBtn(nullptr)
    , m_zoneName("")
    , m_zonesHidden(false)
    , m_drawingMode(false)
{
    setupUI();
    applyMilitaryTheme();
}

void ZoneControlsWidget::setupUI()
{
    m_mainLayout = new QVBoxLayout(this);
    m_mainLayout->setContentsMargins(0, 0, 0, 0);
    m_mainLayout->setSpacing(8);

    // Header
    m_header = new QLabel("Zones", this);
    m_mainLayout->addWidget(m_header);

    // Controls widget
    m_controlsWidget = new QWidget(this);
    m_controlsLayout = new QHBoxLayout(m_controlsWidget);
    m_controlsLayout->setContentsMargins(0, 0, 0, 0);
    m_controlsLayout->setSpacing(8);

    // Zone name input
    m_zoneNameEdit = new QLineEdit(m_controlsWidget);
    m_zoneNameEdit->setPlaceholderText("Zone label (optional)");
    m_zoneNameEdit->setMaximumWidth(150);
    m_controlsLayout->addWidget(m_zoneNameEdit);

    // Draw Zone button
    m_drawZoneBtn = new QPushButton("Draw Zone", m_controlsWidget);
    m_drawZoneBtn->setCursor(Qt::PointingHandCursor);
    m_controlsLayout->addWidget(m_drawZoneBtn);

    // Clear Zones button
    m_clearZonesBtn = new QPushButton("Clear Zones", m_controlsWidget);
    m_clearZonesBtn->setCursor(Qt::PointingHandCursor);
    m_controlsLayout->addWidget(m_clearZonesBtn);

    // Hide/Show Zones button
    m_hideZonesBtn = new QPushButton("Hide Zones", m_controlsWidget);
    m_hideZonesBtn->setCursor(Qt::PointingHandCursor);
    m_controlsLayout->addWidget(m_hideZonesBtn);

    m_mainLayout->addWidget(m_controlsWidget);

    // Connect signals
    connect(m_drawZoneBtn, &QPushButton::clicked, this, &ZoneControlsWidget::onDrawZoneClicked);
    connect(m_clearZonesBtn, &QPushButton::clicked, this, &ZoneControlsWidget::onClearZonesClicked);
    connect(m_hideZonesBtn, &QPushButton::clicked, this, &ZoneControlsWidget::onHideZonesClicked);
    connect(m_zoneNameEdit, &QLineEdit::textChanged, this, &ZoneControlsWidget::onZoneNameChanged);
}

void ZoneControlsWidget::applyMilitaryTheme()
{
    m_header->setStyleSheet(
        "font-weight: 700; "
        "font-size: 14px; "
        "color: #00ff00; "
        "padding: 8px; "
        "background: #0d4d0d; "
        "border: 1px solid #00aa00; "
        "border-radius: 2px; "
        "margin: 8px 0px 4px 0px; "
        "font-family: 'Courier New', monospace; "
        "letter-spacing: 1px;"
    );

    m_zoneNameEdit->setStyleSheet(
        "background: #1a1a1a; "
        "color: #cccccc; "
        "border: 1px solid #333333; "
        "border-radius: 2px; "
        "padding: 6px 8px; "
        "font-family: 'Courier New', monospace; "
        "font-size: 11px;"
    );

    QString buttonStyle = 
        "QPushButton{ "
        "background: #2a2a2a; "
        "color: #00ff00; "
        "border: 1px solid #555555; "
        "padding: 6px 12px; "
        "border-radius: 2px; "
        "min-height: 28px; "
        "font-weight: 600; "
        "font-size: 11px; "
        "font-family: 'Courier New', monospace; "
        "letter-spacing: 0.5px; "
        "}"
        "QPushButton:hover{ "
        "background: #3a3a3a; "
        "border: 1px solid #00aa00; "
        "color: #ffffff; "
        "}"
        "QPushButton:pressed{ "
        "background: #1a1a1a; "
        "border: 1px solid #00ff00; "
        "}";

    m_drawZoneBtn->setStyleSheet(buttonStyle);
    m_clearZonesBtn->setStyleSheet(buttonStyle);
    m_hideZonesBtn->setStyleSheet(buttonStyle);
}

QString ZoneControlsWidget::getZoneName() const
{
    return m_zoneName;
}

void ZoneControlsWidget::setZoneName(const QString& name)
{
    m_zoneName = name;
    m_zoneNameEdit->setText(name);
}

void ZoneControlsWidget::clearZoneName()
{
    m_zoneName.clear();
    m_zoneNameEdit->clear();
}

void ZoneControlsWidget::setDrawingMode(bool enabled)
{
    m_drawingMode = enabled;
    m_drawZoneBtn->setText(enabled ? "Finish" : "Draw Zone");
}

void ZoneControlsWidget::onDrawZoneClicked()
{
    if (m_drawingMode) {
        // Currently in drawing mode, finish drawing
        m_drawingMode = false;
        m_drawZoneBtn->setText("Draw Zone");
        emit finishDrawingRequested();
    } else {
        // Not in drawing mode, start drawing
        m_drawingMode = true;
        m_drawZoneBtn->setText("Finish");
        emit drawZoneRequested();
    }
}

void ZoneControlsWidget::onClearZonesClicked()
{
    emit clearZonesRequested();
}

void ZoneControlsWidget::onHideZonesClicked()
{
    m_zonesHidden = !m_zonesHidden;
    m_hideZonesBtn->setText(m_zonesHidden ? "Show Zones" : "Hide Zones");
    
    if (m_zonesHidden) {
        emit hideZonesRequested();
    } else {
        emit showZonesRequested();
    }
}

void ZoneControlsWidget::onZoneNameChanged()
{
    m_zoneName = m_zoneNameEdit->text();
    emit zoneNameChanged(m_zoneName);
}
