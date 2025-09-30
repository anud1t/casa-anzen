#include "event_card.hpp"
#include <QMouseEvent>
#include <QFont>

EventCard::EventCard(QWidget* parent)
    : QFrame(parent)
    , m_layout(nullptr)
    , m_titleBadge(nullptr)
    , m_thumbnail(nullptr)
    , m_captionLabel(nullptr)
{
    setupUI();
    applyMilitaryTheme();
}

void EventCard::setupUI()
{
    setObjectName("eventCard");
    setFrameShape(QFrame::NoFrame);
    
    m_layout = new QVBoxLayout(this);
    m_layout->setContentsMargins(16, 16, 16, 16);
    m_layout->setSpacing(12);

    // Title badge
    m_titleBadge = new QLabel(this);
    m_titleBadge->setAlignment(Qt::AlignHCenter);
    m_titleBadge->setMinimumHeight(28);
    
    QFont badgeFont = m_titleBadge->font();
    badgeFont.setBold(true);
    badgeFont.setPointSize(std::max(11, badgeFont.pointSize()));
    badgeFont.setLetterSpacing(QFont::PercentageSpacing, 110);
    m_titleBadge->setFont(badgeFont);
    
    m_layout->addWidget(m_titleBadge, 0, Qt::AlignHCenter);

    // Thumbnail
    m_thumbnail = new QLabel(this);
    m_thumbnail->setObjectName("thumbLabel");
    m_thumbnail->setAlignment(Qt::AlignCenter);
    m_thumbnail->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    m_thumbnail->setMinimumHeight(140);
    m_thumbnail->setScaledContents(true);
    
    m_layout->addWidget(m_thumbnail);

    // Caption
    m_captionLabel = new QLabel(this);
    m_captionLabel->setObjectName("captionLabel");
    m_captionLabel->setWordWrap(true);
    m_captionLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::MinimumExpanding);
    m_captionLabel->setMinimumHeight(50);
    m_captionLabel->setMaximumHeight(300);
    m_captionLabel->setVisible(false);
    
    m_layout->addWidget(m_captionLabel);
    m_layout->addStretch(1);
}

void EventCard::applyMilitaryTheme()
{
    setStyleSheet(
        "#eventCard{ "
        "background: #1a1a1a; "
        "border: 1px solid #333333; "
        "border-radius: 2px; "
        "}"
        "#eventCard:hover{ "
        "border: 1px solid #555555; "
        "background: #222222; "
        "}"
    );

    m_titleBadge->setStyleSheet(
        "color: #00ff00; "
        "background: #0d4d0d; "
        "padding: 4px 12px; "
        "border-radius: 2px; "
        "border: 1px solid #00aa00; "
        "font-weight: 600; "
        "font-family: 'Courier New', monospace; "
        "letter-spacing: 1px;"
    );

    m_thumbnail->setStyleSheet(
        "border: 1px solid #333333; "
        "border-radius: 2px; "
        "background: #0a0a0a; "
        "padding: 2px;"
    );

    m_captionLabel->setStyleSheet(
        "color: #cccccc; "
        "background: #0f0f0f; "
        "border: 1px solid #333333; "
        "border-radius: 2px; "
        "padding: 8px; "
        "font-size: 11px; "
        "line-height: 1.3; "
        "font-weight: 400; "
        "font-family: 'Courier New', monospace; "
        "letter-spacing: 0.2px;"
    );
}

void EventCard::setTitle(const QString& title)
{
    m_title = title;
    m_titleBadge->setText(title.toUpper());
}

void EventCard::setThumbnail(const QPixmap& thumbnail)
{
    m_thumbnailPixmap = thumbnail;
    m_thumbnail->setPixmap(thumbnail);
}

void EventCard::setCaption(const QString& caption)
{
    m_caption = caption;
    m_captionLabel->setText(caption);
    m_captionLabel->setVisible(!caption.isEmpty());
    updateCaptionSizing();
}

void EventCard::setCaptionVisible(bool visible)
{
    m_captionLabel->setVisible(visible);
}

QString EventCard::getTitle() const
{
    return m_title;
}

QPixmap EventCard::getThumbnail() const
{
    return m_thumbnailPixmap;
}

QString EventCard::getCaption() const
{
    return m_caption;
}

void EventCard::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        emit clicked();
    }
    QFrame::mousePressEvent(event);
}

void EventCard::updateCaptionSizing()
{
    if (m_captionLabel && m_captionLabel->isVisible()) {
        m_captionLabel->setMaximumWidth(width() - 32);
        m_captionLabel->adjustSize();
        m_captionLabel->updateGeometry();
    }
}
