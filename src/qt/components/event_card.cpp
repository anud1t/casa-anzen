#include "event_card.hpp"
#include <QMouseEvent>
#include <QFont>
#include <QTimer>

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
    
    // Allow the card to receive mouse events for selection
    setFocusPolicy(Qt::NoFocus);
    
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
    // Remove fixed maximum width to allow dynamic sizing
    m_captionLabel->setVisible(false);
    m_captionLabel->setTextFormat(Qt::PlainText);
    m_captionLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    
    m_layout->addWidget(m_captionLabel);
}

void EventCard::applyMilitaryTheme()
{
    // Use a more efficient approach with separate stylesheets to avoid full repaints
    setStyleSheet(
        "#eventCard{ "
        "background: #1a1a1a; "
        "border: 1px solid #333333; "
        "border-radius: 2px; "
        "}"
    );
    
    // Store base stylesheet for efficient updates
    m_baseStyleSheet = styleSheet();

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
        "padding: 12px; "
        "font-size: 12px; "
        "line-height: 1.4; "
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
    // Update caption sizing to match new thumbnail width
    // Use a single-shot timer to ensure thumbnail is fully rendered
    QTimer::singleShot(0, this, &EventCard::updateCaptionSizing);
}

void EventCard::setCaption(const QString& caption)
{
    m_caption = caption;
    m_captionLabel->setText(caption);
    m_captionLabel->setVisible(!caption.isEmpty());
    // Update caption sizing to match thumbnail width
    updateCaptionSizing();
}

void EventCard::setCaptionVisible(bool visible)
{
    m_captionLabel->setVisible(visible);
}

void EventCard::setSelected(bool selected)
{
    if (selected) {
        setStyleSheet(m_baseStyleSheet + 
            "QFrame#eventCard{ "
            "border: 2px solid #00ff00; "
            "background: #1a2a1a; "
            "}");
    } else {
        setStyleSheet(m_baseStyleSheet);
    }
    setProperty("selected", selected);
    update();
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

void EventCard::resizeEvent(QResizeEvent* event)
{
    QFrame::resizeEvent(event);
    updateCaptionSizing();
}

void EventCard::updateCaptionSizing()
{
    if (m_captionLabel && m_captionLabel->isVisible()) {
        // Make caption width match the thumbnail width for uniform appearance
        if (m_thumbnail && m_thumbnail->isVisible()) {
            int thumbnailWidth = m_thumbnail->width();
            if (thumbnailWidth > 0) {
                // Set maximum width to match thumbnail, with a small margin for padding
                int captionWidth = thumbnailWidth - 4; // Account for padding/borders
                m_captionLabel->setMaximumWidth(captionWidth);
            }
        }
        m_captionLabel->setWordWrap(true);
        m_captionLabel->adjustSize();
        m_captionLabel->updateGeometry();
    }
}
