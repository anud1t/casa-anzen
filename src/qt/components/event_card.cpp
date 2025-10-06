#include "event_card.hpp"
#include <QMouseEvent>
#include <QFont>
#include <QTimer>

// Define the standard thumbnail size
const QSize EventCard::THUMBNAIL_SIZE(200, 150);

EventCard::EventCard(QWidget* parent)
    : QFrame(parent)
    , m_layout(nullptr)
    , m_titleBadge(nullptr)
    , m_thumbnail(nullptr)
    , m_captionLabel(nullptr)
    , m_captionOverlayMode(false)
    , m_hasAICaption(false)
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
    m_layout->setContentsMargins(8, 4, 8, 4);
    m_layout->setSpacing(2);

    // Title badge
    m_titleBadge = new QLabel(this);
    m_titleBadge->setAlignment(Qt::AlignHCenter);
    m_titleBadge->setMinimumHeight(20);
    m_titleBadge->setMaximumHeight(20);
    
    QFont badgeFont = m_titleBadge->font();
    badgeFont.setBold(true);
    badgeFont.setPointSize(std::max(11, badgeFont.pointSize()));
    badgeFont.setLetterSpacing(QFont::PercentageSpacing, 110);
    m_titleBadge->setFont(badgeFont);
    
    m_layout->addWidget(m_titleBadge, 0, Qt::AlignHCenter);

    // Create a container for thumbnail and caption overlay
    QWidget* contentContainer = new QWidget(this);
    contentContainer->setFixedSize(THUMBNAIL_SIZE);
    contentContainer->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    contentContainer->setContentsMargins(0, 0, 0, 0);
    
    // Thumbnail
    m_thumbnail = new QLabel(contentContainer);
    m_thumbnail->setObjectName("thumbLabel");
    m_thumbnail->setAlignment(Qt::AlignCenter);
    m_thumbnail->setGeometry(0, 0, THUMBNAIL_SIZE.width(), THUMBNAIL_SIZE.height());
    m_thumbnail->setScaledContents(false);  // We'll handle scaling ourselves
    
    // Caption overlay (initially hidden)
    m_captionLabel = new QLabel(contentContainer);
    m_captionLabel->setObjectName("captionLabel");
    m_captionLabel->setWordWrap(true);
    m_captionLabel->setGeometry(0, 0, THUMBNAIL_SIZE.width(), THUMBNAIL_SIZE.height());
    m_captionLabel->setAlignment(Qt::AlignCenter);
    m_captionLabel->setVisible(false);
    m_captionLabel->setTextFormat(Qt::PlainText);
    m_captionLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    m_captionLabel->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    m_captionLabel->setFixedSize(THUMBNAIL_SIZE);
    
    m_layout->addWidget(contentContainer, 0, Qt::AlignHCenter);
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
        "padding: 2px 8px; "
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
        "background: rgba(15, 15, 15, 0.95); "
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
    // Store the full resolution image
    m_thumbnailPixmap = thumbnail;
    
    // Create a uniform-sized thumbnail for display
    // Scale the image to fit within THUMBNAIL_SIZE while preserving aspect ratio
    m_displayThumbnail = thumbnail.scaled(THUMBNAIL_SIZE, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    
    // Display the uniform-sized thumbnail
    m_thumbnail->setPixmap(m_displayThumbnail);
}

void EventCard::setCaption(const QString& caption)
{
    m_caption = caption;
    m_captionLabel->setText(caption);
    
    // Only show caption if we're in overlay mode or it's not an AI caption
    if (m_captionOverlayMode && m_hasAICaption) {
        m_captionLabel->setVisible(true);
        m_thumbnail->setVisible(false);
    } else {
        m_captionLabel->setVisible(false);
        m_thumbnail->setVisible(true);
    }
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
    return m_thumbnailPixmap;  // Returns full resolution image
}

QPixmap EventCard::getDisplayThumbnail() const
{
    return m_displayThumbnail;  // Returns uniform-sized thumbnail
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
}

void EventCard::setCaptionOverlayMode(bool overlay)
{
    m_captionOverlayMode = overlay;
    
    if (m_hasAICaption) {
        if (overlay) {
            m_captionLabel->setVisible(true);
            m_thumbnail->setVisible(false);
        } else {
            m_captionLabel->setVisible(false);
            m_thumbnail->setVisible(true);
        }
    }
}

void EventCard::toggleCaptionOverlay()
{
    if (m_hasAICaption) {
        setCaptionOverlayMode(!m_captionOverlayMode);
    }
}

void EventCard::setAICaption(const QString& caption)
{
    m_hasAICaption = true;
    m_caption = caption;
    m_captionLabel->setText(caption);
    
    // Initially show thumbnail, not caption
    m_captionLabel->setVisible(false);
    m_thumbnail->setVisible(true);
    m_captionOverlayMode = false;
}
