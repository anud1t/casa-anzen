#include "event_feed_widget.hpp"
#include "../components/event_card.hpp"
#include <QListWidgetItem>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QJsonObject>
#include <QBuffer>
#include <QImage>
#include <QListWidget>
#include <QListView>

EventFeedWidget::EventFeedWidget(QWidget* parent)
    : QWidget(parent)
    , m_mainLayout(nullptr)
    , m_header(nullptr)
    , m_toolbar(nullptr)
    , m_eventList(nullptr)
    , m_viewBtn(nullptr)
    , m_captionBtn(nullptr)
    , m_deleteBtn(nullptr)
    , m_deleteAllBtn(nullptr)
    , m_networkManager(nullptr)
{
    setupUI();
    applyMilitaryTheme();
    setupEventToolbar();
}

void EventFeedWidget::setupUI()
{
    m_mainLayout = new QVBoxLayout(this);
    m_mainLayout->setContentsMargins(12, 12, 12, 12);
    m_mainLayout->setSpacing(8);

    // Header
    m_header = new QLabel("Event Feed", this);
    m_mainLayout->addWidget(m_header);

    // Event list
    m_eventList = new QListWidget(this);
    m_eventList->setViewMode(QListView::ListMode);
    m_eventList->setResizeMode(QListView::Adjust);
    m_eventList->setUniformItemSizes(true);  // Enable for better scrolling performance
    m_eventList->setSpacing(14);
    m_eventList->setMovement(QListView::Static);
    m_eventList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_eventList->setSelectionBehavior(QAbstractItemView::SelectItems);
    m_eventList->setFocusPolicy(Qt::StrongFocus);
    
    // Enable smooth scrolling and optimize for performance
    m_eventList->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
    m_eventList->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
    m_eventList->setAutoScroll(false);  // Disable auto-scroll to prevent interference
    
    // Enable performance optimizations
    m_eventList->setAttribute(Qt::WA_OpaquePaintEvent, true);
    m_eventList->setAttribute(Qt::WA_NoSystemBackground, true);
    m_eventList->setAttribute(Qt::WA_StaticContents, true);
    
    m_mainLayout->addWidget(m_eventList, 1);

    // Network manager for caption requests
    m_networkManager = new QNetworkAccessManager(this);
}

void EventFeedWidget::applyMilitaryTheme()
{
    m_header->setStyleSheet(
        "font-weight: 700; "
        "font-size: 14px; "
        "color: #00ff00; "
        "padding: 8px; "
        "background: #0d4d0d; "
        "border: 1px solid #00aa00; "
        "border-radius: 2px; "
        "margin: 0px 0px 8px 0px; "
        "font-family: 'Courier New', monospace; "
        "letter-spacing: 1px;"
    );

    m_eventList->setStyleSheet(
        "QListWidget{ "
        "background: #0a0a0a; "
        "border: 1px solid #333333; "
        "border-radius: 2px; "
        "padding: 4px; "
        "}"
        "QListWidget::item{ "
        "color: #cccccc; "
        "border: none; "
        "margin: 2px 0px; "
        "}"
        "QListWidget::item:selected{ "
        "background: transparent; "
        "border: none; "
        "}"
        "QListWidget::item:selected:hover{ "
        "background: transparent; "
        "border: none; "
        "}"
        "QScrollBar:vertical { "
        "background: #1a1a1a; "
        "width: 10px; "
        "border-radius: 2px; "
        "}"
        "QScrollBar::handle:vertical { "
        "background: #333333; "
        "border-radius: 2px; "
        "min-height: 20px; "
        "}"
        "QScrollBar::handle:vertical:hover { "
        "background: #555555; "
        "}"
    );
}

void EventFeedWidget::setupEventToolbar()
{
    m_toolbar = new QWidget(this);
    QHBoxLayout* toolbarLayout = new QHBoxLayout(m_toolbar);
    toolbarLayout->setContentsMargins(8, 4, 8, 4);
    toolbarLayout->setSpacing(8);

    m_viewBtn = new QPushButton("VIEW", m_toolbar);
    m_captionBtn = new QPushButton("CAPTION", m_toolbar);
    m_deleteBtn = new QPushButton("DELETE", m_toolbar);
    m_deleteAllBtn = new QPushButton("DELETE ALL", m_toolbar);

    m_viewBtn->setCursor(Qt::PointingHandCursor);
    m_captionBtn->setCursor(Qt::PointingHandCursor);
    m_deleteBtn->setCursor(Qt::PointingHandCursor);
    m_deleteAllBtn->setCursor(Qt::PointingHandCursor);

    toolbarLayout->addWidget(m_viewBtn);
    toolbarLayout->addWidget(m_captionBtn);
    toolbarLayout->addWidget(m_deleteBtn);
    toolbarLayout->addStretch();
    toolbarLayout->addWidget(m_deleteAllBtn);

    m_toolbar->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    m_toolbar->setMinimumHeight(44);
    m_deleteAllBtn->setObjectName("Danger");

    m_toolbar->setStyleSheet(
        "QWidget{ "
        "background: #1a1a1a; "
        "border: 1px solid #333333; "
        "border-radius: 2px; "
        "padding: 4px; "
        "}"
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
        "}"
        "QPushButton#Danger{ "
        "background: #4a0d0d; "
        "color: #ff0000; "
        "border: 1px solid #aa0000; "
        "}"
        "QPushButton#Danger:hover{ "
        "background: #6a0d0d; "
        "border: 1px solid #ff0000; "
        "color: #ffffff; "
        "}"
    );

    m_mainLayout->insertWidget(1, m_toolbar);

    // Connect signals
    connect(m_viewBtn, &QPushButton::clicked, this, &EventFeedWidget::onViewClicked);
    connect(m_captionBtn, &QPushButton::clicked, this, &EventFeedWidget::onCaptionClicked);
    connect(m_deleteBtn, &QPushButton::clicked, this, &EventFeedWidget::onDeleteClicked);
    connect(m_deleteAllBtn, &QPushButton::clicked, this, &EventFeedWidget::onDeleteAllClicked);
    connect(m_eventList, &QListWidget::itemSelectionChanged, this, &EventFeedWidget::onSelectionChanged);
}

void EventFeedWidget::addEvent(const QString& title, const QPixmap& thumbnail, const QString& caption, const QString& imagePath)
{
    QListWidgetItem* item = createEventItem(title, thumbnail, caption);
    
    // Store the image path in the item's data
    if (!imagePath.isEmpty()) {
        item->setData(Qt::UserRole, imagePath);
    }
    
    m_eventList->addItem(item);
    m_eventList->setItemWidget(item, new EventCard(this));

    EventCard* card = qobject_cast<EventCard*>(m_eventList->itemWidget(item));
    if (card) {
        card->setTitle(title);
        card->setThumbnail(thumbnail);
        card->setCaption(caption);
        connect(card, &EventCard::clicked, this, &EventFeedWidget::onEventCardClicked);

        // Ensure the list item height closely follows the card's content
        card->adjustSize();
        QSize hint = card->sizeHint();
        hint.setWidth(qMax(hint.width(), 300));
        hint.setHeight(qMax(hint.height(), 160));
        item->setSizeHint(hint);
        // Removed viewport()->update() to prevent unnecessary full repaints
    }
}

void EventFeedWidget::clearEvents()
{
    m_eventList->clear();
}

void EventFeedWidget::setCaptionForItem(QListWidgetItem* item, const QString& caption)
{
    qDebug() << "EventFeedWidget::setCaptionForItem called with caption:" << caption;
    if (!item) {
        qDebug() << "No item provided to setCaptionForItem";
        return;
    }
    
    EventCard* card = qobject_cast<EventCard*>(m_eventList->itemWidget(item));
    if (card) {
        qDebug() << "Setting caption on EventCard:" << caption;
        const QString trimmed = caption.trimmed();
        if (trimmed == "(no caption)" || trimmed.isEmpty()) {
            // Do not append placeholder text
            return;
        }
        // Replace the previous AI caption, but keep the original capture block (first paragraph)
        QString existing = card->getCaption();
        const int sepIdx = existing.indexOf("\n\n");
        QString baseBlock = sepIdx >= 0 ? existing.left(sepIdx) : existing;
        QString merged = baseBlock.isEmpty() ? trimmed : baseBlock + "\n\n" + trimmed;
        card->setCaption(merged);

        // Ensure the list item grows to fit the new caption text
        card->adjustSize();
        card->updateGeometry();
        QSize cardHint = card->sizeHint();
        QSize itemHint = item->sizeHint();
        int requiredHeight = qMax(cardHint.height(), 220); // provide comfortable minimum
        if (itemHint.height() < requiredHeight) {
            itemHint.setHeight(requiredHeight);
            item->setSizeHint(itemHint);
        }
        m_eventList->viewport()->update();
    } else {
        qDebug() << "No EventCard found for item";
    }
}

QListWidgetItem* EventFeedWidget::getLastItem() const
{
    int count = m_eventList->count();
    if (count > 0) {
        return m_eventList->item(count - 1);
    }
    return nullptr;
}

QListWidgetItem* EventFeedWidget::createEventItem(const QString& /*title*/, const QPixmap& /*thumbnail*/, const QString& /*caption*/)
{
    QListWidgetItem* item = new QListWidgetItem();
    QSize hint(300, 170); // lower default height; will grow to fit content
    item->setSizeHint(hint);
    return item;
}

void EventFeedWidget::onViewClicked()
{
    QListWidgetItem* currentItem = m_eventList->currentItem();
    if (currentItem) {
        emit viewRequested(currentItem);
    }
}

void EventFeedWidget::onCaptionClicked()
{
    QListWidgetItem* currentItem = m_eventList->currentItem();
    qDebug() << "Caption button clicked, current item:" << (currentItem ? "YES" : "NO");
    if (currentItem) {
        QString imagePath = currentItem->data(Qt::UserRole).toString();
        qDebug() << "Requesting caption for image path:" << imagePath;
        emit captionRequested(currentItem);
    } else {
        qDebug() << "No item selected for captioning";
    }
}

void EventFeedWidget::onDeleteClicked()
{
    QListWidgetItem* currentItem = m_eventList->currentItem();
    if (currentItem) {
        emit deleteRequested(currentItem);
    }
}

void EventFeedWidget::onDeleteAllClicked()
{
    emit deleteAllRequested();
}

void EventFeedWidget::onEventCardClicked()
{
    // Handle event card click if needed
}

void EventFeedWidget::onSelectionChanged()
{
    QListWidgetItem* currentItem = m_eventList->currentItem();
    qDebug() << "Selection changed, current item:" << (currentItem ? "YES" : "NO");
    
    // Only update the previously selected and newly selected items for better performance
    if (!m_eventList) return;
    
    // Update the newly selected item
    if (currentItem) {
        QWidget* w = m_eventList->itemWidget(currentItem);
        EventCard* card = qobject_cast<EventCard*>(w);
        if (card) {
            card->setSelected(true);
        }
    }
    
    // Update the previously selected item (if different)
    static QListWidgetItem* lastSelectedItem = nullptr;
    if (lastSelectedItem && lastSelectedItem != currentItem) {
        QWidget* w = m_eventList->itemWidget(lastSelectedItem);
        EventCard* card = qobject_cast<EventCard*>(w);
        if (card) {
            card->setSelected(false);
        }
    }
    
    lastSelectedItem = currentItem;
}
