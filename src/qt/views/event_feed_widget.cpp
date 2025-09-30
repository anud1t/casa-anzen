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
    m_eventList->setUniformItemSizes(false);
    m_eventList->setSpacing(14);
    m_eventList->setMovement(QListView::Static);
    m_eventList->setSelectionMode(QAbstractItemView::SingleSelection);
    
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
}

void EventFeedWidget::addEvent(const QString& title, const QPixmap& thumbnail, const QString& caption)
{
    QListWidgetItem* item = createEventItem(title, thumbnail, caption);
    m_eventList->addItem(item);
    m_eventList->setItemWidget(item, new EventCard(this));
    
    EventCard* card = qobject_cast<EventCard*>(m_eventList->itemWidget(item));
    if (card) {
        card->setTitle(title);
        card->setThumbnail(thumbnail);
        card->setCaption(caption);
        connect(card, &EventCard::clicked, this, &EventFeedWidget::onEventCardClicked);
    }
}

void EventFeedWidget::clearEvents()
{
    m_eventList->clear();
}

void EventFeedWidget::setCaptionForItem(QListWidgetItem* item, const QString& caption)
{
    if (!item) return;
    
    EventCard* card = qobject_cast<EventCard*>(m_eventList->itemWidget(item));
    if (card) {
        card->setCaption(caption);
    }
}

QListWidgetItem* EventFeedWidget::createEventItem(const QString& /*title*/, const QPixmap& /*thumbnail*/, const QString& /*caption*/)
{
    QListWidgetItem* item = new QListWidgetItem();
    QSize hint(300, 200);
    hint.setHeight(std::max(hint.height(), 190));
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
    if (currentItem) {
        emit captionRequested(currentItem);
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
