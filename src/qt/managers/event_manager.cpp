#include "event_manager.hpp"
#include "../views/event_feed_widget.hpp"
#include "caption_manager.hpp"
#include <QDebug>
#include <QCoreApplication>
#include <QProcess>

EventManager::EventManager(QObject* parent)
    : QObject(parent)
    , m_eventFeed(nullptr)
    , m_captionManager(nullptr)
{
    setupEventHandling();
    ensureCaptureDirectories();
}

void EventManager::setEventFeed(EventFeedWidget* eventFeed)
{
    m_eventFeed = eventFeed;
    if (m_eventFeed) {
        connect(m_eventFeed, &EventFeedWidget::viewRequested, this, &EventManager::viewEvent);
        connect(m_eventFeed, &EventFeedWidget::captionRequested, this, [this](QListWidgetItem* item){
            QString path = item->data(Qt::UserRole).toString();
            requestCaption(item, path);
        });
        connect(m_eventFeed, &EventFeedWidget::deleteRequested, this, &EventManager::deleteEvent);
        connect(m_eventFeed, &EventFeedWidget::deleteAllRequested, this, &EventManager::deleteAllEvents);
    }
}

void EventManager::setCaptionManager(CaptionManager* captionManager)
{
    m_captionManager = captionManager;
    
    if (m_captionManager) {
        connect(m_captionManager, &CaptionManager::captionReady,
                this, &EventManager::onCaptionReady);
        connect(m_captionManager, &CaptionManager::captionFailed,
                this, &EventManager::onCaptionFailed);
    }
}

void EventManager::addEvent(const QString& title, const QPixmap& thumbnail, const QString& caption, const QString& imagePath)
{
    if (!m_eventFeed) {
        qDebug() << "Event feed not set, cannot add event";
        return;
    }

    // Add the event to the feed (manual captioning only)
    m_eventFeed->addEvent(title, thumbnail, caption, imagePath);
    
    emit eventAdded(title);
    // Reduced verbosity by default
}

void EventManager::deleteEvent(QListWidgetItem* item)
{
    if (!item || !m_eventFeed) {
        return;
    }

    // Cancel any in-flight caption requests tied to this item
    if (m_captionManager) {
        m_captionManager->cancelRequestsForItem(item);
    }

    // Get event path from item data if available
    QString path = item->data(Qt::UserRole).toString();
    if (!path.isEmpty()) {
        QFile file(path);
        if (file.exists()) {
            if (!file.remove()) {
                qDebug() << "Failed to delete file:" << path;
            }
        }
    }

    // Remove only the selected item from the list
    m_eventFeed->removeItem(item);
    
    emit eventDeleted("Event");
}

void EventManager::deleteAllEvents()
{
    if (!m_eventFeed) {
        return;
    }

    // Abort all in-flight caption requests before clearing UI/items
    if (m_captionManager) {
        QListWidget* list = m_eventFeed->getList();
        if (list) {
            for (int i = 0; i < list->count(); ++i) {
                m_captionManager->cancelRequestsForItem(list->item(i));
            }
        }
    }

    // Remove files under data/captures recursively
    QDir capRoot(captureDirPath());
    if (capRoot.exists()) {
        capRoot.removeRecursively();
        // Recreate base dir to avoid issues elsewhere
        QDir().mkpath(captureDirPath());
    }

    // Clear UI
    m_eventFeed->clearEvents();
    
    emit allEventsDeleted();
}

void EventManager::clearEvents()
{
    if (m_eventFeed) {
        // Cancel pending requests as well
        if (m_captionManager) {
            QListWidget* list = m_eventFeed->getList();
            if (list) {
                for (int i = 0; i < list->count(); ++i) {
                    m_captionManager->cancelRequestsForItem(list->item(i));
                }
            }
        }
        m_eventFeed->clearEvents();
    }
}

void EventManager::requestCaption(QListWidgetItem* item, const QString& imagePath)
{
    qDebug() << "EventManager::requestCaption called with imagePath:" << imagePath;
    if (!m_captionManager) {
        qDebug() << "Caption manager not set, cannot request caption";
        return;
    }

    qDebug() << "Requesting caption for event:" << imagePath;
    m_captionManager->requestCaption(item, imagePath);
}

void EventManager::viewEvent(QListWidgetItem* item)
{
    if (!item) {
        return;
    }

    QString path = item->data(Qt::UserRole).toString();
    if (!path.isEmpty()) {
        // Open file with default application
        QProcess::startDetached("xdg-open", QStringList() << path);
    }
}

QString EventManager::getCaptureDirPath() const
{
    return captureDirPath();
}

QString EventManager::getCaptureSubdirPath(const QString& sub) const
{
    return captureSubdirPath(sub);
}

void EventManager::onCaptionReady(QListWidgetItem* item, const QString& caption)
{
    if (m_eventFeed) {
        m_eventFeed->setCaptionForItem(item, caption);
    }
    emit captionReady(item, caption);
}

void EventManager::onCaptionFailed(QListWidgetItem* item, const QString& error)
{
    qDebug() << "Caption failed for item:" << error;
    emit captionFailed(item, error);
}

void EventManager::setupEventHandling()
{
    // Setup any additional event handling logic
}

QString EventManager::captureDirPath() const
{
    // Resolve to project root's data/captures: executable is in build/, so go up one
    QDir appDir(QCoreApplication::applicationDirPath());
    appDir.cdUp();
    QString path = appDir.filePath("data/captures");
    return path;
}

QString EventManager::captureSubdirPath(const QString& sub) const
{
    QDir root(captureDirPath());
    return root.filePath(sub);
}

void EventManager::ensureCaptureDirectories()
{
    QString basePath = captureDirPath();
    QDir().mkpath(basePath);
    QDir().mkpath(captureSubdirPath("persons"));
    QDir().mkpath(captureSubdirPath("vehicles"));
}
